use crate::control::{restricted_in_user, Control, Entry, Exception, Interrupted};
use crate::decoder::{
    compact_add_constant, compact_constant5, compact_constant8, compact_memory_shape,
    compact_register, compact_stack_constant, long_roles, mpy_roles, AluOp, CmpOp, CompactOpcode,
    LogicOp, MpyOp, MvkForm, Opcode, SetClrOp, ShiftKind,
};
use crate::memory::DataClock;
use crate::sploop::{LoopBuffer, LoopStart, ReplayPackets};
use crate::system::{CoreBus, Endian, Fabric, Permission};
use crate::trace::{Event, Trace};
use crate::view::{
    address_shape, bank_index, bank_letter, bank_name, binary_operand_shape, encoded_address_bank,
    encoded_alu_unit, encoded_data_bank, marked, opposite, predicate_register, register_file_name,
    sign_extend, source_fields, spmask_mask, unit_mask_bit, Bank, DecodedOpcode, Encoding,
    Instruction, SploopDump, UnitIssue,
};
use crate::writeback::{ControlWrite, RegisterWrite, StoreWrite, WritebackQueue};
use std::collections::BTreeMap;

/// IERR bit 3, the opcode exception: an instruction word that names no C64x+ instruction, or
/// a restricted one in User mode.
const IERR_OPX: u32 = 1 << 3;
/// IERR bit 4, the resource conflict exception: two instructions in one cycle claiming the
/// same functional unit, cross path or memory path.
const IERR_RCX: u32 = 1 << 4;
/// IERR bit 5, the resource access exception: User mode reaching a control register it may
/// not touch.
const IERR_RAX: u32 = 1 << 5;
/// IERR bit 6, the privilege exception, raised alongside RAX or OPX whenever the cause was
/// User mode attempting something reserved to Supervisor (SPRU732J 8.2.4).
const IERR_PRX: u32 = 1 << 6;

const L2_DECODE_BASE: u32 = 0x0080_0000;
const L2_DECODE_SIZE: usize = 2 * 1024 * 1024;
const DECODE_PAGE_BYTES: usize = 512;
const DECODE_PAGE_SLOTS: usize = DECODE_PAGE_BYTES / 2;
const DECODE_PAGES: usize = L2_DECODE_SIZE / DECODE_PAGE_BYTES;

#[derive(Clone, Copy)]
struct DecodedEntry {
    word: u32,
    header: Option<u32>,
    instruction: Instruction,
}

#[derive(Clone, Copy)]
struct Fetched {
    /// The instruction the view presents: its encoding with the fetch-packet header pairing.
    inst: Instruction,
    /// The raw encoding and compact header matched the exact-address decode cache.
    decode_cached: bool,
    next: u32,
    parallel: bool,
    header_based: bool,
    /// The fetch packet header's PROT bit: four NOP cycles follow every load in the packet.
    prot: bool,
}

/// The two 32-bit words of a doubleword access, paired with the register pair. Little endian
/// puts the word at the lower address in the even register and big endian in the odd one
/// (SPRU732J LDDW: "In big-endian mode, the first 32-bit word in memory is loaded into the odd
/// register"); the same swap maps a pair's (even, odd) to (first, second) word in memory, so
/// one function serves loads and stores.
fn doubleword_halves(endian: Endian, first: u32, second: u32) -> (u32, u32) {
    match endian {
        Endian::Little => (first, second),
        Endian::Big => (second, first),
    }
}

/// A fetch packet's last word is a header when its top nibble is 0xE (SPRUFE8B section 3.7).
fn is_packet_header(word: u32) -> bool {
    word & 0xf000_0000 == 0xe000_0000
}

/// The header's PROT bit: four NOP cycles follow every load in the packet.
#[inline(always)]
fn packet_protected(header: u32) -> bool {
    header & (1 << 20) != 0
}

/// What a header-based fetch packet holds at one address.
///
/// Every packet-shape decision is here, and it is here because two paths make them: the fetch
/// path, reading a retained L1P line, and the debugger, reading through the host view. They used
/// to make them separately, from two copies of the same constants - the header's compact bits,
/// the compact slot index, the big-endian halfword swap, the p-bit's position - so a correction
/// to any of them had to be applied twice, and a compact-encoding defect could be visible
/// through one path and not the other. Only the byte reading differs between them now.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PacketSlot {
    /// The header word itself, which is not an instruction.
    Header,
    /// A compact instruction: read a halfword at `halfword_address`, and take its p-bit from
    /// bit `parallel_bit` of the header.
    Compact {
        halfword_address: u32,
        parallel_bit: u32,
    },
    /// A 32-bit instruction, which keeps its own p-bit in bit 0 even inside a header-based
    /// packet; only compact slots take theirs from the header.
    Wide,
}

/// What the header says sits at `address`. The caller has already established that the packet
/// containing `address` has a header.
#[inline(always)]
fn packet_slot(address: u32, header: u32, endian: Endian) -> PacketSlot {
    let offset = address & 0x1f;
    if offset == 28 {
        return PacketSlot::Header;
    }
    let word_index = offset / 4;
    if word_index < 7 && header & (1 << (21 + word_index)) != 0 {
        return PacketSlot::Compact {
            // Big-endian C6000 keeps compact slot order defined by the numeric low and high
            // halves of the containing word, so the two halfwords trade byte addresses.
            halfword_address: match endian {
                Endian::Little => address,
                Endian::Big => address ^ 2,
            },
            parallel_bit: word_index * 2 + (offset & 2) / 2,
        };
    }
    PacketSlot::Wide
}

#[derive(Clone, Copy)]
struct FetchLine {
    base: u32,
    bytes: [u8; 32],
}

impl FetchLine {
    /// The caller only ever reads within the 32-byte line (u32 reads at offsets 0-28, u16
    /// reads at 0-26), so `read_unaligned` is safe. Specializing on the endian at compile time
    /// lets the BE form compile to one load plus a byte swap; a runtime `match` on endian gets
    /// folded back into byte-by-byte assembly.
    #[inline(always)]
    fn read_u16<const BE: bool>(&self, address: u32) -> u16 {
        let offset = (address - self.base) as usize;
        let value =
            unsafe { core::ptr::read_unaligned(self.bytes.as_ptr().add(offset) as *const u16) };
        if BE {
            value.swap_bytes()
        } else {
            value
        }
    }

    #[inline(always)]
    fn read_u32<const BE: bool>(&self, address: u32) -> u32 {
        let offset = (address - self.base) as usize;
        let value =
            unsafe { core::ptr::read_unaligned(self.bytes.as_ptr().add(offset) as *const u32) };
        if BE {
            value.swap_bytes()
        } else {
            value
        }
    }
}

/// A pending branch either carries a resolved address or names the register whose value the
/// hardware samples when the branch finally redirects the fetch stream.
#[derive(Clone, Copy)]
enum BranchTarget {
    Address(u32),
}

/// How a cycle ended, which is all the cycle boundary needs to know to close it.
#[derive(Clone, Copy)]
enum CycleEnd {
    Frozen,
    Committed,
    LoopStage(u32),
}

/// The execution counters without the histogram, so a snapshot can carry them by value.
///
/// The per-address histogram behind `Stats::pc_histogram` grows with the program and is asked
/// for separately; a front end polling ten times a second must not copy a map of every address
/// the program has issued from just to show a packet count.
#[derive(Clone, Copy, Default, serde::Serialize)]
pub struct StatsCounters {
    pub packets: u64,
    pub retired: u64,
    pub replay_packets: u64,
    pub branches_taken: u64,
    pub interrupts: u64,
    pub stores: u64,
    pub faults: u64,
    pub idle_cycles: u64,
    pub stall_memory: u64,
    pub stall_nop: u64,
    pub stall_interrupt: u64,
    pub halt_cycles: u64,
    pub saturations: u64,
    pub interrupt_wait_cycles: u64,
    pub interrupt_waited: u64,
}

/// Cumulative execution counters. Cheap enough to maintain unconditionally, so every caller -
/// the runner's state dump, the debugger's `info`, the dashboard - reports the same numbers.
#[derive(Default)]
pub struct Stats {
    /// Execute packets issued from the fetch path.
    pub packets: u64,
    /// Instructions whose predicate held, so the datapath actually did their work.
    pub retired: u64,
    /// Execute packets re-issued out of the SPLOOP buffer.
    pub replay_packets: u64,
    pub branches_taken: u64,
    pub interrupts: u64,
    pub stores: u64,
    pub faults: u64,
    /// Cycles spent in IDLE waiting for an interrupt.
    pub idle_cycles: u64,
    /// Cycles held by an L1D/L1P stall.
    pub stall_memory: u64,
    /// Cycles held by a multicycle NOP.
    pub stall_nop: u64,
    /// Cycles of the interrupt pipe-down (the drain into a service routine).
    pub stall_interrupt: u64,
    /// Cycles parked on a stop address with a clean pipeline.
    pub halt_cycles: u64,
    /// Functional-unit saturation events (SSR/CSR.SAT).
    pub saturations: u64,
    /// Cycles a ready interrupt waited to be taken (SPLOOP drain or branch shadow), summed
    /// over the interrupts that had to wait.
    pub interrupt_wait_cycles: u64,
    /// How many taken interrupts had to wait for the pipeline to drain.
    pub interrupt_waited: u64,
    /// Issue counts by execute-packet address, for hotspot profiling.
    pub pc_histogram: BTreeMap<u32, u64>,
}

impl Stats {
    /// The counters without the histogram.
    pub fn counters(&self) -> StatsCounters {
        StatsCounters {
            packets: self.packets,
            retired: self.retired,
            replay_packets: self.replay_packets,
            branches_taken: self.branches_taken,
            interrupts: self.interrupts,
            stores: self.stores,
            faults: self.faults,
            idle_cycles: self.idle_cycles,
            stall_memory: self.stall_memory,
            stall_nop: self.stall_nop,
            stall_interrupt: self.stall_interrupt,
            halt_cycles: self.halt_cycles,
            saturations: self.saturations,
            interrupt_wait_cycles: self.interrupt_wait_cycles,
            interrupt_waited: self.interrupt_waited,
        }
    }
}

/// One consistent view of the CPU's state: both register files, the program counter and the
/// whole control register file. The snapshot is Copy and serializable, so a front end can
/// diff two of them or ship one over the wire. The control registers are read in ascending
/// order, which is the correct TSC latch sequence.
#[derive(Clone, Copy, serde::Serialize)]
pub struct CpuSnapshot {
    pub a: [u32; 32],
    pub b: [u32; 32],
    pub pc: u32,
    pub control: [u32; 32],
}

/// Optional high-water observation for a linker-defined stack window.
#[derive(Clone, Copy, serde::Serialize)]
pub struct StackMemoryObservation {
    pub start: u32,
    pub end: u32,
    pub initialized: bool,
    pub min_sp: u32,
    pub min_sp_cycle: u64,
    pub lowest_write: Option<u32>,
    pub lowest_write_cycle: Option<u64>,
    pub out_of_bounds_sp: bool,
    pub out_of_bounds_write: bool,
}

pub struct Cpu {
    a: [u32; 32],
    b: [u32; 32],
    pc: u32,
    cycle: u64,
    /// How many device cycles this core has asked the fabric to advance. Exactly one per step
    /// of this CPU, stalls included, which is what keeps the peripherals on the same time base
    /// as the pipeline.
    device_ticks: u64,
    /// Which megamodule this CPU is. On a single-core device it is always zero; on a C6472 it
    /// is what DNUM reports and what selects this core's memories and interrupt controller.
    core: usize,
    /// The device fabric, lent to this CPU for the cycle it is running. The device owns it
    /// between cycles and hands it to one core at a time, which is what lets six cores share
    /// one set of peripherals without any of them holding a reference to another.
    attached: Option<Box<Fabric>>,
    /// Every effect scheduled for a later cycle: register writes, stores, control-register
    /// writes and saturation flags. The queue orders them; `commit_writes` performs them.
    writeback: WritebackQueue,
    control: Control,
    /// Branches in flight. The C64x+ allows several branches to be resolving at once and
    /// each takes effect on its own cycle, so a single pending slot is not enough.
    branches: Vec<(u64, BranchTarget)>,
    branch_block_until: u64,
    interrupt_stall: u8,
    stall_cycles: u32,
    /// Registers written at the end of the previous cycle by something other than a load:
    /// reading one of these across a cross path costs a stall cycle (SPRUFE8B §3.8.4).
    previous_writes: [u32; 2],
    scheduling_load: bool,
    /// Stall cycles: unlike a multicycle NOP these freeze the whole pipeline, so no write
    /// back, branch or loop timer advances while they are counted.
    stalls: u64,
    pending_stalls: u32,
    /// Most recently fetched 32-byte packet. L1P already owns these bytes; retaining the packet
    /// avoids repeating a cache lookup and copy for every execute packet within the same line.
    packet_line: Option<FetchLine>,
    packet_line_epoch: u64,
    /// The L1P stale epoch the buffered packet was last checked against (see `Cpu::step`).
    packet_line_stale_epoch: u64,
    packet_line_reusable: bool,
    /// Set while an instruction from a PROT fetch packet executes, and `prot_load` latched when
    /// such an instruction actually performs a load.
    prot_packet: bool,
    prot_load: bool,
    /// The SPLOOP packet address of a loop that drained for a pending interrupt, consumed
    /// when that interrupt is finally taken.
    interrupted_sploop: Option<u32>,
    /// Set when a return from interrupt restores TSR with SPLX set: the SPLOOP at the return
    /// address pipes back up instead of starting a fresh loop, so the instructions in parallel
    /// with it and the SPMASKed program-memory instructions run as NOPs (SPRUFE8B 7.13.5).
    pipe_up: bool,
    /// Set by IDLE: program fetch stops until an interrupt is taken (SPRUFE8B IDLE).
    idling: bool,
    /// The mode instructions actually execute in. A context restore puts CXM back into TSR in
    /// E1, but the mode itself changes coincident with the branch target, so the delay slots
    /// of a `B IRP`/`B NRP` still run in Supervisor (SPRU732J 8.1).
    user_mode: bool,
    /// The cycle a pending `B IRP`/`B NRP` redirects the fetch stream, and the mode it
    /// installs when it does.
    mode_change: Option<(u64, bool)>,
    sploop: LoopBuffer<Fetched>,
    /// The replay buffer lent to each cycle that issues from the loop buffer, kept here so a
    /// long-running loop reuses one allocation.
    replay: ReplayPackets<Fetched>,
    /// The address of the execute packet currently issuing from program memory, which is what
    /// PCE1 reports and what an internal exception saves in NRP.
    packet_address: u32,
    /// An exception recognised this cycle, delivered by `finish_cycle` once the packet that
    /// raised it has finished. `None` while no exception is pending.
    pending_exception: Option<Exception>,
    /// The first cycle a maskable interrupt was ready to be taken but the pipeline was not
    /// (a branch shadow or an SPLOOP drain), for the interrupt-latency statistic.
    interrupt_ready_since: Option<u64>,
    /// Current external NMI pin level and its edge-latched pending state. On a multicore
    /// device the pin is driven by another core through NMIGR (SPRS612G section 3.7.1).
    nmi_line: bool,
    nmi_pending: bool,
    fault: Option<String>,
    /// The cause of the most recent internal exception the firmware was left to handle.
    last_exception_cause: Option<String>,
    stats: Stats,
    trace: Trace,
    /// Datapath footprint of the most recently dispatched instructions, for the unit view.
    last_issue: Vec<UnitIssue>,
    /// Keeping the rendered issue view allocates and formats on every issued packet. Batch
    /// execution leaves it off; interactive front ends turn it on when they display the view.
    issue_view_enabled: bool,
    /// Addresses the CPU refuses to fetch at. When the PC reaches one with a clean pipeline
    /// the CPU halts - `step` becomes a no-op - so a caller observes
    /// `ready_to_fetch_at(address)` reliably, including when a branch resolves directly onto
    /// the address, which would otherwise fetch and advance past it in the same cycle.
    stop_on_fetch: Vec<u32>,
    /// Whether the loop buffer was active at the end of the previous cycle, so the trace can
    /// report a loop starting and ending from one place rather than from every teardown site.
    sploop_was_active: bool,
    /// Whether the per-address packet histogram is maintained; see `set_profiling`.
    profiling: bool,
    /// The register-write diagnostic is configured once per CPU instead of querying the
    /// process environment on every scheduled write.
    trace_a4: bool,
    stack_observation: Option<StackMemoryObservation>,
    /// Sparse, exact-address decode cache for the 2 MiB L2 SRAM code window. Pages are
    /// allocated only when executed; fetched bytes and compact headers validate every hit.
    decoded_l2: Vec<Option<Box<[Option<DecodedEntry>]>>>,
    validated_l2: Vec<Option<Box<[u8; DECODE_PAGE_SLOTS]>>>,
}

impl Cpu {
    #[cfg(test)]
    pub fn new(pc: u32) -> Self {
        Self::new_with_endian(pc, Endian::Little)
    }

    #[cfg(test)]
    pub fn new_with_endian(pc: u32, endian: Endian) -> Self {
        Self::new_on_device(pc, endian, 0, Box::<Fabric>::default())
    }

    /// A CPU that is core `core` of the device `fabric` describes, attached to it.
    pub fn new_on_device(pc: u32, endian: Endian, core: usize, fabric: Box<Fabric>) -> Self {
        let mut fabric = fabric;
        fabric.set_endian(endian);
        let mut control = Control::default();
        control.set_big_endian(endian == Endian::Big);
        Self {
            idling: false,
            user_mode: false,
            mode_change: None,
            pipe_up: false,
            interrupted_sploop: None,
            a: [0; 32],
            b: [0; 32],
            pc,
            cycle: 0,
            core,
            device_ticks: 0,
            attached: Some(fabric),
            writeback: WritebackQueue::default(),
            control,
            branches: Vec::new(),
            branch_block_until: 0,
            interrupt_stall: 0,
            stall_cycles: 0,
            previous_writes: [0; 2],
            scheduling_load: false,
            stalls: 0,
            pending_stalls: 0,
            packet_line: None,
            packet_line_epoch: 0,
            packet_line_stale_epoch: 0,
            packet_line_reusable: false,
            prot_packet: false,
            prot_load: false,
            sploop: LoopBuffer::default(),
            replay: ReplayPackets::default(),
            stats: Stats::default(),
            trace: Trace::off(),
            last_issue: Vec::new(),
            issue_view_enabled: false,
            stop_on_fetch: Vec::new(),
            sploop_was_active: false,
            profiling: false,
            trace_a4: std::env::var_os("C64XEMU_TRACE_A4").is_some(),
            stack_observation: None,
            decoded_l2: (0..DECODE_PAGES).map(|_| None).collect(),
            validated_l2: (0..DECODE_PAGES).map(|_| None).collect(),
            packet_address: pc,
            pending_exception: None,
            interrupt_ready_since: None,
            nmi_line: false,
            nmi_pending: false,
            fault: None,
            last_exception_cause: None,
        }
    }

    pub fn set_register(&mut self, name: &str, value: u32) -> Result<(), String> {
        let (bank, index) = parse_register(name)?;
        self.write_register(bank, index, value);
        Ok(())
    }

    pub fn configure_stack_observation(&mut self, start: u32, end: u32) -> Result<(), String> {
        if start >= end || start & 3 != 0 || end & 3 != 0 {
            return Err("stack observation window must be non-empty and word aligned".into());
        }
        self.stack_observation = Some(StackMemoryObservation {
            start,
            end,
            initialized: false,
            min_sp: end,
            min_sp_cycle: 0,
            lowest_write: None,
            lowest_write_cycle: None,
            out_of_bounds_sp: false,
            out_of_bounds_write: false,
        });
        Ok(())
    }

    pub fn stack_observation(&self) -> Option<StackMemoryObservation> {
        self.stack_observation
    }

    pub fn register(&self, bank: char, index: usize) -> u32 {
        match bank {
            'A' => self.a[index],
            'B' => self.b[index],
            _ => 0,
        }
    }

    fn write_bytes_on(&mut self, bus: &mut CoreBus<'_>, address: u32, bytes: &[u8]) {
        bus.load_bytes(address, bytes);
    }

    #[cfg(test)]
    pub fn write_words(&mut self, address: u32, words: &[u32]) {
        let endian = self.fabric().endian();
        for (index, word) in words.iter().copied().enumerate() {
            let bytes = match endian {
                Endian::Little => word.to_le_bytes(),
                Endian::Big => word.to_be_bytes(),
            };
            self.write_bytes(address.wrapping_add((index as u32) * 4), &bytes);
        }
    }

    #[cfg(test)]
    pub fn run(&mut self, cycles: u64) {
        self.with_bus(|cpu, bus| {
            for _ in 0..cycles {
                cpu.step(bus);
            }
        });
    }
    #[cfg(test)]
    pub fn step_cycle(&mut self) {
        self.with_bus(|cpu, bus| cpu.step(bus));
    }

    /// One cycle against a fabric the device is holding rather than lending. The device keeps
    /// the fabric for a whole batch of device cycles and hands each running core a bus onto it,
    /// so no core has to take the fabric in and give it back every cycle.
    pub(crate) fn step_on(&mut self, fabric: &mut Fabric) {
        let mut bus = CoreBus::new(self.core, fabric);
        self.step(&mut bus);
    }

    /// Run a bounded batch while no machine-level service is due. Architectural stop-on-fetch
    /// addresses and faults are still checked every cycle; `cio` is an additional fetch boundary
    /// owned by Machine rather than by the debugger's halt list.
    pub(crate) fn run_cycles(
        &mut self,
        cycles: u64,
        cio: Option<u32>,
        ignore_cio_once: bool,
    ) -> u64 {
        self.with_bus(|cpu, bus| cpu.run_cycles_on(bus, cycles, cio, ignore_cio_once))
    }

    fn run_cycles_on(
        &mut self,
        bus: &mut CoreBus<'_>,
        cycles: u64,
        cio: Option<u32>,
        ignore_cio_once: bool,
    ) -> u64 {
        let start = self.cycle;
        let mut first = true;
        while self.cycle.wrapping_sub(start) < cycles && self.fault.is_none() {
            // The pipeline-clean test is shared by the halt, CIO-boundary and replay tests;
            // computing it once keeps the per-cycle bookkeeping to one load set.
            let clean =
                self.pending_stalls == 0 && self.stall_cycles == 0 && self.interrupt_stall == 0;
            if clean && self.stops_at(self.pc) {
                break;
            }
            if cio.is_some_and(|address| self.pc == address && clean && !self.sploop.active())
                && !(first && ignore_cio_once)
            {
                break;
            }
            let replay_only = clean && !self.idling && self.sploop.replaying_alone();
            if replay_only {
                self.execute_sploop_replay(bus);
            } else {
                self.step(bus);
            }
            first = false;
        }
        self.cycle.wrapping_sub(start)
    }

    /// Whether the halt list names this address. A batch run's list is the program's exit
    /// point alone, so the one-address case is compared inline rather than through the slice
    /// search, which is a call: this runs twice per cycle.
    #[inline(always)]
    fn stops_at(&self, pc: u32) -> bool {
        match self.stop_on_fetch.as_slice() {
            [] => false,
            [only] => *only == pc,
            addresses => addresses.contains(&pc),
        }
    }
    pub fn fault(&self) -> Option<&str> {
        self.fault.as_deref()
    }

    pub fn pc(&self) -> u32 {
        self.pc
    }
    pub fn set_pc(&mut self, pc: u32) {
        self.pc = pc & !1;
        self.packet_address = self.pc;
        self.branches.clear();
    }

    pub fn endian(&self) -> Endian {
        self.fabric().endian()
    }

    pub fn cycles(&self) -> u64 {
        self.cycle + self.stalls
    }

    /// True while the SPLOOP buffer is active (TSR.SPLX); useful for diagnosing firmware
    /// that leaves a loop mid-flight.
    pub fn loop_active(&self) -> bool {
        self.sploop.active()
    }

    pub fn ready_to_fetch_at(&self, address: u32) -> bool {
        self.pending_stalls == 0
            && self.stall_cycles == 0
            && self.interrupt_stall == 0
            && !self.sploop.active()
            && self.pc == address
    }

    /// Addresses the CPU halts at instead of fetching; see the field documentation. Replaces
    /// the whole set, so a caller keeps its own list of breakpoints.
    pub fn set_stop_on_fetch(&mut self, addresses: Vec<u32>) {
        self.stop_on_fetch = addresses;
    }

    /// True when the CPU is frozen on a stop address with a clean pipeline. Unlike
    /// `ready_to_fetch_at` this does not require the loop buffer to be empty: a breakpoint
    /// inside a software-pipelined loop legitimately halts mid-buffer.
    pub fn halted(&self) -> bool {
        self.fault.is_none()
            && self.pending_stalls == 0
            && self.stall_cycles == 0
            && self.interrupt_stall == 0
            && self.stops_at(self.pc)
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// Turns the per-address packet histogram - `Stats::pc_histogram`, the hotspot table - on.
    /// Every other counter is always maintained; this one costs a map operation per issued
    /// packet, so only a caller that displays it asks for it.
    pub fn set_profiling(&mut self, on: bool) {
        self.profiling = on;
    }

    /// Turns retention of the most recently issued datapath view on for interactive drivers.
    pub fn set_issue_view(&mut self, on: bool) {
        self.issue_view_enabled = on;
        if !on {
            self.last_issue.clear();
        }
    }

    /// Execute packets dispatched: fetch issues plus loop-buffer replays. What a debugger
    /// counts when it steps by instruction rather than by cycle.
    pub fn packets_executed(&self) -> u64 {
        self.stats.packets + self.stats.replay_packets
    }

    /// Instructions that occupied the datapath on the most recent issuing cycle, including
    /// loop-buffer replays.
    pub fn last_issue(&self) -> &[UnitIssue] {
        &self.last_issue
    }

    /// Installs a trace sink. Tracing is observationally passive: no emulator state or
    /// timing changes when it is enabled.
    pub fn set_trace(&mut self, trace: Trace) {
        self.trace = trace;
    }

    pub fn trace_mut(&mut self) -> &mut Trace {
        &mut self.trace
    }

    /// Full loop-buffer contents with pipeline offsets, when active. The buffer holds the
    /// packets; rendering them is the instruction view's job, so it is passed in.
    pub fn sploop_dump(&self) -> Option<SploopDump> {
        self.sploop.dump(|fetched| fetched.inst.text())
    }

    /// Write a CPU control register directly, as MVC would.
    pub fn write_control(&mut self, register: u8, value: u32) {
        self.control.write(register, value);
    }
    /// Read a CPU control register directly, as MVC would.
    pub fn read_control(&self, register: u8) -> u32 {
        self.control.read(register, self.packet_address)
    }
    /// One consistent view of the CPU's state, read once so every consumer sees the same
    /// cycle: both register files, the PC and the whole control register file.
    pub fn snapshot(&self) -> CpuSnapshot {
        CpuSnapshot {
            a: self.a,
            b: self.b,
            pc: self.pc,
            control: core::array::from_fn(|register| self.read_control(register as u8)),
        }
    }

    /// Drive the external NMI pin. Only a low-to-high transition is latched; a transition
    /// while EFR.NXF remains set is discarded rather than deferred.
    pub fn set_nmi_line(&mut self, asserted: bool) {
        if asserted && !self.nmi_line && self.control.nmi_edge_allowed() {
            self.nmi_pending = true;
        }
        self.nmi_line = asserted;
    }

    /// Take the fabric back from this CPU. Only the device does this: to move it to another
    /// core, or to hold it itself while it runs device cycles.
    pub fn detach(&mut self) -> Box<Fabric> {
        self.attached.take().expect("cpu has no fabric attached")
    }

    /// Give the fabric to this CPU, which then holds it until the device takes it back.
    pub fn attach(&mut self, fabric: Box<Fabric>) {
        debug_assert!(self.attached.is_none(), "cpu already has a fabric");
        self.attached = Some(fabric);
    }

    /// Which megamodule this CPU is, as DNUM reports it.
    pub fn set_core_number(&mut self, core: usize) {
        self.core = core;
        self.control.set_core_number(core as u32);
    }

    /// The fabric this CPU is holding, for the device that lent it.
    pub fn fabric_mut(&mut self) -> &mut Fabric {
        self.attached.as_mut().expect("cpu has no fabric attached")
    }

    /// This core's view of the fabric, for a front end that wants to reach the device the way
    /// this core does: memory through its caches, peripherals through its router.
    pub fn bus_mut(&mut self) -> CoreBus<'_> {
        self.bus()
    }

    /// Take the fabric out of this CPU for the duration of one operation, so the spine works
    /// against a bus resolved once rather than one resolved per access.
    fn with_bus<R>(&mut self, action: impl FnOnce(&mut Self, &mut CoreBus<'_>) -> R) -> R {
        let mut fabric = self.attached.take().expect("cpu has no fabric attached");
        let core = self.core;
        let result = {
            let mut bus = CoreBus::new(core, &mut fabric);
            action(self, &mut bus)
        };
        self.attached = Some(fabric);
        result
    }

    /// Load memory behind the caches, the way an image loader does.
    pub fn write_bytes(&mut self, address: u32, bytes: &[u8]) {
        self.with_bus(|cpu, bus| cpu.write_bytes_on(bus, address, bytes));
    }

    /// Read memory as a debugger does: through the caches, without disturbing them.
    pub fn host_write_bytes(&mut self, address: u32, bytes: &[u8]) {
        self.with_bus(|cpu, bus| cpu.host_write_bytes_on(bus, address, bytes));
    }

    /// Advance the shared peripherals by one device cycle, answering whether the device
    /// stood still: on such a cycle the interrupt lines are the ones the last full tick
    /// selected rather than freshly received events.
    fn tick_device(&mut self, bus: &mut CoreBus<'_>) -> bool {
        let tick = self.device_ticks;
        self.device_ticks += 1;
        bus.tick(tick)
    }

    /// This core's view of the fabric, which is how every access it makes is routed.
    #[inline]
    fn bus(&mut self) -> CoreBus<'_> {
        let core = self.core;
        CoreBus::new(
            core,
            self.attached.as_mut().expect("cpu has no fabric attached"),
        )
    }

    #[inline]
    fn fabric(&self) -> &Fabric {
        self.attached.as_ref().expect("cpu has no fabric attached")
    }

    /// The fabric this core is holding, for a reader that only needs to look at device state.
    #[inline]
    pub fn fabric_ref(&self) -> &Fabric {
        self.fabric()
    }

    /// The cause of the most recent internal exception left for firmware to handle, taken out.
    /// Without this, an emulator defect and a firmware bug look identical from outside: the
    /// exception is delivered, the handler prints IERR, and the reason the model raised it is
    /// lost.
    pub fn take_exception_cause(&mut self) -> Option<String> {
        self.last_exception_cause.take()
    }
    pub fn host_read_bytes(&self, address: u32, length: usize) -> Vec<u8> {
        self.fabric().host_read_bytes(self.core, address, length)
    }

    fn host_write_bytes_on(&mut self, bus: &mut CoreBus<'_>, address: u32, bytes: &[u8]) {
        bus.host_write_bytes(address, bytes);
    }

    /// Read-only mirror of `fetch`: renders the instruction at `address` and returns the
    /// address of the next one.
    ///
    /// Everything is read through the host view, so inspecting code neither fills a cache line
    /// nor advances a replacement clock. The packet shape is not mirrored but shared: both paths
    /// ask `packet_slot` what sits at the address, so the only thing that differs is where the
    /// bytes come from.
    pub fn disassemble(&self, address: u32) -> (String, u32) {
        let packet_base = address & !0x1f;
        let header = self.host_read_u32(packet_base + 28);
        if !is_packet_header(header) {
            let word = self.host_read_u32(address);
            return (
                marked(Instruction::new(address, word, None).text(), word & 1 != 0),
                address.wrapping_add(4),
            );
        }
        match packet_slot(address, header, self.fabric().endian()) {
            // The fetch path steps over the header word; a listing shows it, because a reader
            // walking a packet needs to see where it ends.
            PacketSlot::Header => (
                format!("(fetch packet header 0x{header:08x})"),
                address.wrapping_add(4),
            ),
            PacketSlot::Compact {
                halfword_address,
                parallel_bit,
            } => {
                let word = self.host_read_u16(halfword_address);
                (
                    marked(
                        Instruction::new(address, u32::from(word), Some(header)).text(),
                        header & (1 << parallel_bit) != 0,
                    ),
                    address.wrapping_add(2),
                )
            }
            PacketSlot::Wide => {
                let word = self.host_read_u32(address);
                (
                    marked(Instruction::new(address, word, None).text(), word & 1 != 0),
                    address.wrapping_add(4),
                )
            }
        }
    }

    fn host_read_u32(&self, address: u32) -> u32 {
        let bytes: [u8; 4] = self.host_read_bytes(address, 4).try_into().unwrap();
        match self.fabric().endian() {
            Endian::Little => u32::from_le_bytes(bytes),
            Endian::Big => u32::from_be_bytes(bytes),
        }
    }

    fn host_read_u16(&self, address: u32) -> u16 {
        let bytes: [u8; 2] = self.host_read_bytes(address, 2).try_into().unwrap();
        match self.fabric().endian() {
            Endian::Little => u16::from_le_bytes(bytes),
            Endian::Big => u16::from_be_bytes(bytes),
        }
    }
    fn step(&mut self, bus: &mut CoreBus<'_>) {
        if self.fault.is_some() {
            return;
        }
        if self.pending_stalls != 0 {
            self.pending_stalls -= 1;
            self.stats.stall_memory += 1;
            if self.trace.enabled() {
                self.trace
                    .emit(self.cycle, Event::Stall { reason: "memory" });
            }
            self.cycle_boundary(bus, CycleEnd::Frozen);
            return;
        }
        // IDLE holds the CPU here; the cycle boundary takes any pending interrupt, and
        // entering the service routine ends the idle.
        if self.idling {
            self.stats.idle_cycles += 1;
            self.cycle_boundary(bus, CycleEnd::Committed);
            // Entering a service routine ends the idle (`Cpu::enter`).
            return;
        }
        // Pending branches are resolved at the previous cycle boundary, so `pc` between
        // cycles always names the next packet to fetch.
        if self.interrupt_stall != 0 {
            self.interrupt_stall -= 1;
            self.stats.stall_interrupt += 1;
            if self.trace.enabled() {
                self.trace.emit(
                    self.cycle,
                    Event::Stall {
                        reason: "interrupt",
                    },
                );
            }
            // The pipeline drains into the interrupt: every register write already in flight
            // completes before the service routine runs (SPRUFE8B 5.2.2 and 7.13.4 step 7).
            self.cycle_boundary(bus, CycleEnd::Committed);
            return;
        }

        if self.stall_cycles != 0 {
            self.stall_cycles -= 1;
            self.stats.stall_nop += 1;
            if self.trace.enabled() {
                self.trace.emit(self.cycle, Event::Stall { reason: "nop" });
            }
            // A multicycle NOP in a loop body only idles the program-memory side: the loop
            // buffer keeps issuing its overlapped stages every cycle.
            if self.sploop.active() {
                self.execute_sploop_replay(bus);
                return;
            }
            self.cycle_boundary(bus, CycleEnd::Committed);
            return;
        }
        // Halt requested at this fetch boundary: leave the pipeline clean so a caller
        // observes `halted()` and `ready_to_fetch_at(address)`.
        if self.stops_at(self.pc) {
            self.stats.halt_cycles += 1;
            return;
        }
        // A reloaded nested loop replays from the buffer while the outer loop continues to
        // issue from program memory, so both sides run in the same cycle.
        if self.sploop.replaying_alone() {
            self.execute_sploop_replay(bus);
            return;
        }
        bus.begin_data_cycle();
        let packet_start = self.pc;
        self.packet_address = packet_start;
        let epoch = bus.l1p_epoch();
        if self.packet_line_epoch != epoch || !self.packet_line_reusable {
            self.packet_line = None;
            self.packet_line_epoch = epoch;
        }
        // The coherence check marked an L1P line stale. A packet L1P holds is read back through
        // it - the same bytes, so this changes nothing that executes - so the read can report;
        // one a frozen L1P let past is not held, and stays buffered exactly as it would unchecked.
        let stale_epoch = bus.l1p_stale_epoch();
        if self.packet_line_stale_epoch != stale_epoch {
            self.packet_line_stale_epoch = stale_epoch;
            if self
                .packet_line
                .as_ref()
                .is_some_and(|line| bus.l1p_holds(line.base))
            {
                self.packet_line = None;
            }
        }
        let mut packet_storage = [std::mem::MaybeUninit::<Fetched>::uninit(); 8];
        let mut length = 0usize;
        // One question about the whole packet is answered while it is fetched rather than by
        // walking it again: the first SPMASK it carries.
        let mut spmask = None;
        loop {
            // SAFETY: the slot is within the local array and uninitialised; `fetch` writes it
            // exactly once before the prefix is read as a slice below.
            let (next, parallel) =
                unsafe { self.fetch(bus, self.pc, &mut *packet_storage[length].as_mut_ptr()) };
            // SAFETY: `fetch` just initialised this slot, and `Fetched` is `Copy`.
            let fetched = unsafe { *packet_storage[length].as_ptr() };
            if spmask.is_none() {
                spmask = spmask_bits(fetched);
            }
            self.pc = next;
            length += 1;
            if !parallel || length == packet_storage.len() {
                break;
            }
        }
        // Every element in the prefix was written exactly once above; `Fetched` is Copy and
        // therefore needs no drop when the uninitialised suffix goes out of scope.
        let packet = unsafe {
            std::slice::from_raw_parts(packet_storage.as_ptr().cast::<Fetched>(), length)
        };
        let program_fetch_stalls = bus.fetch_stalls(self.cycles());
        // Cross path stalls are restricted inside the SPLOOP buffer (SPRU732J §7.16), so the
        // check only applies to packets issued from program memory.
        if !self.sploop.active() && self.cross_path_stall(&packet[..length]) {
            self.pc = packet_start;
            self.stats.stall_memory += 1;
            self.cycle_boundary(bus, CycleEnd::Frozen);
            return;
        }
        // The packet is now certain to issue: a cross path stall re-fetches it next cycle and
        // must not count twice.
        self.stats.packets += 1;
        if self.profiling {
            // A map operation per issued packet is around four per cent of a long batch run,
            // so the histogram is only kept for the callers that display it.
            *self.stats.pc_histogram.entry(packet_start).or_insert(0) += 1;
        }
        let keep_issue_view = self.issue_view_enabled || self.trace.enabled();
        if keep_issue_view {
            self.last_issue = packet[..length]
                .iter()
                .map(|fetched| fetched.inst.issue(false))
                .collect();
        }
        if self.trace.enabled() {
            let text = packet[..length]
                .iter()
                .map(|fetched| fetched.inst.text())
                .collect();
            let issues = self.last_issue.clone();
            self.trace.emit(
                self.cycle,
                Event::Packet {
                    pc: packet_start,
                    text,
                    issues,
                },
            );
        }

        let loop_offset = self.current_loop_offset();
        if let Some(offset) = loop_offset {
            self.begin_loop_cycle(offset);
        }
        // A SPMASK in the program-memory packet inhibits the buffered instruction that
        // would use the named unit this cycle (SPRU732J Table in §7.5), which is how prolog
        // setup and epilog reset code is merged into a loop without unit conflicts.
        let program_mask = spmask.unwrap_or(0);
        let mut replay_packets = ReplayPackets::default();
        if let Some(offset) = loop_offset {
            replay_packets = self.take_replay_packets(offset, program_mask);
        }
        let validation_cached = replay_packets.is_empty()
            && self.packet_validation_cached(packet_start, &packet[..length]);
        if !validation_cached {
            let validation_packets = replay_packets
                .packets()
                .chain(std::iter::once(&packet[..length]));
            if let Err((error, cause)) = validate_packets(validation_packets) {
                // The packet is annulled, but what earlier packets have in flight completes as
                // the pipeline drains into the exception (SPRU732J 6.3).
                self.raise_fault(error, cause);
                self.cycle_boundary(bus, CycleEnd::Committed);
                return;
            }
            if replay_packets.is_empty() {
                self.remember_packet_validation(packet_start, &packet[..length]);
            }
        }
        if let Some(offset) = loop_offset {
            self.buffer_loop_packet(offset, &packet[..length]);
        }
        self.control.set_sploop_executing(self.sploop.active());
        if !replay_packets.is_empty() {
            self.stats.replay_packets += replay_packets.len() as u64;
            if keep_issue_view {
                let replayed: Vec<UnitIssue> = replay_packets
                    .instructions
                    .iter()
                    .map(|fetched| fetched.inst.issue(true))
                    .collect();
                // Replayed instructions share the datapath with the freshly fetched packet this
                // cycle, so the unit view shows both.
                self.last_issue.extend(replayed.iter().cloned());
                if self.trace.enabled() {
                    self.trace.emit(
                        self.cycle,
                        Event::Replay {
                            offset: loop_offset.unwrap_or(0),
                            packets: replay_packets.len(),
                            issues: replayed,
                        },
                    );
                }
            }
        }
        for replay in replay_packets.packets() {
            self.execute_fetched(bus, replay);
        }
        if loop_offset.is_some() {
            self.return_replay_packets(replay_packets);
        }
        // A load in a fetch packet whose header sets PROT is followed by four NOP cycles,
        // which is how a short loop body absorbs the load latency without NOPs. An execute
        // packet can span fetch packets (SPRU732J §3.9.2.2: the header describes its own
        // fetch packet), so `execute_fetched` decides per instruction, from the fetch packet
        // it came out of, and the loads are recognised by the ones that actually execute, so
        // every compact and wide addressing form is covered.
        self.prot_load = false;
        // While piping up after an interrupt, the instructions in parallel with the SPLOOP and
        // the SPMASKed instructions coming from program memory execute like NOPs: the prolog
        // setup they carry already ran before the interrupt (SPRUFE8B 7.11.5 and 7.13.5).
        if self.pipe_up {
            let starts_loop = packet[..length]
                .iter()
                .any(|fetched| is_loop_start(*fetched));
            let mask = packet[..length]
                .iter()
                .find_map(|fetched| spmask_bits(*fetched))
                .unwrap_or(0);
            let program_packet: Vec<Fetched> = packet[..length]
                .iter()
                .copied()
                .filter(|fetched| {
                    if starts_loop {
                        return is_loop_start(*fetched);
                    }
                    fetched_unit_mask(*fetched).map_or(true, |unit| mask & unit == 0)
                })
                .collect();
            self.execute_fetched(bus, &program_packet);
        } else {
            self.execute_fetched(bus, packet);
        }
        if self.prot_load {
            self.stall_cycles = self.stall_cycles.max(4);
        }
        let clock = self.data_clock();
        self.pending_stalls += bus.data_stalls(clock);
        self.pending_stalls += program_fetch_stalls;
        self.cycle_boundary(
            bus,
            loop_offset.map_or(CycleEnd::Committed, CycleEnd::LoopStage),
        );
    }

    /// The conditions in SPRUFE8B 7.13.1 for a loop to start draining because an interrupt is
    /// pending: on a stage boundary, termination condition false, not loading, not reloading or
    /// waiting to reload, and at least four cycles into an SPLOOPD or SPLOOPW.
    /// Whether an interrupt may start the loop draining. The buffer owns the rule; the CPU
    /// supplies the cycle and ILC, which are its own.
    fn ready_to_drain_for_interrupt(&self) -> bool {
        self.sploop.interruptible(self.cycle, self.control.ilc())
    }

    /// Stop starting iterations so the loop drains, and remember that the drain is for an
    /// interrupt. Program fetch is already frozen while the buffer runs.
    fn begin_interrupt_drain(&mut self) {
        self.sploop.begin_interrupt_drain();
    }

    fn current_loop_offset(&self) -> Option<u32> {
        self.sploop.offset(self.cycle)
    }

    /// Store one execute packet of the loop body, dropping what the buffer must not hold: the
    /// `SPMASK` itself and the units it names, NOPs - which only occupy schedule slots, and
    /// whose multicycle form would wrongly stall the program-memory side on replay - and
    /// `SPKERNEL`, which marks the end of the body rather than being part of it.
    fn buffer_loop_packet(&mut self, offset: u32, packet: &[Fetched]) {
        if !self.sploop.loading() {
            return;
        }
        let mask = packet
            .iter()
            .find_map(|fetched| spmask_bits(*fetched))
            .unwrap_or(0);
        let instructions = packet
            .iter()
            .copied()
            .filter(|fetched| {
                if spmask_bits(*fetched).is_some() || is_nop(*fetched) {
                    return false;
                }
                if matches!(
                    fetched.inst.decoded(),
                    Some(DecodedOpcode::Wide(Opcode::Spkernel | Opcode::Spkernelr))
                ) {
                    return false;
                }
                fetched_unit_mask(*fetched).map_or(true, |unit| mask & unit == 0)
            })
            .collect();
        self.sploop.buffer(offset, instructions);
    }

    /// Borrow the CPU's replay buffer and fill it with the stages active at `offset`. The
    /// caller hands it back with `return_replay_packets`, so the allocation survives the cycle.
    /// Borrow the CPU's replay buffer and fill it with the stages active at `offset`. The
    /// caller hands it back with `return_replay_packets`, so the allocation survives the cycle.
    fn take_replay_packets(&mut self, offset: u32, mask: u8) -> ReplayPackets<Fetched> {
        let mut replay = std::mem::take(&mut self.replay);
        replay.clear();
        self.sploop.replay(offset, &mut replay, |fetched| {
            fetched_unit_mask(*fetched).is_some_and(|unit| mask & unit != 0)
        });
        replay
    }

    /// Hand the replay buffer back to the CPU, keeping its capacity for the next cycle.
    fn return_replay_packets(&mut self, replay: ReplayPackets<Fetched>) {
        self.replay = replay;
    }

    fn execute_fetched(&mut self, bus: &mut CoreBus<'_>, packet: &[Fetched]) {
        // CALLP records the address of the execute packet that follows it, which is the
        // end of this packet rather than simply the next instruction word.
        let packet_next = packet.last().map_or(0, |fetched| fetched.next);
        for fetched in packet {
            self.prot_packet = fetched.prot;
            match (fetched.inst.encoding, fetched.inst.decoded()) {
                (Encoding::Wide(word), Some(DecodedOpcode::Wide(opcode))) => {
                    if self.predicate_holds(word, opcode) {
                        self.stats.retired += 1;
                        self.execute_wide(
                            bus,
                            fetched.inst.address,
                            fetched.header_based,
                            packet_next,
                            word,
                            opcode,
                        );
                    } else if let Some(count) = multicycle_nop_count(word, opcode) {
                        // BNOP's generated idle cycles do not depend on its predicate.
                        self.stall_cycles = self.stall_cycles.max(count);
                    }
                }
                (Encoding::Compact(word, header), Some(DecodedOpcode::Compact(family))) => {
                    self.stats.retired += 1;
                    self.execute_compact(
                        bus,
                        fetched.inst.address,
                        packet_next,
                        word,
                        header,
                        family,
                    )
                }
                (Encoding::Wide(word), None) => {
                    self.stats.retired += 1;
                    self.fault_invalid_opcode(fetched.inst.address, word);
                }
                (Encoding::Compact(word, _), None) => {
                    self.stats.retired += 1;
                    self.fault_invalid_compact(fetched.inst.address, u32::from(word));
                }
                _ => unreachable!("encoding and cached opcode disagree"),
            }
        }
        self.prot_packet = false;
    }

    /// A cycle that issues from the loop buffer alone. Its loads and stores reach L1D like any
    /// other packet's, so it pays the same bank-conflict and read-miss stalls: SPRU732J §4.3.3
    /// gives the memory stall no exception, and §7.15.2 names the only two stalls the loop
    /// buffer is exempt from - the cross path read and the AMR use. What the buffer saves is
    /// program fetch (SPRU862B §3.4.3.1), not data access.
    fn execute_sploop_replay(&mut self, bus: &mut CoreBus<'_>) {
        let offset = self.current_loop_offset().unwrap();
        bus.begin_data_cycle();
        self.begin_loop_cycle(offset);
        let replay_packets = self.take_replay_packets(offset, 0);
        if let Err((error, cause)) = validate_packets(replay_packets.packets()) {
            // As for a program packet: the replay is annulled, the writes in flight land.
            self.raise_fault(error, cause);
            self.cycle_boundary(bus, CycleEnd::Committed);
            return;
        }
        if !replay_packets.is_empty() {
            self.stats.replay_packets += replay_packets.len() as u64;
            if self.issue_view_enabled || self.trace.enabled() {
                let replayed: Vec<UnitIssue> = replay_packets
                    .instructions
                    .iter()
                    .map(|fetched| fetched.inst.issue(true))
                    .collect();
                self.last_issue = replayed.clone();
                if self.trace.enabled() {
                    self.trace.emit(
                        self.cycle,
                        Event::Replay {
                            offset,
                            packets: replay_packets.len(),
                            issues: replayed,
                        },
                    );
                }
            }
        }
        for replay in replay_packets.packets() {
            self.execute_fetched(bus, replay);
        }
        self.return_replay_packets(replay_packets);
        let clock = self.data_clock();
        self.pending_stalls += bus.data_stalls(clock);
        self.cycle_boundary(bus, CycleEnd::LoopStage(offset));
    }

    /// Stage-boundary bookkeeping, evaluated before the cycle's packets execute so that the
    /// iteration that would run past the end of the loop is never started.
    /// Stage-boundary bookkeeping. The buffer decides what the boundary does; the CPU reads the
    /// registers it needs, because the register file and ILC are the CPU's.
    fn begin_loop_cycle(&mut self, offset: u32) {
        let while_value = self
            .sploop
            .while_register()
            .map(|(bank, register, _)| self.read_register(bank, register));
        let outcome = self
            .sploop
            .begin_stage(offset, self.control.ilc(), while_value);
        if let Some(ilc) = outcome.set_ilc {
            self.control.set_ilc(ilc);
        }
    }

    /// `SPMASKR`/`SPKERNELR` marks the reload point of a nested loop: when the current
    /// invocation finishes draining, ILC is reloaded from RILC and the buffer replays while
    /// program memory carries on with the outer loop (SPRU732J §7.6).
    ///
    /// Whether it reloads at all is decided by the register the `SPLOOP(D)` named in its
    /// would-be predicate field, read at the reload point. The compiler pairs that register
    /// with the outer loop's own branch - `SPMASKR || [B0] BNOP outer` is one execute packet -
    /// so the buffer replays exactly when the outer loop takes another pass, and the last
    /// pass lets it go idle instead of running one inner loop too many.
    /// `SPMASKR`/`SPKERNELR` marks the reload point of a nested loop. The reload selector is a
    /// register, so the CPU evaluates the test and the buffer decides what to do about it.
    fn arm_reload(&mut self) {
        let armed = self
            .sploop
            .reload_register()
            .is_some_and(|(bank, register, zero)| {
                (self.read_register(bank, register) == 0) == zero
            });
        let outcome = self
            .sploop
            .arm_reload(self.cycle, self.control.rilc(), armed);
        if let Some(ilc) = outcome.set_ilc {
            self.control.set_ilc(ilc);
        }
    }

    /// End-of-cycle loop bookkeeping. A loop that drained for an interrupt resumes at its own
    /// `SPLOOP` packet so it pipes back up (SPRUFE8B 7.13.4).
    fn complete_loop_cycle(&mut self, offset: u32) {
        let outcome = self
            .sploop
            .complete_stage(offset, self.cycle, self.control.rilc());
        if let Some(ilc) = outcome.set_ilc {
            self.control.set_ilc(ilc);
        }
        if let Some(address) = outcome.restart {
            self.pc = address;
            self.interrupted_sploop = Some(address);
        }
    }

    fn validation_slot(address: u32) -> Option<(usize, usize)> {
        let offset = address.checked_sub(L2_DECODE_BASE)? as usize;
        (offset < L2_DECODE_SIZE)
            .then_some((offset / DECODE_PAGE_BYTES, (offset % DECODE_PAGE_BYTES) / 2))
    }

    fn packet_validation_cached(&self, address: u32, packet: &[Fetched]) -> bool {
        if !packet.iter().all(|fetched| fetched.decode_cached) {
            return false;
        }
        let Some((page, slot)) = Self::validation_slot(address) else {
            return false;
        };
        self.validated_l2[page]
            .as_ref()
            .is_some_and(|entries| entries[slot] == packet.len() as u8)
    }

    fn remember_packet_validation(&mut self, address: u32, packet: &[Fetched]) {
        let Some((page, slot)) = Self::validation_slot(address) else {
            return;
        };
        let entries =
            self.validated_l2[page].get_or_insert_with(|| Box::new([0; DECODE_PAGE_SLOTS]));
        entries[slot] = packet.len() as u8;
    }

    /// Decode an instruction from scratch. Kept out of `decoded_instruction` so the hot
    /// cache-hit path does not carry `Instruction::new`'s decode workspace on its stack frame.
    #[inline(never)]
    fn decode_new(address: u32, word: u32, header: Option<u32>) -> Instruction {
        Instruction::new(address, word, header)
    }

    /// Decode one instruction into `inst`, returning whether it came from the exact-address
    /// decode cache. Writing into the caller's slot keeps the 24-byte instruction from
    /// round-tripping through a return-value buffer on the hot issue path.
    fn decoded_instruction(
        &mut self,
        address: u32,
        word: u32,
        header: Option<u32>,
        inst: &mut Instruction,
    ) -> bool {
        let Some(offset) = address
            .checked_sub(L2_DECODE_BASE)
            .map(|offset| offset as usize)
            .filter(|offset| *offset < L2_DECODE_SIZE)
        else {
            *inst = Self::decode_new(address, word, header);
            return false;
        };
        let page_index = offset / DECODE_PAGE_BYTES;
        let slot = (offset % DECODE_PAGE_BYTES) / 2;
        if let Some(page) = self.decoded_l2[page_index].as_ref() {
            if let Some(entry) = page[slot] {
                if entry.word == word && entry.header == header {
                    *inst = entry.instruction;
                    return true;
                }
            }
        }
        let instruction = Self::decode_new(address, word, header);
        // `vec![None; N]` fills the boxed page in place; `Box::new([None; N])` would
        // materialize the 11 KB array on the stack first, bloating this hot function's frame.
        let page = self.decoded_l2[page_index]
            .get_or_insert_with(|| vec![None; DECODE_PAGE_SLOTS].into_boxed_slice());
        page[slot] = Some(DecodedEntry {
            word,
            header,
            instruction,
        });
        *inst = instruction;
        false
    }

    /// Fetch one instruction into `slot` and return the address of the next one and whether it
    /// is parallel. Building the entry directly in the caller's packet slot avoids a
    /// return-value spill on the hot issue path.
    fn fetch(
        &mut self,
        bus: &mut CoreBus<'_>,
        mut address: u32,
        slot: &mut Fetched,
    ) -> (u32, bool) {
        // A page that does not grant execute delivers no instruction: the packet reads as NOP
        // and the unit's fault event turns into EXCEP at the end of the cycle.
        let user = self.user_mode;
        if !bus.check_access(address, Permission::Execute, user) {
            let decode_cached = self.decoded_instruction(address, 0, None, &mut slot.inst);
            slot.decode_cached = decode_cached;
            slot.next = address.wrapping_add(4);
            slot.parallel = false;
            slot.header_based = false;
            slot.prot = false;
            return (slot.next, slot.parallel);
        }
        loop {
            let packet_base = address & !0x1f;
            if self
                .packet_line
                .as_ref()
                .map_or(true, |line| line.base != packet_base)
            {
                self.packet_line = Some(FetchLine {
                    base: packet_base,
                    bytes: bus.fetch_packet(packet_base),
                });
                self.packet_line_reusable = bus.fetch_reusable(packet_base);
            }
            let line = self.packet_line.as_ref().unwrap();
            let endian = bus.endian();
            let read32 = |address: u32| match endian {
                Endian::Big => line.read_u32::<true>(address),
                Endian::Little => line.read_u32::<false>(address),
            };
            let read16 = |address: u32| match endian {
                Endian::Big => line.read_u16::<true>(address),
                Endian::Little => line.read_u16::<false>(address),
            };
            let header = read32(packet_base + 28);
            if !is_packet_header(header) {
                let word = read32(address);
                let decode_cached = self.decoded_instruction(address, word, None, &mut slot.inst);
                slot.decode_cached = decode_cached;
                slot.next = address.wrapping_add(4);
                slot.parallel = word & 1 != 0;
                slot.header_based = false;
                slot.prot = false;
                return (slot.next, slot.parallel);
            }
            match packet_slot(address, header, endian) {
                PacketSlot::Header => {
                    address = packet_base.wrapping_add(32);
                    continue;
                }
                PacketSlot::Compact {
                    halfword_address,
                    parallel_bit,
                } => {
                    let word = read16(halfword_address);
                    let decode_cached = self.decoded_instruction(
                        address,
                        u32::from(word),
                        Some(header),
                        &mut slot.inst,
                    );
                    slot.decode_cached = decode_cached;
                    slot.next = address.wrapping_add(2);
                    slot.parallel = header & (1 << parallel_bit) != 0;
                    slot.header_based = true;
                    slot.prot = packet_protected(header);
                }
                PacketSlot::Wide => {
                    let word = read32(address);
                    let decode_cached =
                        self.decoded_instruction(address, word, None, &mut slot.inst);
                    slot.decode_cached = decode_cached;
                    slot.next = address.wrapping_add(4);
                    slot.parallel = word & 1 != 0;
                    slot.header_based = true;
                    slot.prot = packet_protected(header);
                }
            }
            return (slot.next, slot.parallel);
        }
    }

    /// The IDLE permission check, kept cold so the "IDLE" name string is not loaded on the
    /// hot issue path for an instruction the firmware almost never executes.
    #[cold]
    #[inline(never)]
    fn idle_allowed(&mut self, address: u32) -> bool {
        !self.restricted_instruction(address, "IDLE")
    }

    /// Opcode-exception faults, kept cold so the compiler does not hoist their `format!`
    /// templates onto the hot instruction dispatch (the same hoist the duplicate-write and
    /// IDLE helpers remove).
    #[cold]
    #[inline(never)]
    fn fault_unsupported_unit(
        &mut self,
        word: u32,
        unit: &'static str,
        operation: &'static str,
        op: u32,
    ) {
        self.raise_fault(
            format!("unsupported {unit} unit {operation} {op:#04x}: {word:#010x}"),
            1 << 3,
        );
    }

    #[cold]
    #[inline(never)]
    fn fault_unsupported_opcode(&mut self, word: u32) {
        self.raise_fault(format!("unsupported M unit opcode: {word:#010x}"), 1 << 3);
    }

    #[cold]
    #[inline(never)]
    fn fault_invalid_opcode(&mut self, address: u32, word: u32) {
        self.raise_fault(
            format!("invalid opcode at {address:#010x}: {word:#010x}"),
            1 << 3,
        );
    }

    #[cold]
    #[inline(never)]
    fn fault_invalid_compact(&mut self, address: u32, word: u32) {
        self.raise_fault(
            format!("invalid compact opcode at {address:#010x}: {word:#06x}"),
            1 << 3,
        );
    }

    fn execute_wide(
        &mut self,
        bus: &mut CoreBus<'_>,
        address: u32,
        header_based: bool,
        packet_next: u32,
        word: u32,
        opcode: Opcode,
    ) {
        let side = if word & 2 == 0 { Bank::A } else { Bank::B };
        match opcode {
            Opcode::Nop => {
                let count = ((word >> 13) & 0xf) + 1;
                self.stall_cycles = self.stall_cycles.max(count - 1);
            }
            Opcode::Idle => {
                if self.idle_allowed(address) {
                    self.idling = true;
                }
            }
            Opcode::MvkShort => {
                let constant = sign_extend((word >> 18) & 0x1f, 5);
                self.schedule(side, ((word >> 23) & 0x1f) as usize, constant, 0);
            }
            Opcode::MvkLong => {
                let constant = sign_extend((word >> 7) & 0xffff, 16);
                self.schedule(side, ((word >> 23) & 0x1f) as usize, constant, 0);
            }
            Opcode::Mvkh => {
                let dst = ((word >> 23) & 0x1f) as usize;
                let value =
                    (self.read_register(side, dst) & 0xffff) | (((word >> 7) & 0xffff) << 16);
                self.schedule(side, dst, value, 0);
            }
            opcode @ (Opcode::Add
            | Opcode::Sub
            | Opcode::And
            | Opcode::Or
            | Opcode::Xor
            | Opcode::Andn
            | Opcode::CmpEq
            | Opcode::CmpGt
            | Opcode::CmpGtu
            | Opcode::CmpLt
            | Opcode::CmpLtu
            | Opcode::Add2
            | Opcode::Sub2
            | Opcode::Pack2
            | Opcode::Packl4
            | Opcode::Packh4
            | Opcode::Add4
            | Opcode::Lmbd
            | Opcode::Subc
            | Opcode::Shlmb
            | Opcode::Shrmb
            | Opcode::Sub4) => {
                let (left, right, dst) = binary_operands(word, side, &self.a, &self.b);
                let value = match opcode {
                    Opcode::Add => left.wrapping_add(right),
                    // A plain two's-complement subtraction, 0 - INT_MIN included. NEG
                    // is this word with a zero constant (SPRU189F NEG: "the assembler
                    // uses the operation SUB 0, src, dst"; .L opfield 0000110, the SUB
                    // form whose execution is src1 - src2). The NEG page's own
                    // execution line reads "0 -s src2", the saturating operator, but
                    // one encoding has one behaviour, and TI's compiler settles which:
                    // cl6x negates an unsigned int with NEG, where C defines
                    // -0x80000000u as 0x80000000u.
                    Opcode::Sub => left.wrapping_sub(right),
                    Opcode::And => left & right,
                    Opcode::Or => left | right,
                    Opcode::Xor => left ^ right,
                    Opcode::Andn => left & !right,
                    Opcode::CmpEq => u32::from(left == right),
                    Opcode::CmpGt => u32::from((left as i32) > (right as i32)),
                    Opcode::CmpGtu => u32::from(left > right),
                    Opcode::CmpLt => u32::from((left as i32) < (right as i32)),
                    Opcode::CmpLtu => u32::from(left < right),
                    Opcode::Add2 => packed_16(left, right, u16::wrapping_add),
                    Opcode::Sub2 => packed_16(left, right, u16::wrapping_sub),
                    Opcode::Add4 => packed_8(left, right, u8::wrapping_add),
                    Opcode::Sub4 => packed_8(left, right, u8::wrapping_sub),
                    Opcode::Pack2 => ((left & 0xffff) << 16) | (right & 0xffff),
                    Opcode::Lmbd => {
                        // Count the bits above the leftmost bit of src2 that equals src1's LSB.
                        let searched = left & 1;
                        let inverted = if searched == 0 { !right } else { right };
                        inverted.leading_zeros()
                    }
                    Opcode::Subc => {
                        if left >= right {
                            ((left - right) << 1) | 1
                        } else {
                            left << 1
                        }
                    }
                    Opcode::Shlmb => (right << 8) | (left >> 24),
                    Opcode::Shrmb => (right >> 8) | (left << 24),
                    Opcode::Packl4 => {
                        ((left & 0x00ff_0000) << 8)
                            | ((left & 0x0000_00ff) << 16)
                            | ((right & 0x00ff_0000) >> 8)
                            | (right & 0x0000_00ff)
                    }
                    Opcode::Packh4 => {
                        (left & 0xff00_0000)
                            | ((left & 0x0000_ff00) << 8)
                            | ((right & 0xff00_0000) >> 16)
                            | ((right & 0x0000_ff00) >> 8)
                    }
                    _ => unreachable!(),
                };
                self.schedule(side, dst, value, 0);
            }
            Opcode::LongArith => {
                let mut long_saturated = false;
                let op = (word >> 5) & 0x7f;
                let roles = long_roles(word);
                let fields = source_fields(word);
                let (src1_field, src2_field, dst) = (fields.src1, fields.src2, fields.dst);
                let crossed = if fields.cross { opposite(side) } else { side };
                let constant = || {
                    debug_assert!(roles.src1_constant);
                    sign_extend(src1_field as u32, 5) as i32 as i64
                };
                // A long operand ignores the 24 MSBs of the odd register; a long result
                // zero-fills them (SPRUFE8B Figure 2-2).
                let long_at = |bank: Bank| -> i64 {
                    debug_assert!(roles.src2_pair);
                    let low = self.read_register(bank, src2_field) as i64;
                    let high = (self.read_register(bank, src2_field + 1) & 0xff) as i64;
                    let value = (high << 32) | low;
                    (value << 24) >> 24
                };
                let ulong_at = |bank: Bank| -> i64 {
                    debug_assert!(roles.src2_pair);
                    let low = self.read_register(bank, src2_field) as i64;
                    let high = (self.read_register(bank, src2_field + 1) & 0xff) as i64;
                    (high << 32) | low
                };
                let int_src1 = self.read_register(side, src1_field);
                let int_src1_crossed = self.read_register(crossed, src1_field);
                let int_src2 = self.read_register(side, src2_field);
                let int_src2_crossed = self.read_register(crossed, src2_field);
                const LONG_MIN: i64 = -(1 << 39);
                const LONG_MAX: i64 = (1 << 39) - 1;
                let value = match op {
                    0x20 => long_at(side).wrapping_add(constant()),
                    0x21 => long_at(side).wrapping_add(i64::from(int_src1_crossed as i32)),
                    0x23 => i64::from(int_src1 as i32) + i64::from(int_src2_crossed as i32),
                    0x27 => i64::from(int_src1 as i32) - i64::from(int_src2_crossed as i32),
                    0x29 => ulong_at(side).wrapping_add(i64::from(int_src1_crossed)),
                    0x2b => i64::from(int_src1) + i64::from(int_src2_crossed),
                    // SSUB with scst5 src1 onto a 40-bit long: saturates at the long
                    // bounds (SPRU732J, opfield 010 1100).
                    0x2c => {
                        let wide = constant().wrapping_sub(long_at(side));
                        let clamped = wide.clamp(LONG_MIN, LONG_MAX);
                        long_saturated = clamped != wide;
                        clamped
                    }
                    // SUB with scst5 src1, of which NEG is the constant-zero form TI's tools
                    // print: `SUB .L1 0,A1:A0,A3:A2` and `NEG .L1 A1:A0,A3:A2` are one word.
                    // Its execution is defined as the saturating cst -s src2 (SPRU732J), so
                    // 0 - LONG_MIN clamps to LONG_MAX; compiled saturating long subtraction
                    // depends on that. The constant is not what makes it saturate, so every
                    // constant takes this path - `cl6x` assembles a nonzero one
                    // (`SUB .L1 5,A1:A0,A3:A2` is 0100a498h) and it used to raise the
                    // unsupported-operation fault.
                    0x24 => {
                        let wide = constant().wrapping_sub(long_at(side));
                        let clamped = wide.clamp(LONG_MIN, LONG_MAX);
                        long_saturated = clamped != wide;
                        clamped
                    }
                    0x2f => i64::from(int_src1) - i64::from(int_src2_crossed),
                    0x30 => {
                        let wide = long_at(side).wrapping_add(constant());
                        let clamped = wide.clamp(LONG_MIN, LONG_MAX);
                        long_saturated = clamped != wide;
                        clamped
                    }
                    0x31 => {
                        let wide = long_at(side).wrapping_add(i64::from(int_src1_crossed as i32));
                        let clamped = wide.clamp(LONG_MIN, LONG_MAX);
                        long_saturated = clamped != wide;
                        clamped
                    }
                    0x37 => i64::from(int_src1_crossed as i32) - i64::from(int_src2 as i32),
                    0x38 => {
                        let value = long_at(side);
                        let clamped = value.saturating_abs().clamp(LONG_MIN, LONG_MAX);
                        long_saturated = value == LONG_MIN;
                        clamped
                    }
                    0x3f => i64::from(int_src1_crossed) - i64::from(int_src2),
                    0x40 => {
                        let value = long_at(side);
                        let clamped = value.clamp(i64::from(i32::MIN), i64::from(i32::MAX));
                        self.schedule(side, dst, clamped as u32, 0);
                        if clamped != value {
                            self.note_saturation(b'L', side);
                        }
                        return;
                    }
                    0x60 => {
                        // NORM counts the redundant sign bits of the 40-bit value. A zero
                        // operand has no bit to stop the count, so it saturates at the
                        // datapath width minus one - 39, exactly as the 32-bit form returns
                        // 31 for zero.
                        let value = long_at(side);
                        let shifted = value << 24;
                        let normalized = shifted
                            .leading_zeros()
                            .max((!shifted).leading_zeros())
                            .min(40);
                        self.schedule(side, dst, normalized.saturating_sub(1), 0);
                        return;
                    }
                    // Comparisons whose src2 is a 40-bit long: the constant forms take a
                    // five-bit constant on src1 and the register forms take src1 across the
                    // cross path. Each writes a single-register 0 or 1.
                    0x44 | 0x45 | 0x4c | 0x4d | 0x50 | 0x51 | 0x54 | 0x55 | 0x5c | 0x5d => {
                        debug_assert!(!roles.dst_pair);
                        let unsigned_constant = i64::from(src1_field as u32);
                        let result = match op {
                            0x44 => constant() > long_at(side),
                            0x45 => i64::from(int_src1_crossed as i32) > long_at(side),
                            0x4c => unsigned_constant > ulong_at(side),
                            0x4d => i64::from(int_src1_crossed) > ulong_at(side),
                            0x50 => constant() == long_at(side),
                            0x51 => i64::from(int_src1_crossed as i32) == long_at(side),
                            0x54 => long_at(side) > constant(),
                            0x55 => long_at(side) > i64::from(int_src1_crossed as i32),
                            0x5c => unsigned_constant < ulong_at(side),
                            0x5d => i64::from(int_src1_crossed) < ulong_at(side),
                            _ => unreachable!(),
                        };
                        self.schedule(side, dst, u32::from(result), 0);
                        return;
                    }
                    0x63 => {
                        // NORM on a 32-bit source counts its redundant sign bits.
                        let value = int_src2_crossed as i32;
                        let normalized = if value < 0 {
                            (!value).leading_zeros()
                        } else {
                            value.leading_zeros()
                        };
                        self.schedule(side, dst, normalized.saturating_sub(1), 0);
                        return;
                    }
                    _ => {
                        self.fault_unsupported_unit(word, "L", "long operation", op);
                        return;
                    }
                };
                debug_assert!(roles.dst_pair);
                self.schedule(side, dst, value as u32, 0);
                self.schedule(side, dst + 1, ((value >> 32) & 0xff) as u32, 0);
                if long_saturated {
                    self.note_saturation(b'L', side);
                }
            }
            opcode @ (Opcode::Max2
            | Opcode::Min2
            | Opcode::Maxu4
            | Opcode::Minu4
            | Opcode::Packh2
            | Opcode::Packlh2
            | Opcode::Packhl2
            | Opcode::Shr2
            | Opcode::Shru2
            | Opcode::Sadd2
            | Opcode::Ssub2
            | Opcode::Spack2
            | Opcode::Spacku4
            | Opcode::Saddu4
            | Opcode::Saddus2
            | Opcode::Rpack2
            | Opcode::Sadd
            | Opcode::Ssub
            | Opcode::Sshl
            | Opcode::CmpEq2
            | Opcode::CmpGt2
            | Opcode::CmpEq4
            | Opcode::CmpGtu4
            | Opcode::Subabs4) => {
                let (left, right, dst) = binary_operands(word, side, &self.a, &self.b);
                let mut saturated = false;
                let value = match opcode {
                    Opcode::Max2 => {
                        packed_16(
                            left,
                            right,
                            |l, r| {
                                if (l as i16) >= (r as i16) {
                                    l
                                } else {
                                    r
                                }
                            },
                        )
                    }
                    Opcode::Min2 => {
                        packed_16(
                            left,
                            right,
                            |l, r| {
                                if (l as i16) <= (r as i16) {
                                    l
                                } else {
                                    r
                                }
                            },
                        )
                    }
                    Opcode::Maxu4 => packed_8(left, right, |l, r| l.max(r)),
                    Opcode::Minu4 => packed_8(left, right, |l, r| l.min(r)),
                    Opcode::Packh2 => ((left >> 16) << 16) | (right >> 16),
                    Opcode::Packlh2 => ((left & 0xffff) << 16) | (right >> 16),
                    Opcode::Packhl2 => ((left >> 16) << 16) | (right & 0xffff),
                    Opcode::Shr2 => {
                        let count = (left & 0x1f).min(16);
                        let shift = |half: u32| ((((half as u16) as i16) as i32) >> count) as u32;
                        (shift(right >> 16) << 16 & 0xffff_0000) | (shift(right) & 0xffff)
                    }
                    Opcode::Shru2 => {
                        let count = (left & 0x1f).min(16);
                        (((right >> 16) >> count) << 16) | ((right & 0xffff) >> count)
                    }
                    Opcode::Sadd2 => {
                        let (value, lanes) = packed_16_sat(left, right, |l, r| {
                            saturate_i16((l as i16 as i32) + (r as i16 as i32))
                        });
                        saturated = lanes;
                        value
                    }
                    Opcode::Ssub2 => {
                        let (value, lanes) = packed_16_sat(left, right, |l, r| {
                            saturate_i16((l as i16 as i32) - (r as i16 as i32))
                        });
                        saturated = lanes;
                        value
                    }
                    Opcode::Spack2 => {
                        let (high, high_sat) = saturate_i16(left as i32);
                        let (low, low_sat) = saturate_i16(right as i32);
                        saturated = high_sat || low_sat;
                        (u32::from(high) << 16) | u32::from(low)
                    }
                    // Four signed halfwords saturated to unsigned bytes, src1 supplying the
                    // upper pair.
                    Opcode::Spacku4 => {
                        let byte = |half: u32| {
                            let value = half as u16 as i16 as i32;
                            let clamped = value.clamp(0, 255);
                            (clamped as u32, clamped != value)
                        };
                        let lanes = [byte(left >> 16), byte(left), byte(right >> 16), byte(right)];
                        saturated = lanes.iter().any(|(_, lane_sat)| *lane_sat);
                        (lanes[0].0 << 24) | (lanes[1].0 << 16) | (lanes[2].0 << 8) | lanes[3].0
                    }
                    Opcode::Sadd => {
                        let (value, clamped) =
                            saturate_i32(i64::from(left as i32) + i64::from(right as i32));
                        saturated = clamped;
                        value
                    }
                    Opcode::Ssub => {
                        let (value, clamped) =
                            saturate_i32(i64::from(left as i32) - i64::from(right as i32));
                        saturated = clamped;
                        value
                    }
                    // The register form takes six count bits; a count past 31 saturates on the
                    // sign of the shifted value.
                    Opcode::Sshl => {
                        let count = if (word >> 6) & 0x3f == 0x22 {
                            left & 0x1f
                        } else {
                            left & 0x3f
                        };
                        let value = right as i32;
                        if count >= 32 {
                            saturated = value != 0;
                            match value.signum() {
                                0 => 0,
                                1 => i32::MAX as u32,
                                _ => i32::MIN as u32,
                            }
                        } else {
                            let (value, clamped) = saturate_i32(i64::from(value) << count);
                            saturated = clamped;
                            value
                        }
                    }
                    Opcode::CmpEq2 => {
                        u32::from(left >> 16 == right >> 16) << 1
                            | u32::from(left & 0xffff == right & 0xffff)
                    }
                    Opcode::CmpGt2 => {
                        u32::from((left >> 16) as u16 as i16 > (right >> 16) as u16 as i16) << 1
                            | u32::from(left as u16 as i16 > right as u16 as i16)
                    }
                    Opcode::CmpEq4 | Opcode::CmpGtu4 => {
                        let mut value = 0;
                        for lane in 0..4 {
                            let l = (left >> (lane * 8)) & 0xff;
                            let r = (right >> (lane * 8)) & 0xff;
                            let bit = if matches!(opcode, Opcode::CmpEq4) {
                                l == r
                            } else {
                                l > r
                            };
                            value |= u32::from(bit) << lane;
                        }
                        value
                    }
                    Opcode::Saddu4 => {
                        let (value, lanes) = packed_8_sat(left, right, |l, r| {
                            let sum = u32::from(l) + u32::from(r);
                            (sum.min(255) as u8, sum > 255)
                        });
                        saturated = lanes;
                        value
                    }
                    // src1 is unsigned and src2 signed; the result saturates to unsigned.
                    Opcode::Saddus2 => {
                        let (value, lanes) = packed_16_sat(left, right, |l, r| {
                            saturate_u16((l as i32) + (r as i16 as i32))
                        });
                        saturated = lanes;
                        value
                    }
                    // Both sources shift left by one with 32-bit saturation and contribute
                    // their upper halfword, src1 to the upper half of the result.
                    Opcode::Rpack2 => {
                        let shifted = |value: u32| saturate_i32(i64::from(value as i32) << 1);
                        let (high, high_sat) = shifted(left);
                        let (low, low_sat) = shifted(right);
                        saturated = high_sat || low_sat;
                        (high & 0xffff_0000) | (low >> 16)
                    }
                    _ => packed_8(left, right, |l, r| l.abs_diff(r)),
                };
                self.schedule(side, dst, value, 0);
                if saturated {
                    self.note_saturation(encoded_alu_unit(word), side);
                }
            }
            // The L-unit pair-writing family: ADDSUB and ADDSUB2 with their saturating forms,
            // the two packing pairs and the three-way shuffle (SPRUFE8B, all unconditional).
            opcode @ (Opcode::AddSub | Opcode::DPack | Opcode::Shfl3) => {
                let (left, right, dst) = binary_operands(word, side, &self.a, &self.b);
                let op = (word >> 5) & 0x7f;
                let mut saturated = false;
                let (even, odd) = match opcode {
                    Opcode::AddSub => match op {
                        // ADDSUB: the sum goes to the odd register, the difference to the even.
                        0x0c => (left.wrapping_sub(right), left.wrapping_add(right)),
                        0x0d => (
                            packed_16(left, right, u16::wrapping_sub),
                            packed_16(left, right, u16::wrapping_add),
                        ),
                        0x0e => {
                            let (difference, low_sat) =
                                saturate_i32(i64::from(left as i32) - i64::from(right as i32));
                            let (sum, high_sat) =
                                saturate_i32(i64::from(left as i32) + i64::from(right as i32));
                            saturated = low_sat || high_sat;
                            (difference, sum)
                        }
                        _ => {
                            let (difference, low_sat) = packed_16_sat(left, right, |l, r| {
                                saturate_i16((l as i16 as i32) - (r as i16 as i32))
                            });
                            let (sum, high_sat) = packed_16_sat(left, right, |l, r| {
                                saturate_i16((l as i16 as i32) + (r as i16 as i32))
                            });
                            saturated = low_sat || high_sat;
                            (difference, sum)
                        }
                    },
                    // DPACK2 pairs PACK2 with PACKH2; DPACKX2 pairs the two PACKLH2 halves.
                    Opcode::DPack if op == 0x34 => (
                        ((left & 0xffff) << 16) | (right & 0xffff),
                        (left & 0xffff_0000) | (right >> 16),
                    ),
                    Opcode::DPack => (
                        ((left & 0xffff) << 16) | (right >> 16),
                        ((right & 0xffff) << 16) | (left >> 16),
                    ),
                    // SHFL3 interleaves three halfwords into 48 bits: src2's low halfword, then
                    // src1's low and high halfwords, one bit each.
                    _ => {
                        let inputs = [right & 0xffff, left & 0xffff, (left >> 16) & 0xffff];
                        let mut result = 0u64;
                        for bit in 0..16 {
                            for (index, input) in inputs.iter().enumerate() {
                                result |= u64::from((input >> bit) & 1) << (bit * 3 + index);
                            }
                        }
                        (result as u32, (result >> 32) as u32)
                    }
                };
                self.schedule(side, dst, even, 0);
                self.schedule(side, dst + 1, odd, 0);
                if saturated {
                    self.note_saturation(b'L', side);
                }
            }
            Opcode::Dmv => {
                let (left, right, dst) = binary_operands(word, side, &self.a, &self.b);
                self.schedule(side, dst, right, 0);
                self.schedule(side, dst + 1, left, 0);
            }
            opcode @ (Opcode::Abs
            | Opcode::Abs2
            | Opcode::Swap4
            | Opcode::Unpkhu4
            | Opcode::Unpklu4) => {
                let x = word & (1 << 12) != 0;
                let src = self.read_register(
                    if x { opposite(side) } else { side },
                    ((word >> 18) & 0x1f) as usize,
                ) as i32;
                let dst = ((word >> 23) & 0x1f) as usize;
                let raw = src as u32;
                let mut saturated = false;
                let value = match opcode {
                    Opcode::Abs => {
                        saturated = src == i32::MIN;
                        src.saturating_abs() as u32
                    }
                    Opcode::Abs2 => {
                        let (value, lanes) = packed_16_sat(raw, raw, |half, _| {
                            let value = half as i16;
                            (value.saturating_abs() as u16, value == i16::MIN)
                        });
                        saturated = lanes;
                        value
                    }
                    Opcode::Swap4 => ((raw & 0x00ff_00ff) << 8) | ((raw & 0xff00_ff00) >> 8),
                    Opcode::Unpkhu4 => ((raw >> 16) & 0xff) | (raw & 0xff00_0000) >> 8,
                    _ => (raw & 0xff) | (raw & 0xff00) << 8,
                };
                self.schedule(side, dst, value, 0);
                if saturated {
                    self.note_saturation(encoded_alu_unit(word), side);
                }
            }
            opcode @ (Opcode::AddD
            | Opcode::SubD
            | Opcode::Addab
            | Opcode::Subab
            | Opcode::Addah
            | Opcode::Subah
            | Opcode::Addaw
            | Opcode::Subaw
            | Opcode::Addad
            | Opcode::MvkD) => {
                let op = (word >> 7) & 0x3f;
                let fields = source_fields(word);
                let (src1, src2, dst) = (fields.src1, fields.src2, fields.dst);
                if matches!(opcode, Opcode::MvkD) {
                    let value = sign_extend(src1 as u32, 5);
                    self.schedule(side, dst, value, 0);
                    return;
                }
                let immediate = if matches!(opcode, Opcode::Addad) {
                    op & 1 != 0
                } else {
                    op & 2 != 0
                };
                let offset = if immediate {
                    src1 as u32
                } else {
                    self.read_register(side, src1)
                };
                let scale = match opcode {
                    Opcode::Addah | Opcode::Subah => 1,
                    Opcode::Addaw | Opcode::Subaw => 2,
                    Opcode::Addad => 3,
                    _ => 0,
                };
                let base = self.read_register(side, src2);
                let scaled = offset << scale;
                let value = if matches!(
                    opcode,
                    Opcode::SubD | Opcode::Subab | Opcode::Subah | Opcode::Subaw
                ) {
                    base.wrapping_sub(scaled)
                } else {
                    base.wrapping_add(scaled)
                };
                // ADDA/SUBA honour circular addressing on their base register; plain ADD and
                // SUB on the D unit are address arithmetic without it.
                let value = if matches!(opcode, Opcode::AddD | Opcode::SubD) {
                    value
                } else {
                    circular_address(self.control.amr(), side, src2, base, value)
                };
                self.schedule(side, dst, value, 0);
            }
            Opcode::LongShift => {
                // The 40-bit shift family: the shift amount is src1 (register or five-bit
                // constant), the value is src2, and the result is a long register pair whose
                // odd register holds bits 39-32 with the upper 24 bits zero-filled.
                let op = (word >> 6) & 0x3f;
                let fields = source_fields(word);
                let src = fields.src2;
                let dst = fields.dst;
                let count = if op & 1 == 0 {
                    fields.src1 as u32
                } else {
                    self.read_register(side, fields.src1) & 0x3f
                };
                let widening = matches!(op, 0x12 | 0x13);
                let value = if widening {
                    let bank = if fields.cross { opposite(side) } else { side };
                    i64::from(self.read_register(bank, src) as i32)
                } else {
                    let low = i64::from(self.read_register(side, src));
                    let high = i64::from(self.read_register(side, src + 1) & 0xff);
                    let raw = (high << 32) | low;
                    if matches!(op, 0x24 | 0x25) {
                        raw
                    } else {
                        (raw << 24) >> 24
                    }
                };
                let count = count.min(40);
                let shifted = match op {
                    0x12 | 0x13 | 0x30 | 0x31 => value << count,
                    0x24 | 0x25 => ((value as u64) >> count) as i64,
                    _ => value >> count,
                };
                self.schedule(side, dst, shifted as u32, 0);
                self.schedule(side, dst + 1, ((shifted >> 32) & 0xff) as u32, 0);
            }
            opcode @ (Opcode::Shl | Opcode::Shr | Opcode::Shru) => {
                let op = (word >> 6) & 0x3f;
                let fields = source_fields(word);
                let value = self.read_register(
                    if fields.cross { opposite(side) } else { side },
                    fields.src2,
                );
                let count = if op & 1 == 0 {
                    fields.src1 as u32
                } else {
                    self.read_register(side, fields.src1) & 0x3f
                };
                let result = match opcode {
                    Opcode::Shl => value.checked_shl(count).unwrap_or(0),
                    Opcode::Shr => {
                        if count >= 32 {
                            u32::from((value as i32) < 0).wrapping_neg()
                        } else {
                            ((value as i32) >> count) as u32
                        }
                    }
                    Opcode::Shru => value.checked_shr(count).unwrap_or(0),
                    _ => unreachable!(),
                };
                self.schedule(side, fields.dst, result, 0);
            }
            Opcode::MoveLong | Opcode::MoveShort => {
                let src = ((word >> 18) & 0x1f) as usize;
                let dst = ((word >> 23) & 0x1f) as usize;
                let x = word & (1 << 12) != 0;
                let value = self.read_register(if x { opposite(side) } else { side }, src);
                self.schedule(side, dst, value, 0);
            }
            Opcode::MvcFrom => {
                let register = ((word >> 18) & 0x1f) as u8;
                if self.restricted_register(address, register) {
                    return;
                }
                let dst = ((word >> 23) & 0x1f) as usize;
                let value = self.control.read(register, address);
                self.schedule(Bank::B, dst, value, 0);
            }
            Opcode::MvcTo => {
                let register = ((word >> 23) & 0x1f) as u8;
                if self.restricted_register(address, register) {
                    return;
                }
                let src = ((word >> 18) & 0x1f) as usize;
                let bank = if word & (1 << 12) == 0 {
                    Bank::B
                } else {
                    Bank::A
                };
                self.writeback.schedule_control(ControlWrite {
                    due: self.cycle + u64::from(matches!(register, 0x0d | 0x0e)) * 4,
                    register,
                    value: self.read_register(bank, src),
                    user: self.user_mode,
                });
            }
            Opcode::Lddw => {
                let (effective, update) =
                    memory_address(word, self.control.amr(), &self.a, &self.b, 3);
                if let Some((base, value)) = update {
                    self.schedule(encoded_address_bank(word), base, value, 0);
                }
                let aligned = effective & !7;
                let data_bank = encoded_data_bank(word);
                let dst = ((word >> 23) & 0x1f) as usize;
                let (low, high) = doubleword_halves(
                    bus.endian(),
                    self.read_u32(bus, aligned),
                    self.read_u32(bus, aligned.wrapping_add(4)),
                );
                self.schedule_load(data_bank, dst, low, 4);
                self.schedule_load(data_bank, dst + 1, high, 4);
            }
            Opcode::Ldndw => {
                // The `sc` bit selects a scaled offset: 1 shifts the offset left by three (SPRUFE8B
                // LDNDW/STNDW), 0 leaves it in bytes.
                let scale = if word & (1 << 23) != 0 { 3 } else { 0 };
                let (effective, update) =
                    memory_address(word, self.control.amr(), &self.a, &self.b, scale);
                if let Some((base, value)) = update {
                    self.schedule(encoded_address_bank(word), base, value, 0);
                }
                let data_bank = encoded_data_bank(word);
                let dst = (((word >> 24) & 0x0f) * 2) as usize;
                let (low, high) = doubleword_halves(
                    bus.endian(),
                    self.read_u32(bus, effective),
                    self.read_u32(bus, effective.wrapping_add(4)),
                );
                self.schedule_load(data_bank, dst, low, 4);
                self.schedule_load(data_bank, dst + 1, high, 4);
            }
            Opcode::Ldnw => {
                let (effective, update) =
                    memory_address(word, self.control.amr(), &self.a, &self.b, 2);
                if let Some((base, value)) = update {
                    self.schedule(encoded_address_bank(word), base, value, 0);
                }
                let data_bank = encoded_data_bank(word);
                let value = self.read_u32(bus, effective);
                self.schedule_load(data_bank, ((word >> 23) & 0x1f) as usize, value, 4);
            }
            Opcode::LoadLinked => {
                let address = self.read_register(Bank::B, ((word >> 18) & 0x1f) as usize) & !3;
                // LL is a load: it is recorded for the L1D timing and judged by the memory
                // protection unit once, as `read_u32` does it. The bus reads the word and links it.
                self.note_load(bus, address);
                let user = self.user_mode;
                let value = if bus.check_access(address, Permission::Read, user) {
                    bus.load_link(address)
                } else {
                    0
                };
                self.schedule_load(Bank::B, ((word >> 23) & 0x1f) as usize, value, 4);
            }
            Opcode::StoreLinked => {
                let address = self.read_register(Bank::B, ((word >> 18) & 0x1f) as usize) & !3;
                let source = ((word >> 23) & 0x1f) as usize;
                bus.store_link(address, self.read_register(Bank::B, source));
            }
            Opcode::CommitLinked => {
                let address = self.read_register(Bank::B, ((word >> 18) & 0x1f) as usize) & !3;
                let committed = bus.commit_link(address);
                if let Some(value) = committed {
                    self.note_store(bus, address);
                    self.writeback.schedule_store(StoreWrite {
                        due: self.cycle,
                        address,
                        size: 4,
                        value,
                    });
                }
                self.schedule_load(
                    Bank::B,
                    ((word >> 23) & 0x1f) as usize,
                    u32::from(committed.is_some()),
                    4,
                );
            }
            opcode @ (Opcode::Ldb | Opcode::Ldbu | Opcode::Ldh | Opcode::Ldhu | Opcode::Ldw) => {
                let (size, scale) = match opcode {
                    Opcode::Ldb | Opcode::Ldbu => (1, 0),
                    Opcode::Ldh | Opcode::Ldhu => (2, 1),
                    Opcode::Ldw => (4, 2),
                    _ => unreachable!(),
                };
                let (effective, update) =
                    memory_address(word, self.control.amr(), &self.a, &self.b, scale);
                if let Some((base, value)) = update {
                    self.schedule(encoded_address_bank(word), base, value, 0);
                }
                let aligned = effective & !(size - 1);
                let value = match opcode {
                    Opcode::Ldb => sign_extend(self.read_u8(bus, aligned) as u32, 8),
                    Opcode::Ldbu => self.read_u8(bus, aligned) as u32,
                    Opcode::Ldh => sign_extend(self.read_u16(bus, aligned) as u32, 16),
                    Opcode::Ldhu => self.read_u16(bus, aligned) as u32,
                    Opcode::Ldw => self.read_u32(bus, aligned),
                    _ => unreachable!(),
                };
                let data_bank = encoded_data_bank(word);
                self.schedule_load(data_bank, ((word >> 23) & 0x1f) as usize, value, 4);
            }
            Opcode::Stdw => {
                let (effective, update) =
                    memory_address(word, self.control.amr(), &self.a, &self.b, 3);
                if let Some((base, value)) = update {
                    self.schedule(encoded_address_bank(word), base, value, 0);
                }
                let aligned = effective & !7;
                self.note_store(bus, aligned);
                let data_bank = encoded_data_bank(word);
                let src = ((word >> 23) & 0x1f) as usize;
                let (first, second) = doubleword_halves(
                    bus.endian(),
                    self.read_register(data_bank, src),
                    self.read_register(data_bank, src + 1),
                );
                self.schedule_doubleword_store(aligned, first, second);
            }
            Opcode::Stndw => {
                // The `sc` bit selects a scaled offset: 1 shifts the offset left by three (SPRUFE8B
                // LDNDW/STNDW), 0 leaves it in bytes.
                let scale = if word & (1 << 23) != 0 { 3 } else { 0 };
                let (effective, update) =
                    memory_address(word, self.control.amr(), &self.a, &self.b, scale);
                if let Some((base, value)) = update {
                    self.schedule(encoded_address_bank(word), base, value, 0);
                }
                let data_bank = encoded_data_bank(word);
                self.note_store(bus, effective);
                let src = (((word >> 24) & 0x0f) * 2) as usize;
                let (first, second) = doubleword_halves(
                    bus.endian(),
                    self.read_register(data_bank, src),
                    self.read_register(data_bank, src + 1),
                );
                self.schedule_doubleword_store(effective, first, second);
            }
            Opcode::Stnw => {
                let (effective, update) =
                    memory_address(word, self.control.amr(), &self.a, &self.b, 2);
                if let Some((base, value)) = update {
                    self.schedule(encoded_address_bank(word), base, value, 0);
                }
                let data_bank = encoded_data_bank(word);
                self.note_store(bus, effective);
                self.writeback.schedule_store(StoreWrite {
                    due: self.cycle,
                    address: effective,
                    size: 4,
                    value: self.read_register(data_bank, ((word >> 23) & 0x1f) as usize),
                });
            }
            opcode @ (Opcode::Stb | Opcode::Sth | Opcode::Stw) => {
                let (size, scale) = match opcode {
                    Opcode::Stb => (1, 0),
                    Opcode::Sth => (2, 1),
                    Opcode::Stw => (4, 2),
                    _ => unreachable!(),
                };
                let (effective, update) =
                    memory_address(word, self.control.amr(), &self.a, &self.b, scale);
                if let Some((base, value)) = update {
                    self.schedule(encoded_address_bank(word), base, value, 0);
                }
                let data_bank = encoded_data_bank(word);
                self.note_store(bus, effective & !(size - 1));
                let value = self.read_register(data_bank, ((word >> 23) & 0x1f) as usize);
                self.writeback.schedule_store(StoreWrite {
                    due: self.cycle,
                    address: effective & !(size - 1),
                    value,
                    size: size as u8,
                });
            }
            Opcode::Sploop | Opcode::Sploopd | Opcode::Sploopw => {
                self.sploop.start(LoopStart {
                    ii: ((word >> 23) & 0x1f) + 1,
                    start_cycle: self.cycle,
                    start_address: address,
                    // Piping up after an interrupt, SPLOOPD executes like SPLOOP: its delayed
                    // termination testing is not repeated (SPRUFE8B 7.13.5).
                    delayed: opcode != Opcode::Sploop && !self.pipe_up,
                    while_test: (opcode == Opcode::Sploopw)
                        .then(|| predicate_register(word))
                        .flatten(),
                    reload_test: (opcode != Opcode::Sploopw)
                        .then(|| predicate_register(word))
                        .flatten(),
                    nested: opcode != Opcode::Sploopw && predicate_register(word).is_some(),
                });
            }
            Opcode::Spkernel | Opcode::Spkernelr => {
                self.sploop.end_loading();
                self.pipe_up = false;
                if opcode == Opcode::Spkernelr {
                    self.arm_reload();
                }
            }
            // `DINT` disables interrupts, saving the previous enable in TSR.SGIE, and
            // `RINT` restores it. None of these three use a functional unit.
            Opcode::Dint => self.control.disable_interrupts(),
            Opcode::Rint => self.control.restore_interrupts(),
            // `SWE` and `SWENR` request a supervisor service. Every exception type is
            // enabled by TSR.GEE, which an operating system sets once at boot and can never
            // clear, so with exceptions off these behave as they did before: no vector, no
            // flag. `SWENR` enters at REP rather than at the exception vector, so User code
            // cannot choose where it lands in the supervisor (SPRU732J 8.4).
            Opcode::SoftwareException => {
                if self.control.exceptions_enabled() {
                    self.pending_exception = Some(Exception::Software {
                        restricted: word & 0x2000 != 0,
                    });
                }
            }
            Opcode::Spmask => {}
            Opcode::Spmaskr => self.arm_reload(),
            opcode @ (Opcode::Bdec | Opcode::Bpos) => {
                // Both branch when the named register is non-negative, relative to the first
                // word of the containing fetch packet; BDEC also decrements that register.
                let register = ((word >> 23) & 0x1f) as usize;
                let value = self.read_register(side, register) as i32;
                if value >= 0 {
                    let displacement = sign_extend((word >> 13) & 0x3ff, 10) << 2;
                    let target = (address & !0x1f).wrapping_add(displacement);
                    self.branches
                        .push((self.cycle + 6, BranchTarget::Address(target)));
                    self.branch_block_until = self.cycle + 6;
                    if opcode == Opcode::Bdec {
                        self.schedule(side, register, (value - 1) as u32, 0);
                    }
                }
            }
            Opcode::Branch => {
                let displacement = sign_extend((word >> 7) & 0x1f_ffff, 21) << 2;
                let target = (address & !0x1f).wrapping_add(displacement);
                self.branches
                    .push((self.cycle + 6, BranchTarget::Address(target)));
                self.branch_block_until = self.cycle + 6;
            }
            Opcode::BnopImmediate => {
                let scale = if header_based { 1 } else { 2 };
                let displacement = sign_extend((word >> 16) & 0x0fff, 12) << scale;
                let target = (address & !0x1f).wrapping_add(displacement);
                self.branches
                    .push((self.cycle + 6, BranchTarget::Address(target)));
                self.branch_block_until = self.cycle + 6;
                self.stall_cycles = self.stall_cycles.max((word >> 13) & 7);
            }
            Opcode::BranchRegister | Opcode::BnopRegister => {
                let src = ((word >> 18) & 0x1f) as usize;
                let bank = if word & (1 << 12) == 0 {
                    Bank::B
                } else {
                    Bank::A
                };
                // `B src2` reads src2 in E1, so the target is the value the register holds
                // when the branch issues: a write in its own delay slots - which is how the
                // compiler frees the register for the code after the call - cannot move it
                // (SPRUFE8B, "Branch Using a Register", pipeline table).
                let target = self.read_register(bank, src) & !1u32;
                self.branches
                    .push((self.cycle + 6, BranchTarget::Address(target)));
                self.branch_block_until = self.cycle + 6;
                if opcode == Opcode::BnopRegister {
                    self.stall_cycles = self.stall_cycles.max((word >> 13) & 7);
                }
            }
            Opcode::BranchIrp => {
                if self.restricted_instruction(address, "B IRP") {
                    return;
                }
                let target = self.control.return_from_irp() & !1;
                // TSR comes back with SPLX set when the interrupt drained a loop, so the
                // SPLOOP at the return address pipes the buffer up (SPRUFE8B 7.13.5).
                self.pipe_up = self.control.sploop_executing();
                self.branches
                    .push((self.cycle + 6, BranchTarget::Address(target)));
                self.branch_block_until = self.cycle + 6;
                // The restored execution mode takes effect with the branch target; the delay
                // slots still run in the mode the handler was using (SPRU732J 8.1).
                self.mode_change = Some((self.cycle + 6, self.control.user_mode()));
            }
            Opcode::BranchNrp => {
                if self.restricted_instruction(address, "B NRP") {
                    return;
                }
                let target = self.control.return_from_nrp() & !1;
                self.pipe_up = self.control.sploop_executing();
                self.branches
                    .push((self.cycle + 6, BranchTarget::Address(target)));
                self.branch_block_until = self.cycle + 6;
                self.mode_change = Some((self.cycle + 6, self.control.user_mode()));
            }
            Opcode::Addkpc => {
                let displacement = sign_extend((word >> 16) & 0x7f, 7) << 2;
                let value = (address & !0x1f).wrapping_add(displacement);
                let dst = ((word >> 23) & 0x1f) as usize;
                self.schedule(side, dst, value, 0);
                self.stall_cycles = self.stall_cycles.max((word >> 13) & 7);
            }
            Opcode::Callp => {
                let displacement = sign_extend((word >> 7) & 0x1f_ffff, 21) << 2;
                let target = (address & !0x1f).wrapping_add(displacement);
                // The return register is fully determined by the functional unit side:
                // CALLP.S1 writes A3 and CALLP.S2 writes B3.
                self.schedule(side, 3, packet_next, 0);
                self.branches
                    .push((self.cycle + 6, BranchTarget::Address(target)));
                self.branch_block_until = self.cycle + 6;
                self.stall_cycles = self.stall_cycles.max(5);
            }
            opcode @ (Opcode::LoadLong | Opcode::StoreLong | Opcode::AddaLong) => {
                let op = (word >> 4) & 7;
                let base_register = if word & (1 << 7) == 0 { 14 } else { 15 };
                let register = ((word >> 23) & 0x1f) as usize;
                let offset = (word >> 8) & 0x7fff;
                let base = self.read_register(Bank::B, base_register);
                let (size, signed) = match op {
                    0 => (2u32, false),
                    1 => (1, false),
                    2 => (1, true),
                    3 => (1, false),
                    4 => (2, true),
                    5 => (2, false),
                    _ => (4, false),
                };
                let scale = match (opcode, op) {
                    (Opcode::AddaLong, 3) => 0,
                    (Opcode::AddaLong, 5) => 1,
                    (Opcode::AddaLong, _) => 2,
                    (_, _) => size.trailing_zeros(),
                };
                let effective = base.wrapping_add(offset << scale);
                match opcode {
                    Opcode::AddaLong => self.schedule(side, register, effective, 0),
                    Opcode::LoadLong => {
                        let aligned = effective & !(size - 1);
                        let value = match (size, signed) {
                            (1, false) => self.read_u8(bus, aligned) as u32,
                            (1, true) => sign_extend(self.read_u8(bus, aligned) as u32, 8),
                            (2, false) => self.read_u16(bus, aligned) as u32,
                            (2, true) => sign_extend(self.read_u16(bus, aligned) as u32, 16),
                            _ => self.read_u32(bus, aligned),
                        };
                        self.schedule_load(side, register, value, 4);
                    }
                    _ => {
                        self.note_store(bus, effective & !(size - 1));
                        self.writeback.schedule_store(StoreWrite {
                            due: self.cycle,
                            address: effective & !(size - 1),
                            size: size as u8,
                            value: self.read_register(side, register),
                        })
                    }
                }
            }
            Opcode::Mpy
            | Opcode::MpyCompound
            | Opcode::MpyUnary
            | Opcode::MpyExtended
            | Opcode::Xormpy
            | Opcode::Gmpy => {
                self.execute_m_unit(word, opcode, side);
            }
            Opcode::SubCross | Opcode::SubCrossS => {
                // `SUB .L1X src1, src2, dst`: here the cross path feeds src1 while src2
                // stays on the unit's own side.
                let x = word & (1 << 12) != 0;
                let src1 = self.read_register(
                    if x { opposite(side) } else { side },
                    ((word >> 13) & 0x1f) as usize,
                );
                let src2 = self.read_register(side, ((word >> 18) & 0x1f) as usize);
                let dst = ((word >> 23) & 0x1f) as usize;
                self.schedule(side, dst, src1.wrapping_sub(src2), 0);
            }
            opcode @ (Opcode::BitField | Opcode::BitFieldRegister) => {
                let register_form = matches!(opcode, Opcode::BitFieldRegister);
                let fields = source_fields(word);
                let (op, csta, cstb, src) = if register_form {
                    // The register form takes both field positions from one control register,
                    // packed as csta in bits 9-5 and cstb in bits 4-0; the value - src2 - is
                    // the cross-path operand here.
                    let control = self.read_register(side, fields.src1);
                    let src = self.read_register(
                        if fields.cross { opposite(side) } else { side },
                        fields.src2,
                    );
                    let op = match (word >> 6) & 0x3f {
                        0x2b => 0,
                        0x2f => 1,
                        0x3b => 2,
                        _ => 3,
                    };
                    (op, (control >> 5) & 0x1f, control & 0x1f, src)
                } else {
                    // The constant form: csta in src1's field and cstb at bits 8-12, which is
                    // where the operand rendering reads them too.
                    (
                        (word >> 6) & 3,
                        fields.src1 as u32,
                        (word >> 8) & 0x1f,
                        self.read_register(side, fields.src2),
                    )
                };
                let dst = fields.dst;
                let value = match op {
                    0 => (src << csta.min(31)) >> cstb.min(31),
                    1 => (((src << csta.min(31)) as i32) >> cstb.min(31)) as u32,
                    _ => {
                        // For SET and CLR, csta is the LSB of the field and cstb the MSB
                        // (SPRUFE8B: "csta is the LSB of the field and cstb is the MSB").
                        let (low, high) = (csta.min(31), cstb.min(31));
                        let width = high.saturating_sub(low) + 1;
                        let mask = if width >= 32 {
                            u32::MAX
                        } else {
                            ((1u32 << width) - 1) << low
                        };
                        if op == 2 {
                            src | mask
                        } else {
                            src & !mask
                        }
                    }
                };
                self.schedule(side, dst, value, 0);
            }
            Opcode::Addk => {
                let constant = sign_extend((word >> 7) & 0xffff, 16);
                let dst = ((word >> 23) & 0x1f) as usize;
                let value = self.read_register(side, dst).wrapping_add(constant);
                self.schedule(side, dst, value, 0);
            }
            Opcode::Unknown => self.fault_invalid_opcode(address, word),
        }
    }

    /// M unit: multiplies and the packed multiply/shift helpers.
    ///
    /// Results are written through the normal write-back queue with the architectural
    /// latency of each class: 2 cycles for the 16x16 forms and 4 for the 32-bit and packed
    /// ones. The 9- and 10-cycle classes are MPYI and MPYID, which are C67x instructions this
    /// device does not have - their encodings decode to nothing.
    fn execute_m_unit(&mut self, word: u32, opcode: Opcode, side: Bank) {
        // The field positions come from `source_fields` and the operand roles from
        // `mpy_roles`, which is the table that names each family: the constant and pair roles
        // the rendering prints are the ones asserted here.
        let fields = source_fields(word);
        let roles = mpy_roles(opcode, word);
        let src1_field = fields.src1;
        let src2 = self.read_register(
            if fields.cross { opposite(side) } else { side },
            fields.src2,
        );
        let src1 = self.read_register(side, src1_field);
        let dst = fields.dst;
        let low = |value: u32| (value & 0xffff) as u16;
        let high = |value: u32| (value >> 16) as u16;
        let signed = |value: u16| i32::from(value as i16);
        let unsigned = |value: u16| i32::from(value);
        // SMPY doubles the product, which saturates only for 0x8000 * 0x8000.
        let saturate_double = |product: i32| match product.checked_mul(2) {
            Some(value) => (value as u32, false),
            None => (0x7fff_ffff, true),
        };
        let mut saturated = false;
        match opcode {
            Opcode::Mpy => {
                let op = (word >> 7) & 0x1f;
                let immediate = || {
                    debug_assert!(roles.src1_constant && !roles.src1_constant_unsigned);
                    sign_extend(src1_field as u32, 5) as i32
                };
                let (value, delay) = match op {
                    0x00 => ((signed(low(src1)) * signed(low(src2))) as u32, 1),
                    0x01 => ((signed(high(src1)) * signed(high(src2))) as u32, 1),
                    0x02 => {
                        let (value, clamped) =
                            saturate_double(signed(high(src1)) * signed(high(src2)));
                        saturated = clamped;
                        (value, 1)
                    }
                    0x03 => ((signed(high(src1)) * unsigned(high(src2))) as u32, 1),
                    0x05 => ((unsigned(high(src1)) * signed(high(src2))) as u32, 1),
                    0x07 => ((unsigned(high(src1)) * unsigned(high(src2))) as u32, 1),
                    0x09 => ((signed(high(src1)) * signed(low(src2))) as u32, 1),
                    0x0a => {
                        let (value, clamped) =
                            saturate_double(signed(high(src1)) * signed(low(src2)));
                        saturated = clamped;
                        (value, 1)
                    }
                    0x0b => ((signed(high(src1)) * unsigned(low(src2))) as u32, 1),
                    0x0d => ((unsigned(high(src1)) * signed(low(src2))) as u32, 1),
                    0x0f => ((unsigned(high(src1)) * unsigned(low(src2))) as u32, 1),
                    0x11 => ((signed(low(src1)) * signed(high(src2))) as u32, 1),
                    0x12 => {
                        let (value, clamped) =
                            saturate_double(signed(low(src1)) * signed(high(src2)));
                        saturated = clamped;
                        (value, 1)
                    }
                    0x13 => ((signed(low(src1)) * unsigned(high(src2))) as u32, 1),
                    0x15 => ((unsigned(low(src1)) * signed(high(src2))) as u32, 1),
                    0x17 => ((unsigned(low(src1)) * unsigned(high(src2))) as u32, 1),
                    0x18 => ((immediate() * signed(low(src2))) as u32, 1),
                    0x19 => ((signed(low(src1)) * signed(low(src2))) as u32, 1),
                    0x1a => {
                        let (value, clamped) =
                            saturate_double(signed(low(src1)) * signed(low(src2)));
                        saturated = clamped;
                        (value, 1)
                    }
                    0x1b => ((signed(low(src1)) * unsigned(low(src2))) as u32, 1),
                    0x1d => ((unsigned(low(src1)) * signed(low(src2))) as u32, 1),
                    0x1e => ((immediate() * unsigned(low(src2))) as u32, 1),
                    0x1f => ((unsigned(low(src1)) * unsigned(low(src2))) as u32, 1),
                    0x10 => ((src1 as i32).wrapping_mul(src2 as i32) as u32, 3),
                    // The 64-bit MPY32 forms, whose result is a register pair.
                    0x14 | 0x16 => {
                        debug_assert!(roles.dst_pair);
                        let product = if op == 0x14 {
                            i64::from(src1 as i32) * i64::from(src2 as i32)
                        } else {
                            i64::from(src1 as i32) * i64::from(src2)
                        };
                        self.schedule(side, dst, product as u32, 3);
                        self.schedule(side, dst + 1, (product >> 32) as u32, 3);
                        return;
                    }
                    _ => {
                        self.fault_unsupported_unit(word, "M", "multiply", op);
                        return;
                    }
                };
                self.schedule(side, dst, value, delay);
                if saturated {
                    self.note_saturation(b'M', side);
                }
            }
            Opcode::MpyCompound => {
                let op = (word >> 6) & 0x1f;
                match op {
                    0x14 | 0x15 => {
                        let half = if op == 0x14 {
                            signed(high(src1))
                        } else {
                            signed(low(src1))
                        };
                        let product = i64::from(half) * i64::from(src2 as i32);
                        self.schedule(side, dst, product as u32, 3);
                        self.schedule(side, dst + 1, (product >> 32) as u32, 3);
                    }
                    0x10 | 0x0e => {
                        let half = if op == 0x10 {
                            signed(high(src1))
                        } else {
                            signed(low(src1))
                        };
                        let product = i64::from(half) * i64::from(src2 as i32);
                        let rounded = (product + 0x4000) >> 15;
                        self.schedule(side, dst, rounded as u32, 3);
                    }
                    0x18 | 0x19 => {
                        let product = if op == 0x18 {
                            u64::from(src1) * u64::from(src2)
                        } else {
                            u64::from(src1).wrapping_mul(i64::from(src2 as i32) as u64)
                        };
                        self.schedule(side, dst, product as u32, 3);
                        self.schedule(side, dst + 1, (product >> 32) as u32, 3);
                    }
                    0x00 | 0x01 => {
                        let saturating = op == 0x01;
                        let lower = signed(low(src1)) * signed(low(src2));
                        let upper = signed(high(src1)) * signed(high(src2));
                        let (lower, upper) = if saturating {
                            let (lower, low_sat) = saturate_double(lower);
                            let (upper, high_sat) = saturate_double(upper);
                            saturated = low_sat || high_sat;
                            (lower, upper)
                        } else {
                            (lower as u32, upper as u32)
                        };
                        self.schedule(side, dst, lower, 3);
                        self.schedule(side, dst + 1, upper, 3);
                    }
                    0x09 | 0x0b | 0x0c => {
                        let lower = signed(low(src1)) * signed(low(src2));
                        let upper = signed(high(src1)) * signed(high(src2));
                        // DOTPN2 subtracts the product of the lower halfwords from the
                        // product of the upper halfwords (SPRUFE8B DOTPN2); DOTP2 adds them.
                        let value = if op == 0x09 {
                            upper.wrapping_sub(lower)
                        } else {
                            lower.wrapping_add(upper)
                        };
                        self.schedule(side, dst, value as u32, 3);
                        if op == 0x0b {
                            debug_assert!(roles.dst_pair);
                            self.schedule(side, dst + 1, (value >> 31) as u32, 3);
                        }
                    }
                    0x1a | 0x1c | 0x1d | 0x1e => {
                        let count = if op == 0x1e {
                            debug_assert!(roles.src1_constant && roles.src1_constant_unsigned);
                            src1_field as u32 & 0x1f
                        } else {
                            src1 & 0x1f
                        };
                        let value = match op {
                            0x1a | 0x1c => {
                                // The count is a 2s-complement value limited to -31..31; a
                                // negative count reverses the direction. Left shifts saturate
                                // to 32 bits, right shifts sign extend (SPRUFE8B SSHVL/SSHVR).
                                let signed_count = (src1 as i32).clamp(-31, 31);
                                let left = if op == 0x1c {
                                    signed_count
                                } else {
                                    -signed_count
                                };
                                if left >= 0 {
                                    let (value, clamped) =
                                        saturate_i32(i64::from(src2 as i32) << left);
                                    saturated = clamped;
                                    value
                                } else {
                                    ((src2 as i32) >> -left) as u32
                                }
                            }
                            _ => src2.rotate_left(count),
                        };
                        self.schedule(side, dst, value, 1);
                    }
                    // Four byte products into a register pair: byte 0 into the lower half of
                    // dst_e, byte 3 into the upper half of dst_o.
                    0x04 | 0x05 => {
                        let byte = |value: u32, lane: u32| (value >> (lane * 8)) & 0xff;
                        let product = |lane: u32| {
                            let left = if op == 0x05 {
                                i32::from(byte(src1, lane) as u8 as i8)
                            } else {
                                byte(src1, lane) as i32
                            };
                            ((left * byte(src2, lane) as i32) as u32) & 0xffff
                        };
                        self.schedule(side, dst, product(0) | (product(1) << 16), 3);
                        self.schedule(side, dst + 1, product(2) | (product(3) << 16), 3);
                    }
                    // Byte dot products: src1 is signed for DOTPSU4, unsigned for DOTPU4.
                    0x02 | 0x06 => {
                        let mut total: i32 = 0;
                        for lane in 0..4 {
                            let left = if op == 0x02 {
                                i32::from(((src1 >> (lane * 8)) & 0xff) as u8 as i8)
                            } else {
                                ((src1 >> (lane * 8)) & 0xff) as i32
                            };
                            let right = ((src2 >> (lane * 8)) & 0xff) as i32;
                            total = total.wrapping_add(left * right);
                        }
                        self.schedule(side, dst, total as u32, 3);
                    }
                    // Halfword dot products of signed src1 by unsigned src2, rounded by 2^15
                    // and shifted right by 16. DOTPNRSU2 negates the lower product.
                    0x07 | 0x0d => {
                        let lower = i64::from(signed(low(src1))) * i64::from(low(src2));
                        let upper = i64::from(signed(high(src1))) * i64::from(high(src2));
                        let sum = if op == 0x07 {
                            upper - lower
                        } else {
                            upper + lower
                        };
                        self.schedule(side, dst, ((sum + 0x8000) >> 16) as u32, 3);
                    }
                    // Galois field multiply of four bytes under GFPGFR's polynomial and size.
                    0x11 => {
                        let gfpgfr = self.control.gfpgfr();
                        let bits = ((gfpgfr >> 24) & 0x7) + 1;
                        let polynomial = gfpgfr & 0xff;
                        let mask = (1u32 << bits) - 1;
                        let multiply = |left: u32, right: u32| {
                            let (left, right) = (left & mask, right & mask);
                            let mut product = 0u32;
                            for bit in (0..bits).rev() {
                                product <<= 1;
                                if (right >> bit) & 1 != 0 {
                                    product ^= left;
                                }
                                if product & (1 << bits) != 0 {
                                    product ^= (1 << bits) | polynomial;
                                }
                            }
                            product & mask
                        };
                        let mut value = 0u32;
                        for lane in 0..4 {
                            let shift = lane * 8;
                            value |= (multiply(src1 >> shift, src2 >> shift) & 0xff) << shift;
                        }
                        self.schedule(side, dst, value, 3);
                    }
                    0x12 | 0x13 => {
                        let value = if op == 0x13 {
                            packed_16(src1, src2, |left, right| {
                                (((left as i16 as i32) + (right as i16 as i32) + 1) >> 1) as u16
                            })
                        } else {
                            packed_8(src1, src2, |left, right| {
                                ((u32::from(left) + u32::from(right) + 1) >> 1) as u8
                            })
                        };
                        self.schedule(side, dst, value, 1);
                    }
                    _ => self.fault_unsupported_unit(word, "M", "compound", op),
                }
                if saturated {
                    self.note_saturation(b'M', side);
                }
            }
            // GMPY multiplies src1 by the nine low bits of src2 in GF(2^32), reducing by the
            // polynomial in GPLYA or GPLYB according to the side (SPRUFE8B GMPY).
            Opcode::Gmpy => {
                let polynomial = self.control.polynomial(matches!(side, Bank::B));
                let mut product = 0u32;
                let mut mask = 0x100;
                for _ in 0..8 {
                    if src2 & mask != 0 {
                        product ^= src1;
                    }
                    mask >>= 1;
                    let shifted = product << 1;
                    product = if product & 0x8000_0000 != 0 {
                        polynomial ^ shifted
                    } else {
                        shifted
                    };
                }
                if src2 & 1 != 0 {
                    product ^= src1;
                }
                self.schedule(side, dst, product, 3);
            }
            // XORMPY is a carry-less multiply of the whole of src1 by the nine low bits of
            // src2, left asserted with no reduction (SPRUFE8B XORMPY).
            Opcode::Xormpy => {
                let mut product = 0u32;
                for bit in 0..9 {
                    if (src2 >> bit) & 1 != 0 {
                        product ^= src1 << bit;
                    }
                }
                self.schedule(side, dst, product, 3);
            }
            // The C64x+ complex-multiply family in the `m_1_or_2_src` space. Every one is a
            // four-cycle instruction and every worked example in SPRUFE8B is reproduced by
            // these expressions.
            Opcode::MpyExtended => {
                let op = (word >> 6) & 0x1f;
                let src1_pair = || {
                    debug_assert!(roles.src1_pair);
                    let even = self.read_register(side, src1_field);
                    let odd = self.read_register(side, src1_field + 1);
                    (
                        i64::from(signed(low(even))),
                        i64::from(signed(high(even))),
                        i64::from(signed(low(odd))),
                        i64::from(signed(high(odd))),
                    )
                };
                let (src1_low, src1_high) =
                    (i64::from(signed(low(src1))), i64::from(signed(high(src1))));
                let (src2_low, src2_high) =
                    (i64::from(signed(low(src2))), i64::from(signed(high(src2))));
                let pack_halves = |upper: u32, lower: u32| (upper << 16) | (lower & 0xffff);
                match op {
                    // CMPY: dst_o takes the real part, dst_e the imaginary part.
                    0x0a => {
                        let (odd, odd_sat) =
                            saturate_i32(src1_high * src2_high - src1_low * src2_low);
                        let (even, even_sat) =
                            saturate_i32(src1_high * src2_low + src1_low * src2_high);
                        saturated = odd_sat || even_sat;
                        self.schedule(side, dst, even, 3);
                        self.schedule(side, dst + 1, odd, 3);
                    }
                    // CMPYR rounds by 2^15 and keeps the upper halfwords; CMPYR1 rounds by
                    // 2^14 and shifts left by one with saturation first.
                    0x0b | 0x0c => {
                        let round = if op == 0x0b { 0x8000 } else { 0x4000 };
                        let convert = |value: i64| {
                            let (first, first_sat) = saturate_i32(value + round);
                            if op == 0x0b {
                                (first >> 16, first_sat)
                            } else {
                                let (doubled, second_sat) =
                                    saturate_i32(i64::from(first as i32) << 1);
                                (doubled >> 16, first_sat || second_sat)
                            }
                        };
                        let (real, real_sat) = convert(src1_high * src2_high - src1_low * src2_low);
                        let (imag, imag_sat) = convert(src1_high * src2_low + src1_low * src2_high);
                        saturated = real_sat || imag_sat;
                        self.schedule(side, dst, pack_halves(real, imag), 3);
                    }
                    // DDOTP4 multiplies the halfwords of src1 by the sign-extended bytes of
                    // src2; no saturation case exists.
                    0x18 => {
                        let byte = |index: u32| i64::from((src2 >> (index * 8)) as u8 as i8);
                        let even = src1_high * byte(1) + src1_low * byte(0);
                        let odd = src1_high * byte(3) + src1_low * byte(2);
                        self.schedule(side, dst, even as u32, 3);
                        self.schedule(side, dst + 1, odd as u32, 3);
                    }
                    // DDOTPH2 and DDOTPL2 read a register pair on src1.
                    0x17 | 0x16 => {
                        let (even_low, even_high, odd_low, odd_high) = src1_pair();
                        let (first, second) = if op == 0x17 {
                            (
                                odd_low * src2_low + odd_high * src2_high,
                                src2_high * odd_low + src2_low * even_high,
                            )
                        } else {
                            (
                                even_low * src2_low + even_high * src2_high,
                                src2_high * odd_low + src2_low * even_high,
                            )
                        };
                        let (first, first_sat) = saturate_i32(first);
                        let (second, second_sat) = saturate_i32(second);
                        saturated = first_sat || second_sat;
                        if op == 0x17 {
                            self.schedule(side, dst, second, 3);
                            self.schedule(side, dst + 1, first, 3);
                        } else {
                            self.schedule(side, dst, first, 3);
                            self.schedule(side, dst + 1, second, 3);
                        }
                    }
                    // The rounded forms pack the two results into one register: DDOTPH2R puts
                    // its first result in the upper half, DDOTPL2R in the lower half.
                    0x15 | 0x14 => {
                        let (even_low, even_high, odd_low, odd_high) = src1_pair();
                        let (first, second) = if op == 0x15 {
                            (
                                odd_low * src2_low + odd_high * src2_high,
                                src2_high * odd_low + src2_low * even_high,
                            )
                        } else {
                            (
                                even_low * src2_low + even_high * src2_high,
                                src2_high * odd_low + src2_low * even_high,
                            )
                        };
                        let (first, first_sat) = saturate_i32(first + 0x8000);
                        let (second, second_sat) = saturate_i32(second + 0x8000);
                        saturated = first_sat || second_sat;
                        let value = if op == 0x15 {
                            pack_halves(first >> 16, second >> 16)
                        } else {
                            pack_halves(second >> 16, first >> 16)
                        };
                        self.schedule(side, dst, value, 3);
                    }
                    // MPY2IR: each halfword of src1 times the 32-bit src2, rounded by 2^14 and
                    // shifted right by 15.
                    0x0f => {
                        let convert = |half: i64| {
                            saturate_i32((half * i64::from(src2 as i32) + 0x4000) >> 15)
                        };
                        let (even, even_sat) = convert(src1_low);
                        let (odd, odd_sat) = convert(src1_high);
                        saturated = even_sat || odd_sat;
                        self.schedule(side, dst, even, 3);
                        self.schedule(side, dst + 1, odd, 3);
                    }
                    // SMPY32 doubles the 64-bit product and keeps the upper 32 bits.
                    0x19 => {
                        let product = i64::from(src1 as i32) * i64::from(src2 as i32);
                        let (value, clamped) = saturate_i32(product.wrapping_mul(2) >> 32);
                        saturated = clamped;
                        self.schedule(side, dst, value, 3);
                    }
                    _ => self.fault_unsupported_unit(word, "M", "extended", op),
                }
                if saturated {
                    self.note_saturation(b'M', side);
                }
            }
            Opcode::MpyUnary => {
                let op = (word >> 13) & 0x1f;
                let value = match op {
                    0x1a => src2,
                    0x1e => {
                        let mut value = 0u32;
                        for lane in 0..4 {
                            let byte = (src2 >> (lane * 8)) & 0xff;
                            value |= byte.count_ones() << (lane * 8);
                        }
                        value
                    }
                    0x1f => src2.reverse_bits(),
                    // XPND4/XPND2 replicate the low bits of src2 across bytes and halfwords.
                    0x18 => {
                        let mut value = 0u32;
                        for lane in 0..4 {
                            if (src2 >> lane) & 1 != 0 {
                                value |= 0xff << (lane * 8);
                            }
                        }
                        value
                    }
                    0x19 => {
                        let low = if src2 & 1 != 0 { 0x0000_ffff } else { 0 };
                        let high = if src2 & 2 != 0 { 0xffff_0000u32 } else { 0 };
                        low | high
                    }
                    // SHFL interleaves the two halfwords; DEAL splits even and odd bits.
                    0x1c => {
                        let mut value = 0u32;
                        for bit in 0..16 {
                            value |= ((src2 >> bit) & 1) << (2 * bit);
                            value |= ((src2 >> (bit + 16)) & 1) << (2 * bit + 1);
                        }
                        value
                    }
                    0x1d => {
                        let mut even = 0u32;
                        let mut odd = 0u32;
                        for bit in 0..16 {
                            even |= ((src2 >> (2 * bit)) & 1) << bit;
                            odd |= ((src2 >> (2 * bit + 1)) & 1) << bit;
                        }
                        (odd << 16) | even
                    }
                    _ => {
                        self.fault_unsupported_unit(word, "M", "unary", op);
                        return;
                    }
                };
                let delay = if op == 0x1a { 3 } else { 1 };
                self.schedule(side, dst, value, delay);
            }
            _ => self.fault_unsupported_opcode(word),
        }
    }

    /// Execute one compact instruction against the family `decode_compact` identifies. The
    /// match is exhaustive over the decoder's families. Payload fields carry the identity bits
    /// (op fields, the saturation gate, the data shape) so execution and naming read them once;
    /// operand fields - registers, constants, cross-path selectors - are read here as needed.
    fn execute_compact(
        &mut self,
        bus: &mut CoreBus<'_>,
        address: u32,
        packet_next: u32,
        word: u16,
        header: u32,
        family: CompactOpcode,
    ) {
        let side = if word & 1 == 0 { Bank::A } else { Bank::B };
        match family {
            CompactOpcode::Nop => {
                let count = ((word >> 13) & 7) as u32 + 1;
                self.stall_cycles = self.stall_cycles.max(count - 1);
            }
            CompactOpcode::Spmask | CompactOpcode::Spmaskr => {
                // The compact `SPMASKR` is also the reload point of a nested loop.
                if matches!(family, CompactOpcode::Spmaskr) {
                    self.arm_reload();
                }
            }
            CompactOpcode::Bnop => {
                // Predicated compact instructions select A0 or B0 with the s bit and invert
                // the sense with the z bit at position 4.
                let conditional = word & 0x002e == 0x002a;
                let unsigned = word & 0xc03e == 0xc00a || word & 0xc02e == 0xc02a;
                let predicate_register = self.read_register(side, 0);
                let predicate = !conditional || (predicate_register == 0) == (word & (1 << 4) != 0);
                if predicate {
                    let raw = if unsigned {
                        ((word >> 6) & 0xff) as u32
                    } else {
                        sign_extend(((word >> 6) & 0x7f) as u32, 7)
                    };
                    let target = (address & !0x1f).wrapping_add(raw << 1);
                    self.branches
                        .push((self.cycle + 6, BranchTarget::Address(target)));
                    self.branch_block_until = self.cycle + 6;
                }
                // The 8-bit compact BNOP forms have no NOP-count field: their count is
                // hardwired to five (OHWCST5 in the binutils table), so the branch fills all
                // of its own delay slots. The 7-bit forms carry the count in bits 13-15.
                let count = if unsigned {
                    5
                } else {
                    ((word >> 13) & 7) as u32
                };
                self.stall_cycles = self.stall_cycles.max(count);
            }
            CompactOpcode::Callp => {
                let displacement = sign_extend(((word >> 6) & 0x03ff) as u32, 10) << 2;
                let target = (address & !0x1f).wrapping_add(displacement);
                self.schedule(Bank::B, 3, packet_next, 0);
                self.branches
                    .push((self.cycle + 6, BranchTarget::Address(target)));
                self.branch_block_until = self.cycle + 6;
                self.stall_cycles = self.stall_cycles.max(5);
            }
            CompactOpcode::Mpy { op, saturating } => {
                // `m_m3`: the compact M unit multiplies. The header SAT bit picks the
                // saturating SMPY forms, which double the product, and the two-bit
                // destination field names an even register in the compact window.
                let src1 = self.read_register(side, compact_register((word >> 13) as u32, header));
                let src2_side = if word & (1 << 12) != 0 {
                    opposite(side)
                } else {
                    side
                };
                let src2 =
                    self.read_register(src2_side, compact_register((word >> 7) as u32, header));
                let dst = compact_register(((word >> 10) as u32 & 3) << 1, header);
                let low = |value: u32| i32::from(value as u16 as i16);
                let high = |value: u32| i32::from((value >> 16) as u16 as i16);
                let product = match op {
                    MpyOp::Mpy => low(src1) * low(src2),
                    MpyOp::MpyH => high(src1) * high(src2),
                    MpyOp::MpyLh => low(src1) * high(src2),
                    MpyOp::MpyHl => high(src1) * low(src2),
                };
                let (value, clamped) = if saturating {
                    match product.checked_mul(2) {
                        Some(doubled) => (doubled as u32, false),
                        None => (0x7fff_ffff, true),
                    }
                } else {
                    (product as u32, false)
                };
                if clamped {
                    // Same instruction, shorter encoding: SMPY reports through SSR and
                    // CSR.SAT whichever way it was encoded.
                    self.note_saturation(b'M', side);
                }
                self.schedule(side, dst, value, 1);
            }
            CompactOpcode::MvkWrite => {
                let condition = (word >> 14) & 3;
                let predicate = match condition {
                    0 => self.read_register(Bank::A, 0) != 0,
                    1 => self.read_register(Bank::A, 0) == 0,
                    2 => self.read_register(Bank::B, 0) != 0,
                    3 => self.read_register(Bank::B, 0) == 0,
                    _ => unreachable!(),
                };
                if predicate {
                    let dst = compact_register((word >> 7) as u32, header);
                    let value = ((word >> 13) & 1) as u32;
                    self.schedule(side, dst, value, 0);
                }
            }
            CompactOpcode::LdstB15(shape) => {
                // `d_dstk` addresses B15 with a scaled unsigned offset and never updates it.
                let data_bank = if word & (1 << 12) == 0 {
                    Bank::A
                } else {
                    Bank::B
                };
                let register = compact_register((word >> 4) as u32, header);
                let constant = (((word >> 7) & 7) << 2) | ((word >> 13) & 3);
                let address = self
                    .read_register(Bank::B, 15)
                    .wrapping_add((constant as u32) << 2);
                if shape.load {
                    let value = self.read_u32(bus, address & !3);
                    self.schedule_load(data_bank, register, value, 4);
                } else {
                    self.note_store(bus, address & !3);
                    self.writeback.schedule_store(StoreWrite {
                        due: self.cycle,
                        address: address & !3,
                        size: 4,
                        value: self.read_register(data_bank, register),
                    });
                }
            }
            CompactOpcode::LdstStack(shape) => {
                // `d_pp`: the stack-pointer forms. Loads pre-increment B15, stores
                // post-decrement it, and the register field is four bits wide and never
                // remapped by RS.
                let data_bank = if word & (1 << 12) == 0 {
                    Bank::A
                } else {
                    Bank::B
                };
                let register = ((word >> 7) & 0x0f) as usize;
                let load = shape.load;
                let doubleword = shape.size == 8;
                let count = ((word >> 13) & 1) as u32 + 1;
                let pointer = self.read_register(Bank::B, 15);
                let step = shape.size * count;
                let (base, updated) = if load {
                    let incremented = pointer.wrapping_add(step);
                    (incremented, incremented)
                } else {
                    (pointer, pointer.wrapping_sub(step))
                };
                self.schedule(Bank::B, 15, updated, 0);
                let aligned = if doubleword { base & !7 } else { base & !3 };
                if load {
                    if doubleword {
                        let (low, high) = doubleword_halves(
                            bus.endian(),
                            self.read_u32(bus, aligned),
                            self.read_u32(bus, aligned.wrapping_add(4)),
                        );
                        self.schedule_load(data_bank, register, low, 4);
                        self.schedule_load(data_bank, register + 1, high, 4);
                    } else {
                        let value = self.read_u32(bus, aligned);
                        self.schedule_load(data_bank, register, value, 4);
                    }
                } else {
                    self.note_store(bus, aligned);
                    if doubleword {
                        let (first, second) = doubleword_halves(
                            bus.endian(),
                            self.read_register(data_bank, register),
                            self.read_register(data_bank, register + 1),
                        );
                        self.schedule_doubleword_store(aligned, first, second);
                    } else {
                        self.writeback.schedule_store(StoreWrite {
                            due: self.cycle,
                            address: aligned,
                            size: 4,
                            value: self.read_register(data_bank, register),
                        });
                    }
                }
            }
            CompactOpcode::LdstPointer(shape) => {
                // `d_dinc` post-increments, `d_ddec` pre-decrements, `d_dind` adds a scaled
                // register offset and `d_doff4` a scaled four-bit constant, neither of the
                // latter two updating the pointer.
                let indexed = word & (1 << 10) != 0;
                let updating = indexed && word & (1 << 11) != 0;
                let data_bank = if word & (1 << 12) == 0 {
                    Bank::A
                } else {
                    Bank::B
                };
                let (size, signed, register, aligned_access) = compact_memory_shape(word, header);
                // The non-aligned doubleword forms encode their offsets unscaled - register
                // indices and constants count bytes, not doublewords (binutils
                // mem_offset_noscale on dinddw/doff4dw/dincdw/ddecdw).
                let scale = if size == 8 && !aligned_access {
                    1
                } else {
                    size
                };
                let pointer = 4 + ((word >> 7) & 3) as usize;
                let value = self.read_register(side, pointer);
                let amr = self.control.amr();
                let wrap = |updated: u32| circular_address(amr, side, pointer, value, updated);
                let (base, update) = if updating {
                    // The updating form's increment counts access-sized units: compiled
                    // chunk-striding loops advance eight bytes per iteration with cst=0
                    // (DSP_maxval_cn's vector loop).
                    let count = ((word >> 13) & 1) as u32 + 1;
                    let step = size * count;
                    if word & (1 << 14) != 0 {
                        let decremented = wrap(value.wrapping_sub(step));
                        (decremented, Some(decremented))
                    } else {
                        (value, Some(wrap(value.wrapping_add(step))))
                    }
                } else if indexed {
                    // A register-offset index is scaled by the access size for the aligned
                    // forms; only the non-aligned doubleword (dinddw) adds its offset
                    // register raw, in bytes - compiled _mem8 loops materialise a byte
                    // displacement there.
                    let offset =
                        self.read_register(side, compact_register((word >> 13) as u32, header));
                    let scaled = if size == 8 && !aligned_access {
                        offset
                    } else {
                        offset.wrapping_mul(size)
                    };
                    (wrap(value.wrapping_add(scaled)), None)
                } else {
                    let offset = (((word >> 13) & 7) | (((word >> 11) & 1) << 3)) as u32;
                    (wrap(value.wrapping_add(offset * scale)), None)
                };
                if let Some(updated) = update {
                    self.schedule(side, pointer, updated, 0);
                }
                if shape.load {
                    if size == 8 {
                        let aligned = if aligned_access { base & !7 } else { base };
                        let (low, high) = doubleword_halves(
                            bus.endian(),
                            self.read_u32(bus, aligned),
                            self.read_u32(bus, aligned.wrapping_add(4)),
                        );
                        self.schedule_load(data_bank, register, low, 4);
                        self.schedule_load(data_bank, register + 1, high, 4);
                    } else {
                        let aligned = if aligned_access {
                            base & !(size - 1)
                        } else {
                            base
                        };
                        let loaded = match size {
                            1 => self.read_u8(bus, aligned) as u32,
                            2 => self.read_u16(bus, aligned) as u32,
                            4 => self.read_u32(bus, aligned),
                            _ => unreachable!(),
                        };
                        let loaded = if signed {
                            sign_extend(loaded, size * 8)
                        } else {
                            loaded
                        };
                        self.schedule_load(data_bank, register, loaded, 4);
                    }
                } else if size == 8 {
                    let aligned = if aligned_access { base & !7 } else { base };
                    self.note_store(bus, aligned);
                    let (first, second) = doubleword_halves(
                        bus.endian(),
                        self.read_register(data_bank, register),
                        self.read_register(data_bank, register + 1),
                    );
                    self.schedule_doubleword_store(aligned, first, second);
                } else {
                    self.writeback.schedule_store(StoreWrite {
                        due: self.cycle,
                        address: if aligned_access {
                            base & !(size - 1)
                        } else {
                            base
                        },
                        size: size as u8,
                        value: self.read_register(data_bank, register),
                    });
                }
            }
            CompactOpcode::AddConst => {
                let dst = compact_register((word >> 4) as u32, header);
                let src = compact_register((word >> 7) as u32, header);
                // `scst_l3i`: the four-bit field is signed, except that zero encodes 8
                // because adding zero needs no instruction.
                let constant = compact_add_constant(word);
                let source_bank = if word & (1 << 12) == 0 {
                    side
                } else {
                    opposite(side)
                };
                let value = self.read_register(source_bank, src).wrapping_add(constant);
                self.schedule(side, dst, value, 0);
            }
            CompactOpcode::ShiftConst5(kind) => {
                // `s_ssh5`: shift by a five-bit constant. Op field 3 is not a shift here:
                // that encoding belongs to `s_s2sh`, the register-count form.
                let register = compact_register((word >> 7) as u32, header);
                let count = compact_constant5(word);
                let value = self.read_register(side, register);
                let result = match kind {
                    ShiftKind::Shl => value.checked_shl(count).unwrap_or(0),
                    ShiftKind::Shr => {
                        if count >= 32 {
                            u32::from((value as i32) < 0).wrapping_neg()
                        } else {
                            ((value as i32) >> count) as u32
                        }
                    }
                    ShiftKind::Shru => value.checked_shr(count).unwrap_or(0),
                    ShiftKind::Sshl => unreachable!("constant shift forms never encode Sshl"),
                };
                self.schedule(side, register, result, 0);
            }
            CompactOpcode::Spkernel => {
                self.sploop.end_loading();
                self.pipe_up = false;
            }
            CompactOpcode::Sploop { delayed, nested } => {
                self.sploop.start(LoopStart {
                    ii: (((word >> 7) & 7) | (((word >> 14) & 1) << 3)) as u32 + 1,
                    start_cycle: self.cycle,
                    start_address: address,
                    // The compact form encodes SPLOOP in op bit 0; the SPLOOPD and SPLOOPDR
                    // encodings both take the delayed ILC path. There is no compact SPLOOPW.
                    delayed: delayed && !self.pipe_up,
                    while_test: None,
                    // The predicated compact form (`nfu_uspldr`) is the nested-loop SPLOOPD.
                    // Its selector is only ever A0 or B0, chosen by bit 0, and never inverted:
                    // the z bit is part of the fixed pattern.
                    reload_test: nested.then_some((
                        if word & 1 == 0 { Bank::A } else { Bank::B },
                        0,
                        false,
                    )),
                    nested,
                });
            }
            CompactOpcode::ExtExtu { extu } => {
                // Compact EXT/EXTU with the fixed 16/16 and 24/24 field positions
                // (`s_s2ext`).
                let src = self.read_register(side, compact_register((word >> 7) as u32, header));
                let dst = compact_register((word >> 13) as u32, header);
                let shift = if (word >> 11) & 1 == 0 { 16 } else { 24 };
                let value = if extu {
                    (src << shift) >> shift
                } else {
                    (((src << shift) as i32) >> shift) as u32
                };
                self.schedule(side, dst, value, 0);
            }
            CompactOpcode::SetClrExtu(op) => {
                // `s_sc5`: single-bit SET and CLR, plus the EXTU form that extracts one bit,
                // all with a five-bit constant assembled from bits 13-15 and 11-12.
                let register = compact_register((word >> 7) as u32, header);
                let constant = compact_constant5(word);
                let value = self.read_register(side, register);
                // SET and CLR write the register they read; the single-bit EXTU form writes
                // A0 or B0, which is fully determined by the functional unit (`zreg` in
                // binutils).
                let (result, dst) = match op {
                    SetClrOp::Extu => ((value << constant) >> 31, 0),
                    SetClrOp::Set => (value | (1 << constant), register),
                    SetClrOp::Clr => (value & !(1 << constant), register),
                };
                self.schedule(side, dst, result, 0);
            }
            CompactOpcode::Addk => {
                // `s_sx5`: ADDK with an unsigned five-bit constant assembled from bits
                // 13-15 and 11-12.
                let dst = compact_register((word >> 7) as u32, header);
                let constant = compact_constant5(word);
                let value = self.read_register(side, dst).wrapping_add(constant);
                self.schedule(side, dst, value, 0);
            }
            CompactOpcode::AddawB15 => {
                // `d_dx5`: ADDAW from B15 with a word-scaled unsigned five-bit constant.
                let dst = compact_register((word >> 7) as u32, header);
                let constant = compact_constant5(word);
                let value = self.read_register(Bank::B, 15).wrapping_add(constant << 2);
                self.schedule(side, dst, value, 0);
            }
            CompactOpcode::AddawSubawB15 { subaw } => {
                // `d_dx5p`: ADDAW / SUBAW adjusting B15 in place, constant from bits 13-15
                // and 8-9.
                let constant = compact_stack_constant(word) << 2;
                let base = self.read_register(Bank::B, 15);
                let value = if subaw {
                    base.wrapping_sub(constant)
                } else {
                    base.wrapping_add(constant)
                };
                self.schedule(Bank::B, 15, value, 0);
            }
            CompactOpcode::AddSub2op { subtract } => {
                // `s_sx2op` and `d_dx2op`: two-operand ADD / SUB where the destination is
                // also the first source and bit 12 puts src2 on the cross path.
                let register = compact_register((word >> 13) as u32, header);
                let src2 = compact_register((word >> 7) as u32, header);
                let cross = word & (1 << 12) != 0;
                let left = self.read_register(side, register);
                let right = self.read_register(if cross { opposite(side) } else { side }, src2);
                let value = if subtract {
                    left.wrapping_sub(right)
                } else {
                    left.wrapping_add(right)
                };
                self.schedule(side, register, value, 0);
            }
            CompactOpcode::ShiftReg(kind) => {
                // `s_s2sh`: shifts by a register amount, destination doubling as the first
                // source. The `Sshl` form is the saturating shift left.
                let register = compact_register((word >> 7) as u32, header);
                let count =
                    self.read_register(side, compact_register((word >> 13) as u32, header)) & 0x3f;
                let value = self.read_register(side, register);
                let result = match kind {
                    ShiftKind::Shl => value.checked_shl(count).unwrap_or(0),
                    ShiftKind::Shr => {
                        if count >= 32 {
                            u32::from((value as i32) < 0).wrapping_neg()
                        } else {
                            ((value as i32) >> count) as u32
                        }
                    }
                    ShiftKind::Shru => value.checked_shr(count).unwrap_or(0),
                    ShiftKind::Sshl => {
                        let shifted = i64::from(value as i32) << count.min(40);
                        let (clamped, saturated) = saturate_i32(shifted);

                        if saturated {
                            // The compact shift is an S-unit instruction and reports its
                            // saturation the same way `SSHL .S` does.
                            self.note_saturation(b'S', side);
                        }
                        clamped
                    }
                };
                self.schedule(side, register, result, 0);
            }
            CompactOpcode::CmpeqConst3 => {
                // `l_lx3c`: CMPEQ against an unsigned three-bit constant. Like `l_l2c` its
                // one-bit destination names a predicate register directly, so RS never
                // remaps it.
                let src2 = self.read_register(side, compact_register((word >> 7) as u32, header));
                let dst = ((word >> 11) & 1) as usize;
                let constant = ((word >> 13) & 7) as u32;
                self.schedule(side, dst, u32::from(src2 == constant), 0);
            }
            CompactOpcode::ShiftCoded3 { left } => {
                // Compact S-unit shift with a coded 3-bit shift amount (`s_s3i`); the
                // format is only valid when the fetch packet header has no branch (BR) flag.
                let dst = compact_register((word >> 4) as u32, header);
                let src = compact_register((word >> 7) as u32, header);
                let x = word & (1 << 12) != 0;
                let count = match (word >> 13) & 7 {
                    0 => 16,
                    7 => 8,
                    value => value as u32,
                };
                let value = self.read_register(if x { opposite(side) } else { side }, src);
                let result = if left {
                    value.checked_shl(count).unwrap_or(0)
                } else {
                    ((value as i32) >> count.min(31)) as u32
                };
                self.schedule(side, dst, result, 0);
            }
            CompactOpcode::CmplConst1(op) => {
                // Compact L-unit compare against a 0/1 constant (`l_lx1c`), writing a
                // predicate register that the header RS bit never remaps.
                let src2 = self.read_register(side, compact_register((word >> 7) as u32, header));
                let dst = ((word >> 11) & 1) as usize;
                let constant = ((word >> 13) & 1) as u32;
                let value = match op {
                    CmpOp::Cmplt => u32::from((constant as i32) < (src2 as i32)),
                    CmpOp::Cmpgt => u32::from((constant as i32) > (src2 as i32)),
                    CmpOp::Cmpltu => u32::from(constant < src2),
                    CmpOp::Cmpgtu => u32::from(constant > src2),
                };
                self.schedule(side, dst, value, 0);
            }
            CompactOpcode::BnopReg => {
                // The compact register branch samples its register at issue too.
                let register = ((word >> 7) & 0x0f) as usize;
                let target = self.read_register(Bank::B, register) & !1u32;
                self.branches
                    .push((self.cycle + 6, BranchTarget::Address(target)));
                self.branch_block_until = self.cycle + 6;
                self.stall_cycles = self.stall_cycles.max(((word >> 13) & 7) as u32);
            }
            CompactOpcode::MvkConst8 => {
                let dst = compact_register((word >> 7) as u32, header);
                let constant = compact_constant8(word);
                self.schedule(side, dst, constant, 0);
            }
            CompactOpcode::MvkCond(form) => {
                let register = compact_register((word >> 7) as u32, header);
                let current = self.read_register(side, register);
                match form {
                    // Writing zero is named as a move of the constant, not as ZERO.
                    MvkForm::Mvk0 => self.schedule(side, register, 0, 0),
                    MvkForm::Mvk1 => self.schedule(side, register, 1, 0),
                    MvkForm::Add => self.schedule(side, register, current.wrapping_add(1), 0),
                    MvkForm::Xor => self.schedule(side, register, current ^ 1, 0),
                    MvkForm::Neg => self.schedule(side, register, 0u32.wrapping_sub(current), 0),
                    MvkForm::Sub => self.schedule(side, register, current.wrapping_sub(1), 0),
                    MvkForm::Mvc => self.writeback.schedule_control(ControlWrite {
                        due: self.cycle + 4,
                        register: 0x0d,
                        value: current,
                        user: self.user_mode,
                    }),
                }
            }
            CompactOpcode::Mv => {
                // `lsdmvfr` moves a 3-bit source (bits 7-9) into a 5-bit destination
                // assembled from bits 10-11 and 13-15; `lsdmvto` moves a 5-bit source
                // assembled from bits 10-11 and 7-9 into the 3-bit destination at bits
                // 13-15.
                let from_wide = word & 0x0066 == 0x0046;
                let high = (((word >> 10) & 3) << 3) as usize;
                let (src, dst) = if from_wide {
                    (
                        compact_register((word >> 7) as u32, header),
                        high | ((word >> 13) & 7) as usize,
                    )
                } else {
                    (
                        high | ((word >> 7) & 7) as usize,
                        compact_register((word >> 13) as u32, header),
                    )
                };
                let source_bank = if word & (1 << 12) == 0 {
                    side
                } else {
                    opposite(side)
                };
                self.schedule(side, dst, self.read_register(source_bank, src), 0);
            }
            CompactOpcode::Logic2op(op) => {
                // `l_l2c` writes A0/A1 or B0/B1: its one-bit destination is the `reg_nors`
                // form, which the header RS bit never remaps into the A16-A23 window.
                let src1 = compact_register((word >> 13) as u32, header);
                let src2 = compact_register((word >> 7) as u32, header);
                let dst = ((word >> 4) & 1) as usize;
                let x = word & (1 << 12) != 0;
                let left = self.read_register(side, src1);
                let right = self.read_register(if x { opposite(side) } else { side }, src2);
                let value = match op {
                    LogicOp::And => left & right,
                    LogicOp::Or => left | right,
                    LogicOp::Xor => left ^ right,
                    LogicOp::Cmpeq => u32::from(left == right),
                    LogicOp::Cmplt => u32::from((left as i32) < (right as i32)),
                    LogicOp::Cmpgt => u32::from((left as i32) > (right as i32)),
                    LogicOp::Cmpltu => u32::from(left < right),
                    LogicOp::Cmpgtu => u32::from(left > right),
                };
                self.schedule(side, dst, value, 0);
            }
            CompactOpcode::MvkConst5 => {
                let constant = compact_constant5(word);
                let dst = compact_register((word >> 7) as u32, header);
                self.schedule(side, dst, sign_extend(constant, 5), 0);
            }
            CompactOpcode::Alu3op(op) => {
                // Compact three-operand ALU on the L and S units. The SAT bit in the fetch
                // packet header selects the saturating SADD/SSUB forms; SSUB has no .S unit
                // form, so only the .L unit's subtract saturates.
                let src1 = compact_register((word >> 13) as u32, header);
                let src2 = compact_register((word >> 7) as u32, header);
                let dst = compact_register((word >> 4) as u32, header);
                let x = word & (1 << 12) != 0;
                let left = self.read_register(side, src1) as i32;
                let right = self.read_register(if x { opposite(side) } else { side }, src2) as i32;
                let (value, clamped) = match op {
                    AluOp::Add => ((left as u32).wrapping_add(right as u32), false),
                    AluOp::Sub => ((left as u32).wrapping_sub(right as u32), false),
                    AluOp::Sadd => saturate_i32(i64::from(left) + i64::from(right)),
                    AluOp::Ssub => saturate_i32(i64::from(left) - i64::from(right)),
                };
                if clamped {
                    // A compact encoding is a re-encoding of the same instruction, so it
                    // sets SSR and CSR.SAT exactly as the 32-bit form does. The L-unit
                    // forms have bits 1-3 clear; everything else here is the S unit.
                    let unit = if (word >> 1) & 7 == 0 { b'L' } else { b'S' };

                    self.note_saturation(unit, side);
                }
                self.schedule(side, dst, value, 0);
            }
        }
    }

    /// Record that a functional unit saturated. SSR names the unit by side, so the bit is
    /// L1, L2, S1, S2, M1, M2 in that order.
    fn note_saturation(&mut self, unit: u8, side: Bank) {
        self.stats.saturations += 1;
        let bit = match (unit, side) {
            (b'L', Bank::A) => 0,
            (b'L', Bank::B) => 1,
            (b'S', Bank::A) => 2,
            (b'S', Bank::B) => 3,
            (b'M', Bank::A) => 4,
            _ => 5,
        };
        self.writeback.schedule_saturation(self.cycle + 1, bit);
    }

    /// The same-cycle duplicate-write fault, kept cold so `schedule`'s hot path does not carry
    /// the `format!` argument materialisation for a branch it almost never takes.
    #[cold]
    #[inline(never)]
    fn fault_duplicate_write(&mut self, bank: Bank, register: usize) {
        self.raise_fault(
            format!(
                "multiple writes to side {}{register} in one execute packet",
                bank_name(bank)
            ),
            1 << 4,
        );
    }

    fn schedule(&mut self, bank: Bank, register: usize, value: u32, delay: u64) {
        let write = RegisterWrite::new(
            self.cycle,
            delay,
            bank,
            register,
            value,
            self.scheduling_load,
        );
        if self.writeback.schedule_register(write).is_err() {
            // Two instructions in one execute packet writing the same register is the conflict
            // SPRUFE8B section 3.8.8 calls "easily detectable": the assembler rejects it, and a
            // packet that contains it is as malformed as one that claims a unit twice.
            self.fault_duplicate_write(bank, register);
            return;
        }
        if self.trace_a4 && register == 4 {
            eprintln!(
                "trace cycle {} schedule side {}{register} due {} value {value:#010x} load {} pc {:#010x}",
                self.cycle,
                bank_name(bank),
                write.due,
                self.scheduling_load,
                self.pc
            );
        }
    }

    /// Schedule a load destination, which is exempt from the cross path stall rule.
    fn schedule_load(&mut self, bank: Bank, register: usize, value: u32, delay: u64) {
        self.scheduling_load = true;
        self.schedule(bank, register, value, delay);
        self.scheduling_load = false;
    }

    /// An instruction that reads a register across a cross path one cycle after a non-load
    /// instruction wrote it costs one stall cycle (SPRUFE8B §3.8.4).
    fn cross_path_stall(&self, packet: &[Fetched]) -> bool {
        packet.iter().any(|fetched| {
            fetched
                .inst
                .resources()
                .ok()
                .and_then(|(_, cross, _)| cross)
                .is_some_and(|(side, register)| {
                    let bank = match opposite(side) {
                        Bank::A => 0,
                        Bank::B => 1,
                    };
                    self.previous_writes[bank] & (1 << register) != 0
                })
        })
    }

    /// A stall cycle: the pipeline is frozen, so only the peripherals advance. The record of
    /// the previous cycle's writes is dropped because the hazard it describes is now two
    /// cycles old.
    fn freeze(&mut self, bus: &mut CoreBus<'_>) {
        self.stalls += 1;
        self.previous_writes = [0; 2];
        let _ = self.tick_device(bus);
    }

    /// Tell the hierarchy about a data load; the cycle's bank conflicts and read misses are
    /// costed from what it has been told.
    fn note_load(&mut self, bus: &mut CoreBus<'_>, address: u32) {
        self.prot_load |= self.prot_packet;
        bus.note_load(address);
    }

    /// A doubleword store as two word writes in this cycle, `first` at `address` and `second`
    /// four bytes above it; `doubleword_halves` has already put the pair in memory order.
    fn schedule_doubleword_store(&mut self, address: u32, first: u32, second: u32) {
        for (offset, value) in [(0, first), (4, second)] {
            self.writeback.schedule_store(StoreWrite {
                due: self.cycle,
                address: address.wrapping_add(offset),
                size: 4,
                value,
            });
        }
    }

    /// Tell the hierarchy about a data store. It answers at once with the stall cycles the store
    /// costs - the write buffer holding the CPU when all four of its entries are busy.
    fn note_store(&mut self, bus: &mut CoreBus<'_>, address: u32) {
        let clock = self.data_clock();
        self.pending_stalls += bus.note_store(address, clock);
    }

    /// The clocks the hierarchy times this cycle's data accesses on: the CPU's own, because the
    /// stalls it has served and the ones still queued are the pipeline's state.
    fn data_clock(&self) -> DataClock {
        DataClock {
            cycle: self.cycle,
            elapsed: self.cycles(),
            queued: self.pending_stalls,
        }
    }

    /// The write-back at the end of a cycle. Nothing due is the common shape of a stall cycle,
    /// and of any cycle whose packets scheduled nothing that lands now; the queue answers that
    /// with one mask test, so the check is inlined into every caller and only a cycle with
    /// something due enters the out-of-line drain.
    #[inline(always)]
    fn commit_writes(&mut self, bus: &mut CoreBus<'_>) {
        self.previous_writes = [0; 2];
        if !self.writeback.nothing_due(self.cycle) {
            self.commit_due_writes(bus);
        }
    }

    /// Perform everything the write-back queue has due this cycle: register writes, then
    /// stores, then control-register writes, then saturation flags.
    #[inline(never)]
    fn commit_due_writes(&mut self, bus: &mut CoreBus<'_>) {
        let cycle = self.cycle;
        let traced = self.trace.enabled();
        // RegisterWrite is Copy, so the queue's borrow ends at the pop.
        while let Some(write) = self.writeback.pop_register(cycle) {
            let register = usize::from(write.register);
            if !write.from_load {
                let bank = match write.bank {
                    Bank::A => 0,
                    Bank::B => 1,
                };
                self.previous_writes[bank] |= 1 << register;
            }
            if traced {
                let name = format!("{}{}", bank_letter(write.bank), register);
                let old = self.read_bank(write.bank, register);
                self.trace.emit(
                    cycle,
                    Event::Write {
                        name,
                        old,
                        new: write.value,
                    },
                );
            }
            self.write_register(write.bank, register, write.value);
        }
        while let Some(store) = self.writeback.pop_store(cycle) {
            // A page that does not grant the write swallows it and reports the fault.
            let user = self.user_mode;
            if bus.check_access(store.address, Permission::Write, user) {
                if let Some(observation) = self.stack_observation.as_mut() {
                    let lo = u64::from(store.address);
                    let hi = lo + u64::from(store.size);
                    let stack_lo = u64::from(observation.start);
                    let stack_hi = u64::from(observation.end);
                    if lo < stack_hi && hi > stack_lo {
                        if lo < stack_lo || hi > stack_hi {
                            observation.out_of_bounds_write = true;
                        }
                        let address = store.address.max(observation.start);
                        if observation
                            .lowest_write
                            .map_or(true, |current| address < current)
                        {
                            observation.lowest_write = Some(address);
                            observation.lowest_write_cycle = Some(cycle);
                        }
                    }
                }
                self.stats.stores += 1;
                if traced {
                    // The old value comes from the host view: reading it through the
                    // system would allocate a cache line, and tracing must be passive.
                    let old = bus
                        .host_read_bytes(store.address, store.size as usize)
                        .iter()
                        .rev()
                        .fold(0u32, |value, byte| (value << 8) | u32::from(*byte));
                    self.trace.emit(
                        cycle,
                        Event::Store {
                            address: store.address,
                            size: store.size,
                            old,
                            new: store.value,
                        },
                    );
                }
                match store.size {
                    1 => bus.write_u8(store.address, store.value as u8),
                    2 => bus.write_u16(store.address, store.value as u16),
                    4 => bus.write32(store.address, store.value),
                    _ => unreachable!(),
                }
            }
        }
        let mut cursor = 0;
        while let Some(write) = self.writeback.next_control(cycle, &mut cursor) {
            let noted = traced
                .then(|| crate::control::register_name(write.register))
                .flatten()
                .map(|name| (name, self.control.read(write.register, self.packet_address)));
            if write.user {
                self.control.write_user(write.register, write.value);
            } else {
                self.control.write(write.register, write.value);
            }
            if let Some((name, old)) = noted {
                self.trace.emit(
                    cycle,
                    Event::Control {
                        name,
                        old,
                        // The register may narrow or ignore bits, so report what it now
                        // holds rather than what the instruction offered.
                        new: self.control.read(write.register, self.packet_address),
                    },
                );
            }
        }
        // A functional unit's saturation flag appears one cycle later and beats an MVC write.
        let saturated = self.writeback.take_saturations(cycle);
        for bit in 0..8 {
            if saturated & (1 << bit) != 0 {
                self.control.note_saturation(bit);
            }
        }
    }
    /// A CPU-detected fault. With `TSR.GEE` set the architecture turns it into an internal
    /// exception whose cause the handler reads out of IERR; with exceptions disabled there is
    /// no defined behaviour to model, so the run stops with a diagnostic instead.
    fn raise_fault(&mut self, message: String, ierr: u32) {
        self.control.raise_ierr(ierr);
        self.stats.faults += 1;
        if self.trace.enabled() {
            self.trace.emit(
                self.cycle,
                Event::Fault {
                    message: message.clone(),
                    ierr,
                },
            );
        }
        if self.control.exceptions_enabled() {
            // The firmware will take this as an internal exception and the emulator keeps
            // running, so the cause is recorded for a diagnostic caller to report: an emulator
            // defect and a firmware bug look identical from outside without it.
            self.last_exception_cause = Some(format!("{message} (fetch pc {:#010x})", self.pc));
            self.pending_exception.get_or_insert(Exception::Internal);
            return;
        }
        if self.fault.is_none() {
            self.fault = Some(message);
        }
    }

    /// A control register User mode may not reach in either direction. The access does not
    /// happen and the exception reports both the resource and the privilege (SPRU732J 8.2.4.1).
    fn restricted_register(&mut self, address: u32, register: u8) -> bool {
        if !self.user_mode || !restricted_in_user(register) {
            return false;
        }
        self.raise_fault(
            format!("user-mode access to control register {register:#04x} at {address:#010x}"),
            IERR_RAX | IERR_PRX,
        );
        true
    }

    /// `B IRP`, `B NRP` and `IDLE` are Supervisor-only; in User mode they raise an opcode
    /// exception with the privilege bit alongside it (SPRU732J 8.2.4.3).
    fn restricted_instruction(&mut self, address: u32, name: &str) -> bool {
        if !self.user_mode {
            return false;
        }
        self.raise_fault(
            format!("user-mode {name} at {address:#010x}"),
            IERR_OPX | IERR_PRX,
        );
        true
    }

    /// Interrupt entry, the one way a cycle boundary vectors to a handler whatever the
    /// source. The source decides the return address and what becomes of an SPLOOP; the
    /// control registers save the context; everything after is the same for every source:
    /// no branch survives, the vector runs in Supervisor with any restore the interrupted
    /// code had queued cancelled with the branch that would have applied it, the pipeline
    /// refills, and remaining multicycle NOPs - IDLE and generated NOPs (BNOP/ADDKPC/CALLP
    /// and PROT loads) included - are annulled (SPRU732J §5.6.2 / §6.4.2). Memory stalls
    /// live in `pending_stalls` and still delay processing.
    fn enter(&mut self, entry: Entry) {
        let from = self.pc;
        let interrupted = match entry {
            // NRP holds the address of the first annulled execute packet, which differs by
            // source: `SWE`/`SWENR` complete, so the first annulled packet is the one after
            // them, while an illegal opcode or a resource conflict annuls the offending
            // packet itself - Linux relies on exactly that when it reads the faulting word
            // back through NRP to recognise its breakpoint opcode (`arch/c6x/kernel/traps.c`,
            // `process_iexcept`). A service request, an NMI and EXCEP all arrive between
            // packets, so for them the first annulled packet is the one `pc` names.
            //
            // An exception may interrupt a branch shadow, and leaves the loop buffer idle
            // without running an epilog.
            Entry::Exception(source) => {
                let in_sploop = self.sploop.abandon();
                self.interrupted_sploop = None;
                self.pipe_up = false;
                Interrupted {
                    return_pc: if source == Exception::Internal {
                        self.packet_address
                    } else {
                        self.pc
                    },
                    in_sploop,
                    interrupts_blocked: !self.branches.is_empty()
                        || self.cycle < self.branch_block_until,
                }
            }
            // An interrupt waits out branch shadows and SPLOOP drains, so it arrives between
            // packets with the loop, if any, already piped down for the return to resume.
            Entry::Interrupt(_) | Entry::NmiInterrupt => Interrupted {
                return_pc: self.pc,
                in_sploop: self.interrupted_sploop.take().is_some(),
                interrupts_blocked: false,
            },
        };
        self.pc = self.control.enter(entry, interrupted);
        if matches!(
            entry,
            Entry::NmiInterrupt | Entry::Exception(Exception::Nmi)
        ) {
            self.nmi_pending = false;
        }
        self.branches.clear();
        self.branch_block_until = self.cycle;
        self.user_mode = false;
        self.mode_change = None;
        self.interrupt_stall = 9;
        self.stall_cycles = 0;
        self.idling = false;
        match entry {
            Entry::Interrupt(interrupt) => {
                self.note_interrupt(interrupt, from);
                self.record_interrupt_take();
            }
            Entry::NmiInterrupt => {
                self.note_interrupt(0, from);
                self.record_interrupt_take();
            }
            Entry::Exception(_) => {}
        }
    }

    /// The interrupt-latency statistic: an interrupt that became ready while a branch shadow
    /// or an SPLOOP drain blocked the take is measured from the cycle it became ready.
    fn record_interrupt_take(&mut self) {
        if let Some(since) = self.interrupt_ready_since.take() {
            self.stats.interrupt_wait_cycles += self.cycle - since;
            self.stats.interrupt_waited += 1;
        }
    }

    /// End a committed cycle by committing writes, completing its SPLOOP stage if any, then
    /// ticking the device, sampling interrupts, handling interrupt entry, and advancing the
    /// pipeline cycle. A frozen stall only ticks the device and counts a stall: writes and
    /// SPLOOP must not advance while the pipeline is frozen.
    fn cycle_boundary(&mut self, bus: &mut CoreBus<'_>, end: CycleEnd) {
        if matches!(end, CycleEnd::Frozen) {
            self.freeze(bus);
            return;
        }
        self.commit_writes(bus);
        if let CycleEnd::LoopStage(offset) = end {
            self.complete_loop_cycle(offset);
        }
        self.finish_cycle(bus);
    }

    fn finish_cycle(&mut self, bus: &mut CoreBus<'_>) {
        let short_tick = self.tick_device(bus);
        // The interrupt lines and the exception combiner's output are one question about the
        // megamodule's interrupt controller, asked once per cycle.
        let (lines, exception_pending) = bus.take_interrupt_state(short_tick);
        if let Some(interrupt) = self.control.tick(self.cycle, lines) {
            bus.record_dropped_interrupt(interrupt);
        }
        let takeable_any = self.control.takeable_interrupt(true);
        if self.interrupt_stall == 0
            && !self.nmi_pending
            && self.pending_exception.is_none()
            && self.branches.is_empty()
            && self.mode_change.is_none()
            && !self.sploop.active()
            && !self.sploop_was_active
            && takeable_any.is_none()
            && (!self.control.external_exception_ready() || !exception_pending)
        {
            self.cycle += 1;
            return;
        }
        // Nothing to take: the shape of every cycle a loop buffer keeps out of the fast path
        // above. The exception and interrupt blocks below can only act on one of these, so
        // they are skipped as a group rather than each re-testing its own condition.
        if self.nmi_pending
            || self.pending_exception.is_some()
            || takeable_any.is_some()
            || exception_pending
        {
            self.take_exceptions_and_interrupts(takeable_any, exception_pending);
        }
        self.end_cycle();
    }

    /// The exception and interrupt takes of a cycle boundary, out of line because most cycles
    /// have none pending.
    #[inline(never)]
    fn take_exceptions_and_interrupts(
        &mut self,
        takeable_any: Option<u8>,
        exception_pending: bool,
    ) {
        let branches_clear = self.branches.is_empty() && self.cycle >= self.branch_block_until;
        // With GEE set, NMI is the highest-priority exception and is not held off by branch
        // delay slots. Other exceptions follow, then the maskable EXCEP input.
        if self.interrupt_stall == 0 && self.nmi_pending && self.control.nmi_exception_ready() {
            self.enter(Entry::Exception(Exception::Nmi));
        } else if let Some(pending) = self.pending_exception.take() {
            self.enter(Entry::Exception(pending));
        } else if self.interrupt_stall == 0
            && self.control.external_exception_ready()
            && exception_pending
        {
            self.enter(Entry::Exception(Exception::External));
        }
        if self.interrupt_stall != 0 {
            return;
        }
        // NMI uses the interrupt rules while GEE is clear, ahead of every maskable interrupt.
        let interrupt = if self.nmi_pending && self.control.nmi_interrupt_ready(branches_clear) {
            Some(Entry::NmiInterrupt)
        } else {
            self.control
                .takeable_interrupt(branches_clear)
                .map(Entry::Interrupt)
        };
        match interrupt {
            // A loop is never left mid-iteration: the interrupt waits while the buffer drains,
            // and only then is it taken, returning to the SPLOOP packet so the loop pipes back
            // up (SPRUFE8B 7.13.1 and 7.13.4).
            Some(_) if self.sploop.active() => {
                if self.ready_to_drain_for_interrupt() {
                    self.begin_interrupt_drain();
                }
                self.interrupt_ready_since.get_or_insert(self.cycle);
            }
            Some(entry) => self.enter(entry),
            // Ready, but a branch shadow defers the take until the delay slots finish.
            None if takeable_any.is_some() => {
                self.interrupt_ready_since.get_or_insert(self.cycle);
            }
            None => {}
        }
    }

    /// Advance the pipeline cycle, resolve branches and mode changes that come due, and
    /// update the loop buffer's bookkeeping.
    fn end_cycle(&mut self) {
        self.cycle += 1;
        // A loop cycle reaches here with no branch in flight most of the time, and both the
        // search and the retain are calls, so the empty case is tested inline.
        if !self.branches.is_empty() {
            if let Some(target) = self
                .branches
                .iter()
                .filter(|(due, _)| *due == self.cycle)
                .map(|(_, target)| *target)
                .next_back()
            {
                let BranchTarget::Address(address) = target;
                self.pc = address;
                self.stats.branches_taken += 1;
                if self.trace.enabled() {
                    self.trace.emit(self.cycle, Event::Branch { to: address });
                }
                // A taken branch makes the loop buffer idle once its last delay slot has passed
                // (SPRU732J §7.9), which is what lets the compiler put a conditional return in
                // front of an SPLOOP that must then never run. A nested loop is exempt: the
                // branch that closes its outer loop body fires while the buffer holds the inner
                // loop and only redirects the frozen fetch stream.
                self.sploop.end_on_branch();
            }
            self.branches.retain(|(due, _)| *due > self.cycle);
        }
        if let Some((due, user)) = self.mode_change {
            if due <= self.cycle {
                self.user_mode = user;
                self.mode_change = None;
            }
        }
        self.sploop.tick_reload_overlap();
        // One place reports the loop buffer coming and going, rather than every teardown site.
        let sploop_active = self.sploop.active();
        if sploop_active != self.sploop_was_active && self.trace.enabled() {
            let event = match self.sploop.interval() {
                Some(ii) => Event::SploopStart { ii },
                None => Event::SploopEnd,
            };
            self.trace.emit(self.cycle, event);
        }
        self.sploop_was_active = sploop_active;
    }

    #[inline(always)]
    fn read_register(&self, bank: Bank, index: usize) -> u32 {
        match bank {
            Bank::A => self.a[index],
            Bank::B => self.b[index],
        }
    }

    fn write_register(&mut self, bank: Bank, index: usize, value: u32) {
        if bank == Bank::B && index == 15 {
            if let Some(observation) = self.stack_observation.as_mut() {
                if (observation.start..=observation.end).contains(&value) {
                    observation.initialized = true;
                    if value < observation.min_sp {
                        observation.min_sp = value;
                        observation.min_sp_cycle = self.cycle;
                    }
                } else if observation.initialized {
                    observation.out_of_bounds_sp = true;
                }
            }
        }
        match bank {
            Bank::A => self.a[index] = value,
            Bank::B => self.b[index] = value,
        }
    }

    /// Counts an interrupt entry and records it in the trace. `interrupt` is 0 for NMI, which
    /// is what INTC numbers it.
    fn note_interrupt(&mut self, interrupt: u8, from: u32) {
        self.stats.interrupts += 1;
        if self.trace.enabled() {
            self.trace.emit(
                self.cycle,
                Event::Interrupt {
                    interrupt,
                    from,
                    to: self.pc,
                },
            );
        }
    }

    fn read_bank(&self, bank: Bank, index: usize) -> u32 {
        match bank {
            Bank::A => self.a[index],
            Bank::B => self.b[index],
        }
    }
    /// Data reads go through these three accessors so that every addressing form - wide,
    /// compact, long-offset - records its access for the PROT NOP, bank conflict and L1D miss
    /// rules exactly once, and meets the memory protection unit exactly once. A denied read
    /// returns zero and has already raised the unit's fault event.
    fn read_u8(&mut self, bus: &mut CoreBus<'_>, address: u32) -> u8 {
        self.note_load(bus, address);
        let user = self.user_mode;
        if !bus.check_access(address, Permission::Read, user) {
            return 0;
        }
        bus.read_u8(address)
    }

    fn read_u16(&mut self, bus: &mut CoreBus<'_>, address: u32) -> u16 {
        self.note_load(bus, address);
        let user = self.user_mode;
        if !bus.check_access(address, Permission::Read, user) {
            return 0;
        }
        bus.read_u16(address)
    }

    fn read_u32(&mut self, bus: &mut CoreBus<'_>, address: u32) -> u32 {
        self.note_load(bus, address);
        let user = self.user_mode;
        if !bus.check_access(address, Permission::Read, user) {
            return 0;
        }
        bus.read32(address)
    }
}

fn validate_packets<'a>(
    packets: impl IntoIterator<Item = &'a [Fetched]>,
) -> Result<(), (String, u32)> {
    // There are eight functional units. A fixed array avoids allocating in every execute
    // cycle; a ninth claim necessarily conflicts with one of the first eight.
    let mut units: [Option<((u8, Bank), u32)>; 8] = [None; 8];
    let mut unit_count = 0;
    let mut cross_paths: [Option<(u8, u8)>; 2] = [None, None];
    let mut memory_paths = [false; 2];
    for fetched in packets.into_iter().flat_map(|packet| packet.iter()) {
        let (unit, cross, memory_path) = fetched.inst.resources()?;
        if let Some(unit) = unit {
            if let Some((_, prior)) = units[..unit_count]
                .iter()
                .flatten()
                .find(|(claimed, _)| *claimed == unit)
            {
                return Err((
                    format!(
                        "functional unit conflict on .{}{} at {:#010x} and {:#010x}",
                        unit.0 as char,
                        bank_name(unit.1),
                        prior,
                        fetched.inst.address
                    ),
                    IERR_RCX,
                ));
            }
            units[unit_count] = Some((unit, fetched.inst.address));
            unit_count += 1;
        }
        if let Some((side, register)) = cross {
            let slot = bank_index(side);
            match &mut cross_paths[slot] {
                Some((prior, count)) if *prior == register && *count < 2 => *count += 1,
                Some((prior, _)) => {
                    return Err((
                        format!(
                            "cross-path conflict on {}X at {:#010x}: reads {}{} after {}{}",
                            slot + 1,
                            fetched.inst.address,
                            register_file_name(opposite(side)),
                            register,
                            register_file_name(opposite(side)),
                            prior
                        ),
                        IERR_RCX,
                    ));
                }
                None => cross_paths[slot] = Some((register, 1)),
            }
        }
        if let Some(path) = memory_path {
            let slot = bank_index(path);
            if memory_paths[slot] {
                return Err((
                    format!("multiple memory operations on T{}", slot + 1),
                    IERR_RCX,
                ));
            }
            memory_paths[slot] = true;
        }
    }
    Ok(())
}

fn is_loop_start(fetched: Fetched) -> bool {
    match fetched.inst.decoded() {
        Some(DecodedOpcode::Wide(opcode)) => {
            matches!(opcode, Opcode::Sploop | Opcode::Sploopd | Opcode::Sploopw)
        }
        Some(DecodedOpcode::Compact(family)) => {
            matches!(family, CompactOpcode::Sploop { .. })
        }
        None => false,
    }
}

/// True for NOP encodings, including the compact form and multicycle NOP counts.
fn is_nop(fetched: Fetched) -> bool {
    matches!(
        fetched.inst.decoded(),
        Some(DecodedOpcode::Wide(Opcode::Nop) | DecodedOpcode::Compact(CompactOpcode::Nop))
    )
}
fn spmask_bits(fetched: Fetched) -> Option<u8> {
    match (fetched.inst.encoding, fetched.inst.decoded()) {
        (Encoding::Wide(_), Some(DecodedOpcode::Wide(Opcode::Spmask | Opcode::Spmaskr)))
        | (
            Encoding::Compact(..),
            Some(DecodedOpcode::Compact(CompactOpcode::Spmask | CompactOpcode::Spmaskr)),
        ) => Some(spmask_mask(fetched.inst.encoding)),
        _ => None,
    }
}

fn fetched_unit_mask(fetched: Fetched) -> Option<u8> {
    let (unit, _, _) = fetched.inst.resources().ok()?;
    let (unit, bank) = unit?;
    unit_mask_bit(unit, bank)
}

fn binary_operands(word: u32, side: Bank, a: &[u32; 32], b: &[u32; 32]) -> (u32, u32, usize) {
    let l_format = word & 0x1c == 0x18;
    let d_ext_format = word & 0xc3c == 0x830;
    let op = if l_format {
        (word >> 5) & 0x7f
    } else if d_ext_format {
        (word >> 6) & 0x0f
    } else {
        (word >> 6) & 0x3f
    };
    let immediate = if l_format {
        matches!(
            op,
            0x02 | 0x06
                | 0x0e
                | 0x12
                | 0x46
                | 0x4e
                | 0x52
                | 0x56
                | 0x5e
                | 0x6a
                | 0x6e
                | 0x7a
                | 0x7e
        )
    } else if d_ext_format {
        matches!(op, 0x03 | 0x07 | 0x0b | 0x0d | 0x0f)
    } else {
        matches!(op, 0x06 | 0x0a | 0x16 | 0x18 | 0x19 | 0x1a | 0x1e | 0x22)
    };
    let src1 = ((word >> 13) & 0x1f) as usize;
    let src2 = ((word >> 18) & 0x1f) as usize;
    let x = word & (1 << 12) != 0;
    let src2_value = read_snapshot(if x { opposite(side) } else { side }, src2, a, b);
    debug_assert_eq!(immediate, binary_operand_shape(word).immediate);
    // Every constant form on this path takes a 5-bit signed constant except the unsigned
    // compares: CMPGTU and CMPLTU (L unit opfields 0x4e and 0x5e) take a 5-bit *unsigned*
    // constant on C64x and above, which is what lets the compiler realise `x <u 16` as
    // `CMPGTU 16,x,dst` (binutils tic6x opcode table, `ENC(src1, ucst)` with RAN(src1, 16, 31)).
    let immediate_value = if l_format && matches!(op, 0x4e | 0x5e) {
        src1 as u32
    } else {
        sign_extend(src1 as u32, 5)
    };
    let (left, right) = if d_ext_format && op == 0x0d {
        (src2_value, immediate_value)
    } else if l_format && op == 0x1f {
        // The second SSUB int form crosses src1, not src2 (SPRUFE8B SSUB opcode map).
        (
            read_snapshot(if x { opposite(side) } else { side }, src1, a, b),
            read_snapshot(side, src2, a, b),
        )
    } else {
        let left = if immediate {
            immediate_value
        } else {
            read_snapshot(side, src1, a, b)
        };
        (left, src2_value)
    };
    (left, right, ((word >> 23) & 0x1f) as usize)
}

/// Circular addressing. AMR gives each of A4-A7 and B4-B7 a two-bit mode selecting linear
/// addressing or a block size from BK0 or BK1; in circular mode only bits N through 0 of the
/// address arithmetic change, so the pointer wraps inside a 2^(N+1)-byte block and the bits
/// above stay as they were in the base register (SPRUFE8B sections 2.8.3 and 3.9.2).
fn circular_address(amr: u32, side: Bank, register: usize, base: u32, updated: u32) -> u32 {
    if !(4..=7).contains(&register) {
        return updated;
    }
    let shift = (register - 4) * 2 + if matches!(side, Bank::B) { 8 } else { 0 };
    let block = match (amr >> shift) & 3 {
        1 => (amr >> 16) & 0x1f,
        2 => (amr >> 21) & 0x1f,
        _ => return updated,
    };
    if block >= 31 {
        return updated;
    }
    let mask = (1u32 << (block + 1)) - 1;
    (base & !mask) | (updated & mask)
}

/// The address generator's fields for a 32-bit D-unit access (SPRUFE8B Table 3-11): which
/// register file, the base register, whether the offset is a register or a constant, whether it is
/// added or subtracted, and whether the base is updated before or after the access. Both the
/// executor and the operand rendering used by `tools/dis6x_diff.py` read the encoding through
/// this, so there is one reading of it rather than two.
fn memory_address(
    word: u32,
    amr: u32,
    a: &[u32; 32],
    b: &[u32; 32],
    scale: u32,
) -> (u32, Option<(usize, u32)>) {
    let shape = address_shape(word);
    let side = shape.side;
    let mode = (word >> 9) & 0xf;
    let base_register = shape.base;
    let base = read_snapshot(side, base_register, a, b);
    let offset = match shape.offset_register {
        Some(register) => read_snapshot(side, register, a, b),
        None => shape.offset,
    } << scale;
    let modified = if mode & 1 == 0 {
        base.wrapping_sub(offset)
    } else {
        base.wrapping_add(offset)
    };
    let modified = circular_address(amr, side, base_register, base, modified);
    if mode < 8 {
        (modified, None)
    } else if mode & 2 == 0 {
        (modified, Some((base_register, modified)))
    } else {
        (base, Some((base_register, modified)))
    }
}

/// Saturate a wide value to a signed 32-bit result, reporting whether it clamped.
fn saturate_i32(wide: i64) -> (u32, bool) {
    let clamped = wide.clamp(i32::MIN as i64, i32::MAX as i64);
    (clamped as u32, clamped != wide)
}

/// The packed helpers with saturation reporting: the operation returns its lane result and
/// whether that lane saturated.
fn packed_16_sat(left: u32, right: u32, operation: fn(u16, u16) -> (u16, bool)) -> (u32, bool) {
    let (low, low_sat) = operation(left as u16, right as u16);
    let (high, high_sat) = operation((left >> 16) as u16, (right >> 16) as u16);
    (u32::from(low) | u32::from(high) << 16, low_sat || high_sat)
}

fn packed_8_sat(left: u32, right: u32, operation: fn(u8, u8) -> (u8, bool)) -> (u32, bool) {
    let mut result = 0u32;
    let mut saturated = false;
    for lane in 0..4 {
        let shift = lane * 8;
        let (value, lane_sat) = operation((left >> shift) as u8, (right >> shift) as u8);
        saturated |= lane_sat;
        result |= u32::from(value) << shift;
    }
    (result, saturated)
}

fn saturate_i16(wide: i32) -> (u16, bool) {
    let clamped = wide.clamp(-32768, 32767);
    (clamped as u16, clamped != wide)
}

fn saturate_u16(wide: i32) -> (u16, bool) {
    let clamped = wide.clamp(0, 65535);
    (clamped as u16, clamped != wide)
}

fn packed_16(left: u32, right: u32, operation: fn(u16, u16) -> u16) -> u32 {
    let low = operation(left as u16, right as u16) as u32;
    let high = operation((left >> 16) as u16, (right >> 16) as u16) as u32;
    low | high << 16
}

fn packed_8(left: u32, right: u32, operation: fn(u8, u8) -> u8) -> u32 {
    let mut result = 0u32;
    for lane in 0..4 {
        let shift = lane * 8;
        let value = operation((left >> shift) as u8, (right >> shift) as u8);
        result |= (value as u32) << shift;
    }
    result
}

fn parse_register(name: &str) -> Result<(Bank, usize), String> {
    let bytes = name.as_bytes();
    if bytes.len() < 2 {
        return Err(format!("invalid register {name}"));
    }
    let bank = match bytes[0] {
        b'A' | b'a' => Bank::A,
        b'B' | b'b' => Bank::B,
        _ => return Err(format!("invalid register {name}")),
    };
    let index = name[1..]
        .parse::<usize>()
        .map_err(|_| format!("invalid register {name}"))?;
    if index >= 32 {
        return Err(format!("invalid register {name}"));
    }
    Ok((bank, index))
}

/// The extra idle cycles a 32-bit instruction adds to the program-memory side, which the
/// decoder produces whether or not the instruction's predicate holds.
fn multicycle_nop_count(word: u32, opcode: Opcode) -> Option<u32> {
    match opcode {
        Opcode::BnopImmediate | Opcode::BnopRegister => Some((word >> 13) & 7),
        _ => None,
    }
}

/// Whether `word`'s predicate holds against the live register files.
impl Cpu {
    fn predicate_holds(&self, word: u32, opcode: Opcode) -> bool {
        if opcode == Opcode::Callp {
            return true;
        }
        // On the SPLOOP family the would-be predicate field selects the nested-loop register
        // (binutils nfu_uspl: "use op field as a predicate address register selector"); it does
        // not gate execution - a false-predicated SPLOOPD still initiates its buffer, which is
        // how compiled nested loops keep buffering on their final outer iteration.
        if matches!(opcode, Opcode::Sploop | Opcode::Sploopd | Opcode::Sploopw) {
            return true;
        }
        let creg = (word >> 29) & 7;
        let zero = word & (1 << 28) != 0;
        let value = match creg {
            // creg 000 with z set is the C64x+ non-conditional format marker (the long-immediate
            // D-unit adds, the M-unit and non-conditional S-unit groups), not a predicate.
            0 => return true,
            1 => self.b[0],
            2 => self.b[1],
            3 => self.b[2],
            4 => self.a[1],
            5 => self.a[2],
            6 => self.a[0],
            _ => return false,
        };
        (value == 0) == zero
    }
}

fn read_snapshot(bank: Bank, index: usize, a: &[u32; 32], b: &[u32; 32]) -> u32 {
    match bank {
        Bank::A => a[index],
        Bank::B => b[index],
    }
}

#[cfg(test)]
mod tests {
    use super::{is_packet_header, packet_protected, packet_slot, Cpu, Endian, PacketSlot};

    /// The packet shape both the fetch path and the debugger's listing ask about.
    ///
    /// A header marks each of the seven data words as holding one 32-bit instruction or two
    /// compact ones, and the compact slots take their p-bits from the header's low sixteen bits
    /// (SPRUFE8B section 3.7). Before this was one decision it was two copies of these
    /// constants, one of which only the debugger exercised.
    #[test]
    fn a_header_is_recognised_by_its_top_nibble() {
        assert!(is_packet_header(0xe000_0000));
        assert!(is_packet_header(0xefff_ffff));
        // Only 0xE: a word whose top nibble is anything else is an instruction.
        for nibble in 0..16u32 {
            assert_eq!(is_packet_header(nibble << 28), nibble == 0xe);
        }
    }

    #[test]
    fn the_last_word_of_a_packet_is_the_header_not_an_instruction() {
        // Offset 28 of every packet, whatever the header says about the other seven words.
        for base in [0u32, 0x20, 0x0080_0000] {
            assert_eq!(
                packet_slot(base + 28, 0xe0ff_ffff, Endian::Little),
                PacketSlot::Header
            );
        }
    }

    #[test]
    fn the_header_decides_which_words_hold_two_compact_instructions() {
        // Bits 21-27, one per data word. With only word 3's bit set, word 3 is two compact
        // slots and the other six are 32-bit instructions.
        let header = 0xe000_0000 | (1 << (21 + 3));
        for word_index in 0..7u32 {
            let address = word_index * 4;
            let slot = packet_slot(address, header, Endian::Little);
            if word_index == 3 {
                assert!(
                    matches!(slot, PacketSlot::Compact { .. }),
                    "word {word_index}"
                );
            } else {
                assert_eq!(slot, PacketSlot::Wide, "word {word_index}");
            }
        }
    }

    #[test]
    fn compact_slots_are_numbered_across_the_packet_low_half_first() {
        // Every word compact: the sixteen slots run 0-15 in address order, and each takes its
        // p-bit from the header bit of the same number.
        let header = 0xe000_0000 | 0x0fe0_0000;
        for slot_index in 0..14u32 {
            let address = slot_index * 2;
            assert_eq!(
                packet_slot(address, header, Endian::Little),
                PacketSlot::Compact {
                    halfword_address: address,
                    parallel_bit: slot_index,
                },
                "slot {slot_index}"
            );
        }
    }

    /// A big-endian device keeps compact slot order defined by the numeric low and high halves
    /// of the containing word, so the two halfwords of a word trade byte addresses. The slot
    /// number does not move with them: it is the one the header's p-bit belongs to, which is
    /// what makes the register window's RS bit land on the right instruction.
    #[test]
    fn big_endian_compact_slots_trade_halfword_addresses_but_keep_their_number() {
        let header = 0xe000_0000 | 0x0fe0_0000;
        for slot_index in 0..14u32 {
            let address = slot_index * 2;
            let PacketSlot::Compact {
                halfword_address,
                parallel_bit,
            } = packet_slot(address, header, Endian::Big)
            else {
                panic!("slot {slot_index} is not compact");
            };
            assert_eq!(halfword_address, address ^ 2, "slot {slot_index}");
            assert_eq!(parallel_bit, slot_index, "slot {slot_index}");
        }
    }

    #[test]
    fn the_prot_bit_is_read_out_of_the_header() {
        assert!(!packet_protected(0xe000_0000));
        assert!(packet_protected(0xe000_0000 | (1 << 20)));
        // It is bit 20, next to the compact bits but not one of them.
        assert!(!packet_protected(0xe000_0000 | 0x0fe0_0000));
    }

    /// The two walks agree, which is the whole point of there being one decision.
    ///
    /// A mixed packet - word 0 holding two compact instructions, words 1 to 6 holding 32-bit
    /// ones - is walked twice: by the debugger's listing through the host view, and by the fetch
    /// path executing it. Both must land on the same instruction addresses. The listing also
    /// reports the header word, which the fetch path steps over.
    #[test]
    fn the_listing_and_the_fetch_path_walk_a_packet_the_same_way() {
        let mut words = [0u32; 16];
        // Two compact NOPs in word 0, wide NOPs in words 1-6, and a header marking only word 0
        // as compact. Every p-bit is clear, so each instruction is its own execute packet.
        words[0] = 0x0c6e_0c6e;
        words[7] = 0xe000_0000 | (1 << 21);

        let mut cpu = Cpu::new(0);
        cpu.write_words(0, &words);

        // What the listing sees, walking from the start of the packet.
        let mut listed = Vec::new();
        let mut address = 0;
        while address < 32 {
            listed.push(address);
            let (_, next) = cpu.disassemble(address);
            address = next;
        }
        assert_eq!(listed, [0, 2, 4, 8, 12, 16, 20, 24, 28]);

        // Where the fetch path takes the program counter: the same addresses, because both
        // walks make the same compact/wide decision for every word of the packet.
        let mut reached = Vec::new();
        while cpu.pc() < 32 && cpu.fault().is_none() && cpu.cycles() < 200 {
            let pc = cpu.pc();
            if reached.last() != Some(&pc) {
                reached.push(pc);
            }
            cpu.step_cycle();
        }
        assert_eq!(reached, listed);
        assert_eq!(cpu.fault(), None);

        // The header is the one address the two treat differently, and deliberately. The
        // listing shows it; the fetch path issues nothing there, and the cycle that starts on
        // it steps over the header and issues the first instruction of the next packet in the
        // same cycle - so the program counter lands past it at 36, not on it.
        assert_eq!(cpu.pc(), 36);
        let (text, next) = cpu.disassemble(28);
        assert_eq!(text, "(fetch packet header 0xe0200000)");
        assert_eq!(next, 32);
    }

    /// Builds a header-based fetch packet whose first compact slot is `word`, with every
    /// other slot a compact `NOP 1` and all p-bits clear so each slot is its own packet.
    fn compact_packet(word: u16, header_flags: u32) -> [u32; 16] {
        let mut words = [0u32; 16];
        words[0] = 0x0c6e_0000 | u32::from(word);
        for slot in words.iter_mut().take(7).skip(1) {
            *slot = 0x0c6e_0c6e;
        }
        // Bits 21-27 mark every one of the seven data words as holding two compact slots.
        words[7] = 0xe000_0000 | 0x0fe0_0000 | header_flags;
        words[8] = 0x0000_8000;
        words
    }

    fn cycles_to_reach(cpu: &mut Cpu, stop: u32, budget: u64) -> u64 {
        while cpu.cycles() < budget && cpu.fault().is_none() && !cpu.ready_to_fetch_at(stop) {
            cpu.step_cycle();
        }
        cpu.cycles()
    }

    #[test]
    fn a_cross_path_read_of_a_fresh_result_costs_one_stall() {
        // SPRUFE8B §3.8.4: `ADD .S1 A0,A0,A1` followed by `ADD .S2X B0,A1,B1` stalls one
        // cycle because A1 is read over the cross path one cycle after it was written; the
        // same pair reading the untouched A0 does not stall.
        let mut stalling = Cpu::new(0);
        stalling.write_words(0, &[0x0080_01E0, 0x0084_11E2]);
        assert_eq!(cycles_to_reach(&mut stalling, 8, 40), 3);

        let mut clear = Cpu::new(0);
        clear.write_words(0, &[0x0080_01E0, 0x0080_11E2]);
        assert_eq!(cycles_to_reach(&mut clear, 8, 40), 2);
    }

    #[test]
    fn a_load_destination_read_over_the_cross_path_does_not_stall() {
        // The same section: a load destination is exempt, but the pointer the load updates is
        // not, so `LDW .D1T1 *++A0[1],A1` followed by a cross-path read of A0 stalls.
        let mut exempt = Cpu::new(0);
        exempt.write_words(0, &[0x0080_3264, 0x0000_6000, 0x0084_11E2]);
        exempt.set_register("A0", 0x00F0_0000 - 4).unwrap();
        assert_eq!(cycles_to_reach(&mut exempt, 12, 40), 6);

        let mut pointer = Cpu::new(0);
        pointer.write_words(0, &[0x0080_3264, 0x0080_11E2]);
        pointer.set_register("A0", 0x00F0_0000 - 4).unwrap();
        assert_eq!(cycles_to_reach(&mut pointer, 8, 40), 3);
    }

    #[test]
    fn parallel_accesses_to_the_same_l1d_bank_stall() {
        // SPRU862B §3.2.1: L1D has eight 32-bit banks, and two parallel accesses that both
        // hit in the same bank cost one cycle. Thirty-two bytes apart is the same bank.
        let words = [0x020C_0265, 0x020C_02E6];
        let mut conflicting = Cpu::new(0);
        conflicting.write_words(0, &words);
        conflicting.set_register("A3", 0x00F0_0000).unwrap();
        conflicting.set_register("B3", 0x00F0_0020).unwrap();
        assert_eq!(cycles_to_reach(&mut conflicting, 8, 40), 2);

        let mut spread = Cpu::new(0);
        spread.write_words(0, &words);
        spread.set_register("A3", 0x00F0_0000).unwrap();
        spread.set_register("B3", 0x00F0_0004).unwrap();
        assert_eq!(cycles_to_reach(&mut spread, 8, 40), 1);
    }

    #[test]
    fn gmpy_reduces_by_the_side_polynomial() {
        // SPRUFE8B GMPY example: GPLYA = 87654321h, A0 = 1234 5678h, A1 = 0000 0126h gives
        // A2 = C721 A0EFh; the B side takes its polynomial from GPLYB instead.
        let mut cpu = Cpu::new(0);
        cpu.write_words(0, &[0x1104_07f0, 0x1104_07f2]);
        cpu.control.write(0x16, 0x8765_4321);
        cpu.control.write(0x17, 0x8765_4321);
        cpu.set_register("A0", 0x1234_5678).unwrap();
        cpu.set_register("A1", 0x0000_0126).unwrap();
        cpu.set_register("B0", 0x1234_5678).unwrap();
        cpu.set_register("B1", 0x0000_0126).unwrap();
        cycles_to_reach(&mut cpu, 0x40, 80);
        assert_eq!(cpu.register('A', 2), 0xc721_a0ef);
        assert_eq!(cpu.register('B', 2), 0xc721_a0ef);
    }

    /// Every M-unit op field whose operand roles claim a register-pair destination writes two
    /// registers, and every one that does not writes one. The rendering
    /// `tools/dis6x_diff.py` compares prints `a7:a6` from those roles, and the compiler emits
    /// only a handful of the pair forms, so the differential does not cover the space: this
    /// sweeps it against what execution actually schedules.
    #[test]
    fn the_m_unit_pair_destinations_are_the_ones_the_roles_claim() {
        use crate::decoder::{decode, mpy_roles, Opcode};
        const SENTINEL: u32 = 0xdead_beef;
        // Each entry is a family, the bit its op field starts at, and the fixed bits of its
        // format. src1 is A4 (A5:A4 where a pair is read), src2 A8 and the destination A6.
        let families = [
            (Opcode::Mpy, 7, 0x0000_0000u32),
            (Opcode::MpyCompound, 6, 0x0000_0030),
            (Opcode::MpyExtended, 6, 0x1000_0030),
            (Opcode::MpyUnary, 13, 0x0000_00f0),
        ];
        let mut checked = 0;
        for (family, shift, base) in families {
            for op in 0..32u32 {
                let mut word = base | (6 << 23) | (8 << 18) | (op << shift);
                if !matches!(family, Opcode::MpyUnary) {
                    word |= 4 << 13;
                }
                if decode(word) != family {
                    continue;
                }
                let mut cpu = Cpu::new(0);
                cpu.write_words(0, &[word]);
                cpu.set_register("A4", 0x1234_5678).unwrap();
                cpu.set_register("A5", 0x0abc_def0).unwrap();
                cpu.set_register("A8", 0x0000_1234).unwrap();
                cpu.set_register("A6", SENTINEL).unwrap();
                cpu.set_register("A7", SENTINEL).unwrap();
                for _ in 0..12 {
                    cpu.step_cycle();
                }
                if cpu.fault().is_some() {
                    // An op field this device has no instruction for.
                    continue;
                }
                let wrote_pair = cpu.register('A', 7) != SENTINEL;
                assert_eq!(
                    wrote_pair,
                    mpy_roles(family, word).dst_pair,
                    "{family:?} op {op:#04x} ({word:#010x}) writes a pair: {wrote_pair}"
                );
                checked += 1;
            }
        }
        // The op fields this device implements across the four formats: 26 of `m_1_or_2_src`
        // (six are the C67x slots that decode to nothing), 24 of `m_compound`, 10 of the
        // complex-multiply space and 7 unary. A synthesis that stopped decoding would
        // otherwise assert nothing at all.
        assert_eq!(checked, 67);
    }

    /// `SUB .L1 5,A1:A0,A3:A2` - the 40-bit subtraction from a five-bit constant, whose
    /// encoding `cl6x -mv6400+` produces as 0100a498h - subtracts the long from the constant
    /// and writes the pair. The constant-zero form of the same word is what TI's tools print
    /// as NEG, and only that one used to execute.
    #[test]
    fn a_long_subtraction_from_a_constant_writes_the_pair() {
        let mut cpu = Cpu::new(0);
        cpu.write_words(0, &[0x0100_a498]);
        cpu.set_register("A0", 0x0000_0007).unwrap();
        cpu.set_register("A1", 0x0000_0000).unwrap();
        cycles_to_reach(&mut cpu, 0x40, 80);
        assert!(cpu.fault().is_none(), "{:?}", cpu.fault());
        assert_eq!(cpu.register('A', 2), 0xffff_fffe);
        assert_eq!(cpu.register('A', 3), 0xff);
    }

    #[test]
    fn circular_addressing_wraps_inside_the_block() {
        // SPRUFE8B Examples 3-4 and 3-5, both with AMR = 0004 0001h: A4 circular on BK0 with
        // BK0 = 4, so a 32-byte block. `LDW .D1 *++A4[9],A1` from 0x800100 leaves A4 at
        // 0x800104 because nine words is 0x24 bytes, four past the block end, and
        // `ADDAH .D1 A4,A1,A4` with A1 = 0x13 halfwords lands on 0x800106.
        let mut cpu = Cpu::new(0);
        cpu.write_words(0, &[0x0091_3264]);
        cpu.bus_mut().write32(0x0080_0104, 0x1234_5678);
        cpu.control.write(0x00, 0x0004_0001);
        cpu.set_register("A4", 0x0080_0100).unwrap();
        cycles_to_reach(&mut cpu, 0x40, 80);
        assert_eq!(cpu.register('A', 4), 0x0080_0104);
        assert_eq!(cpu.register('A', 1), 0x1234_5678);

        let mut adda = Cpu::new(0);
        adda.write_words(0, &[0x0210_3a40]);
        adda.control.write(0x00, 0x0004_0001);
        adda.set_register("A4", 0x0080_0100).unwrap();
        adda.set_register("A1", 0x13).unwrap();
        cycles_to_reach(&mut adda, 16, 40);
        assert_eq!(adda.register('A', 4), 0x0080_0106);

        // The same ADDAH in linear mode simply advances 0x26 bytes.
        let mut linear = Cpu::new(0);
        linear.write_words(0, &[0x0210_3a40]);
        linear.set_register("A4", 0x0080_0100).unwrap();
        linear.set_register("A1", 0x13).unwrap();
        cycles_to_reach(&mut linear, 16, 40);
        assert_eq!(linear.register('A', 4), 0x0080_0126);
    }

    #[test]
    fn the_pair_writing_l_unit_family_matches_the_manual() {
        // ADDSUB example 1 (SPRUFE8B): the sum lands in the odd register and the difference in
        // the even one. DPACK2 pairs PACK2 with PACKH2 and DPACKX2 the two PACKLH2 halves.
        let program = [
            0x1104_0198u32, // ADDSUB  .L1 A0,A1,A3:A2
            0x1204_01b8,    // ADDSUB2 .L1 A0,A1,A5:A4
            0x1504_0698,    // DPACK2  .L1 A0,A1,A11:A10
            0x1604_0678,    // DPACKX2 .L1 A0,A1,A13:A12
            0x0000_8000,    // NOP 5
        ];
        let mut cpu = Cpu::new(0);
        cpu.write_words(0, &program);
        cpu.set_register("A0", 0x0700_c005).unwrap();
        cpu.set_register("A1", 0xffff_ffff).unwrap();
        cycles_to_reach(&mut cpu, program.len() as u32 * 4, 40);
        assert_eq!(cpu.register('A', 2), 0x0700_c006);
        assert_eq!(cpu.register('A', 3), 0x0700_c004);

        let mut packing = Cpu::new(0);
        packing.write_words(0, &program);
        packing.set_register("A0", 0x1234_5678).unwrap();
        packing.set_register("A1", 0x0abc_def0).unwrap();
        cycles_to_reach(&mut packing, program.len() as u32 * 4, 40);
        assert_eq!(packing.register('A', 10), 0x5678_def0);
        assert_eq!(packing.register('A', 11), 0x1234_0abc);
        assert_eq!(packing.register('A', 12), 0x5678_0abc);
        assert_eq!(packing.register('A', 13), 0xdef0_1234);
        // ADDSUB2 is a packed add into the odd register and a packed subtract into the even.
        assert_eq!(packing.register('A', 4), 0x0778_7788);
        assert_eq!(packing.register('A', 5), 0x1cf0_3568);
    }

    #[test]
    fn the_complex_multiplies_reproduce_the_manual_examples() {
        // Every worked example from the SPRUFE8B pages for the `m_1_or_2_src` complex family.
        let program = [
            0x1104_02b0u32, // CMPY     .M1 A0,A1,A3:A2
            0x1204_02f0,    // CMPYR    .M1 A0,A1,A4
            0x1284_0330,    // CMPYR1   .M1 A0,A1,A5
            0x1304_0630,    // DDOTP4   .M1 A0,A1,A7:A6
            0x1629_05f0,    // DDOTPH2  .M1 A9:A8,A10,A13:A12
            0x1729_05b0,    // DDOTPL2  .M1 A9:A8,A10,A15:A14
            0x1829_0570,    // DDOTPH2R .M1 A9:A8,A10,A16
            0x18a9_0530,    // DDOTPL2R .M1 A9:A8,A10,A17
            0x1904_03f0,    // MPY2IR   .M1 A0,A1,A19:A18
            0x1a04_0670,    // SMPY32   .M1 A0,A1,A20
            0x0000_8000,    // NOP 5
        ];
        let run = |registers: &[(&str, u32)]| {
            let mut cpu = Cpu::new(0);
            cpu.write_words(0, &program);
            for (name, value) in registers {
                cpu.set_register(name, *value).unwrap();
            }
            cycles_to_reach(&mut cpu, program.len() as u32 * 4 + 24, 120);
            assert_eq!(cpu.fault(), None);
            cpu
        };

        // CMPY examples 1 and 2, DDOTP4 example 1, DDOTPH2/DDOTPL2 example 1.
        let cpu = run(&[
            ("A0", 0x0008_0004),
            ("A1", 0x0009_0002),
            ("A8", 0x0005_0003),
            ("A9", 0x0002_0004),
            ("A10", 0x0007_0001),
        ]);
        assert_eq!((cpu.register('A', 2), cpu.register('A', 3)), (0x34, 0x40));
        assert_eq!((cpu.register('A', 12), cpu.register('A', 13)), (0x21, 0x12));
        assert_eq!((cpu.register('A', 14), cpu.register('A', 15)), (0x26, 0x21));

        let cpu = run(&[("A0", 0x7fff_7fff), ("A1", 0x7fff_8000)]);
        assert_eq!(
            (cpu.register('A', 2), cpu.register('A', 3)),
            (0xffff_8001, 0x7ffe_8001)
        );

        // CMPYR and CMPYR1 example 1, and their saturating example 2.
        let cpu = run(&[("A0", 0x0800_0400), ("A1", 0x0900_0200)]);
        assert_eq!(cpu.register('A', 4), 0x0040_0034);
        assert_eq!(cpu.register('A', 5), 0x0080_0068);
        let cpu = run(&[("A0", 0x7fff_7fff), ("A1", 0x7fff_8000)]);
        assert_eq!(cpu.register('A', 4), 0x7fff_0000);
        assert_eq!(cpu.register('A', 5), 0x7fff_ffff);

        // DDOTPH2R and DDOTPL2R examples 1 and 2.
        let cpu = run(&[
            ("A8", 0x46b4_16ba),
            ("A9", 0xbbae_d169),
            ("A10", 0x340b_f73b),
        ]);
        assert_eq!(cpu.register('A', 16), 0xf3b5_f41b);
        assert_eq!(cpu.register('A', 17), 0xf41b_0d98);
        let cpu = run(&[
            ("A8", 0x8000_5678),
            ("A9", 0x1234_8000),
            ("A10", 0x8000_8001),
        ]);
        assert_eq!(cpu.register('A', 16), 0x36e6_7fff);
        assert_eq!(cpu.register('A', 17), 0x7fff_14c4);

        // MPY2IR example 2 and SMPY32 example 1, plus DDOTP4 example 2.
        let cpu = run(&[
            ("A0", 0x8765_4321),
            ("A1", 0x1234_5678),
            ("A8", 0x8000_8000),
            ("A9", 0),
            ("A10", 0),
        ]);
        assert_eq!(
            (cpu.register('A', 18), cpu.register('A', 19)),
            (0x098c_16c1, 0xeed8_e38f)
        );
        assert_eq!(cpu.register('A', 20), 0xeed8_ed1a);
        let cpu = run(&[("A0", 0x8000_8000), ("A1", 0x8080_7f7f)]);
        assert_eq!(
            (cpu.register('A', 6), cpu.register('A', 7)),
            (0xff81_0000, 0x0080_0000)
        );
    }

    #[test]
    fn saturation_sets_the_unit_flag_one_cycle_later() {
        // SPRUFE8B Table 2-9 and Table 2-22: a saturating unit sets its own bit in SSR and the
        // SAT bit in CSR one full cycle after the result is written, MVC can clear SAT but
        // never set it, and SSR names the unit by side - here L2 for `ABS .L2`.
        let mut cpu = Cpu::new(0);
        cpu.write_words(0, &[0x05a0_035a, 0x0000_8000]); // ABS .L2 B8,B11 ; NOP 5
        cpu.set_register("B8", 0x8000_0000).unwrap();
        // One cycle after the result the flags are visible; not before.
        cpu.step_cycle();
        assert_eq!(cpu.register('B', 11), 0x7fff_ffff);
        assert_eq!(cpu.control.read(0x15, 0), 0);
        cpu.step_cycle();
        assert_eq!(cpu.control.read(0x15, 0), 1 << 1);
        assert_ne!(cpu.control.read(0x01, 0) & (1 << 9), 0);
        // Writing one to the SAT bit leaves it set; writing zero clears it.
        cpu.control.write(0x01, 1 << 9);
        assert_ne!(cpu.control.read(0x01, 0) & (1 << 9), 0);
        cpu.control.write(0x01, 0);
        assert_eq!(cpu.control.read(0x01, 0) & (1 << 9), 0);
    }

    #[test]
    fn a_compact_saturating_add_sets_the_saturation_flags() {
        // SPRUFE8B 3.10: a compact instruction is a re-encoding of the 32-bit one, so the
        // compact SADD the header's SAT bit selects has to set its unit's SSR bit and CSR.SAT
        // one cycle later, exactly as `SADD .L1` does. Compiled Zephyr code reaches the
        // saturating intrinsics through this encoding, and read CSR as clear before this.
        let mut cpu = Cpu::new(0);
        // src1 A4, src2 A5, dst A6, side A, bits 1-3 clear for the L unit.
        let mut words = compact_packet(0x82e0, 1 << 14);

        words[0] = (words[0] & 0x0000_ffff) | (0x0c6eu32 << 16);
        cpu.write_words(0, &words);
        cpu.set_register("A4", 0x7fff_ffff).unwrap();
        cpu.set_register("A5", 1).unwrap();
        cpu.step_cycle();
        assert_eq!(cpu.register('A', 6), 0x7fff_ffff);
        assert_eq!(cpu.control.read(0x15, 0), 0);
        cpu.step_cycle();
        assert_eq!(cpu.control.read(0x15, 0), 1);
        assert_ne!(cpu.control.read(0x01, 0) & (1 << 9), 0);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn interrupt_overhead_is_nine_cycles() {
        // SPRUFE8B 5.5.1: overhead for all CPU interrupts is 9 cycles - no new instruction
        // enters E1 during those cycles. With a service routine of `B IRP` plus its five delay
        // slots, taking one interrupt costs 9 + 6 cycles on top of the interrupted stream.
        let add = 0x0210_2058u32; // ADD .L1 1,A4,A4
        let service = [0x0018_00e2u32, 0x0000_8000]; // B .S2 IRP ; NOP 5
        let run = |interrupt: bool| {
            let mut cpu = Cpu::new(0);
            cpu.write_words(0, &[add; 16]);
            cpu.write_words(0x400 + 5 * 32, &service);
            cpu.control.write(0x05, 0x400); // ISTP
            cpu.control.write(0x04, (1 << 5) | 2); // IER
            cpu.control.write(0x01, 1); // CSR.GIE
            if interrupt {
                cpu.control.write(0x02, 1 << 5); // ISR: raise INT5
            }
            let cycles = cycles_to_reach(&mut cpu, 16 * 4, 200);
            assert_eq!(cpu.fault(), None);
            assert_eq!(cpu.register('A', 4), 16);
            cycles
        };
        let quiet = run(false);
        assert_eq!(quiet, 16);
        assert_eq!(run(true) - quiet, 15);
    }

    #[test]
    fn interrupt_makes_an_outstanding_load_visible_to_its_early_consumer() {
        // SPRU732J §5.7.1 p.557 / Example 5-14 p.558: ADD in LDW's delay
        // slots sees old A1 normally, but loaded A1 after an intervening interrupt.
        for interrupted in [false, true] {
            let mut cpu = Cpu::new(0);
            cpu.set_register("A0", 0x1000).unwrap();
            cpu.set_register("A2", 7).unwrap();
            cpu.write_bytes(0x1000, &10u32.to_le_bytes());
            cpu.write_words(0, &[0x0080_0264, 0x0188_2078, 0x0000_4000]);
            // LDW .D1 *A0,A1 ; ADD .L1 A1,A2,A3 ; NOP 3
            cpu.write_words(0x4a0, &[0x0018_00e2, 0x0000_8000]); // B IRP ; NOP 5
            cpu.control.write(0x05, 0x400);
            cpu.control.write(0x04, (1 << 5) | 2);
            cpu.control.write(0x01, 1);
            if interrupted {
                cpu.control.write(0x02, 1 << 5);
            }
            cpu.run(1); // interrupt, if requested, is taken after LDW and before ADD
            if interrupted {
                assert_eq!(cpu.control.read(0x06, 0), 4);
                cpu.run(15); // nine entry cycles and B IRP with five delay slots
            }
            cpu.run(1);
            assert_eq!(cpu.register('A', 3), if interrupted { 17 } else { 7 });
            assert_eq!(cpu.fault(), None);
        }
    }

    #[test]
    fn interrupt_and_exception_annul_multicycle_nop_tails() {
        // SPRU732J §5.6.2 p.557 and §6.4.2 p.572: annulled later cycles of a
        // multicycle NOP save the next packet in IRP/NRP, not the NOP itself.
        for exception in [false, true] {
            for instruction in [0x0001_0000, 0x0001_e000, 0x0188_a162] {
                let mut cpu = Cpu::new(0);
                // NOP 9, IDLE, ADDKPC.S2 +32,B3,5 (all multicycle NOP generators).
                cpu.write_words(0, &[instruction, 0]);
                let vector = 0x400 + if exception { 32 } else { 5 * 32 };
                cpu.write_words(vector, &[0x0210_2058]); // ADD .L1 1,A4,A4
                cpu.control.write(0x05, 0x400);
                cpu.control.write(0x04, (1 << 5) | 2);
                cpu.control.write(0x01, 1);
                if exception {
                    cpu.control.write(0x1a, 4); // TSR.GEE
                }
                cpu.run(2);
                if exception {
                    cpu.set_nmi_line(true);
                } else {
                    cpu.control.write(0x02, 1 << 5);
                }
                cpu.run(1);
                assert_eq!(cpu.pc(), vector);
                assert_eq!(cpu.control.read(if exception { 0x07 } else { 0x06 }, 0), 4);
                cpu.run(9);
                assert_eq!(cpu.register('A', 4), 0);
                cpu.run(1);
                assert_eq!(cpu.register('A', 4), 1, "exception={exception}");
                assert_eq!(cpu.fault(), None);
            }
        }
    }

    #[test]
    fn nmi_exception_saves_branch_blocking_in_ntsr_ib() {
        // SPRU732J §6.1.1.2 p.564 permits NMI in branch delay slots;
        // Table 6-2 p.568: IB means "Exception occurred while interrupts were blocked."
        // §5.5.2 p.549 blocks interrupts in a branch's delay slots.
        for branch in [0, 0x0000_2010] {
            let mut cpu = Cpu::new(0);
            cpu.write_words(0, &[branch, 0, 0, 0, 0, 0]);
            cpu.write_words(0x420, &[0x0210_2058]);
            cpu.control.write(0x05, 0x400);
            cpu.control.write(0x04, 2);
            cpu.control.write(0x1a, 4);
            cpu.run(1);
            cpu.set_nmi_line(true);
            cpu.run(1);
            assert_eq!(cpu.pc(), 0x420);
            assert_eq!(cpu.control.read(0x07, 0), 8);
            assert_eq!(
                cpu.control.read(0x1c, 0) & (1 << 15),
                if branch == 0 { 0 } else { 1 << 15 }
            );
            cpu.run(10);
            assert_eq!(cpu.register('A', 4), 1);
            assert_eq!(cpu.pc(), 0x424); // the pending branch was discarded
            assert_eq!(cpu.fault(), None);
        }
    }

    #[test]
    fn nmi_exception_annuls_generated_nops_without_cancelling_loads() {
        // SPRU732J §3.9.2.2 p.82: PROT adds four NOP cycles, not a memory stall.
        // §3.11 BNOP p.147 and CALLP p.153 likewise generate multicycle NOPs.
        // §6.4.2 p.572 annuls their remaining cycles on exception entry.
        for kind in 0..3 {
            let mut cpu = Cpu::new(0);
            let words = match kind {
                0 => compact_packet(0x0d3c, 1 << 20).to_vec(), // protected LDW
                1 => vec![0x0000_a120],                        // BNOP.S1 0,5
                _ => vec![0x1000_0010],                        // CALLP.S1 0,A3
            };
            cpu.write_words(0, &words);
            cpu.set_register("A6", 0x1000).unwrap();
            cpu.write_bytes(0x1000, &10u32.to_le_bytes());
            cpu.write_words(0x420, &[0x0210_2058]); // ADD .L1 1,A4,A4
            cpu.control.write(0x05, 0x400);
            cpu.control.write(0x04, 2);
            cpu.control.write(0x1a, 4);
            cpu.run(1);
            let next = cpu.pc();
            cpu.set_nmi_line(true);
            cpu.run(1);
            assert_eq!(cpu.pc(), 0x420);
            assert_eq!(cpu.control.read(0x07, 0), next);
            cpu.run(9);
            assert_eq!(cpu.register('A', 4), 0);
            cpu.run(1);
            assert_eq!(cpu.register('A', 4), 1, "kind={kind}");
            if kind == 0 {
                assert_eq!(cpu.register('A', 3), 10);
            }
            assert_eq!(cpu.fault(), None);
        }
    }

    #[test]
    fn an_interrupt_during_a_loop_drains_and_pipes_back_up() {
        // A one-cycle SPLOOP body that counts 100 iterations. An interrupt taken part-way
        // through must drain the loop, run the service routine and pipe the loop back up, so
        // the count comes out the same as an uninterrupted run (SPRUFE8B 7.13.1 to 7.13.5).
        let program = [
            0x0003_8000u32, // SPLOOP 1
            0x0210_2058,    // ADD .L1 1,A4,A4
            0x0003_4000,    // SPKERNEL 0,0
            0x0000_8000,    // NOP 5
        ];
        let service = [
            0x0018_00e2u32, // B .S2 IRP
            0x0000_8000,    // NOP 5
        ];
        let run = |interrupt_at: Option<u64>| {
            let mut cpu = Cpu::new(0);
            cpu.write_words(0, &program);
            // The interrupt service table lives at 0x400; INT5's entry is eight words in.
            cpu.write_words(0x400 + 5 * 32, &service);
            cpu.control.write(0x05, 0x400); // ISTP
            cpu.control.write(0x04, (1 << 5) | 2); // IER: INT5 and NMIE
            cpu.control.write(0x01, 1); // CSR.GIE
            cpu.control.write(0x0d, 100); // ILC
            cpu.set_register("A4", 0).unwrap();
            let mut fired = false;
            while cpu.cycles() < 400 && cpu.fault().is_none() {
                if interrupt_at == Some(cpu.cycles()) && !fired {
                    fired = true;
                    cpu.control.write(0x02, 1 << 5); // ISR: raise INT5
                }
                cpu.step_cycle();
            }
            assert_eq!(cpu.fault(), None);
            cpu.register('A', 4)
        };
        let quiet = run(None);
        assert_eq!(quiet, 100);
        for cycle in [12u64, 17, 23, 40, 61] {
            assert_eq!(run(Some(cycle)), 100, "interrupt at cycle {cycle}");
        }
    }

    #[test]
    fn galois_multiply_and_bit_count_match_the_manual_examples() {
        // SPRUFE8B GMPY4 examples 1 and 2 under the reset GFPGFR (field size 7h, polynomial
        // 1Dh), and BITC4's per-byte counts.
        for (src1, src2, expected) in [
            (0x4523_0001u32, 0x5734_0001u32, 0x7292_0001u32),
            (0xfffe_021f, 0xfffe_0201, 0xe2e3_041f),
        ] {
            let mut cpu = Cpu::new(0);
            cpu.write_words(0, &[0x0398_a470]);
            cpu.set_register("A5", src1).unwrap();
            cpu.set_register("A6", src2).unwrap();
            cycles_to_reach(&mut cpu, 32, 20);
            assert_eq!(cpu.register('A', 7), expected);
        }

        let mut cpu = Cpu::new(0);
        cpu.write_words(0, &[0x0417_c0f0]);
        cpu.set_register("A5", 0xff01_0703).unwrap();
        cycles_to_reach(&mut cpu, 32, 20);
        assert_eq!(cpu.register('A', 8), 0x0801_0302);
    }

    #[test]
    fn byte_merges_and_saturating_shifts_follow_the_manual() {
        // PACKH4 example 1 (A2 = 3789F23Ah, A8 = 04B84975h gives 37F20449h), SHLMB shifting
        // src2 left one byte with src1's top byte merged in, and the three saturating shift
        // forms: SSHVL left, SSHVR right, SSHVR with a negative count shifting left.
        let mut cpu = Cpu::new(0);
        cpu.write_words(
            0,
            &[
                0x0618_ad38, // PACKH4 .L1 A5,A6,A12
                0x0598_ae70, // SHLMB  .S1 A5,A6,A11
                0x0494_c730, // SSHVL  .M1 A5,A6,A9
                0x0b14_c8e0, // SSHL   .S1 A5,A6,A22
                0x1818_aef0, // RPACK2 .S1 A5,A6,A16
            ],
        );
        cpu.set_register("A5", 0x3789_f23a).unwrap();
        cpu.set_register("A6", 0x04b8_4975).unwrap();
        cycles_to_reach(&mut cpu, 20, 40);
        assert_eq!(cpu.register('A', 12), 0x37f2_0449);
        assert_eq!(cpu.register('A', 11), 0xb849_7537);
        // A6 = 04B84975h limits to a shift of 31, so the left shift saturates.
        assert_eq!(cpu.register('A', 9), 0x7fff_ffff);
        assert_eq!(cpu.register('A', 22), 0x7fff_ffff);
        // src1 doubles to 6F13E474h and src2 to 09709 2EAh, so the halfwords are 6F13h/0970h.
        assert_eq!(cpu.register('A', 16), 0x6f13_0970);

        let mut shifts = Cpu::new(0);
        shifts.write_words(
            0,
            &[
                0x0514_c6b0, // SSHVR .M1 A5,A6,A10
            ],
        );
        shifts.set_register("A5", 0x1234_5678).unwrap();
        shifts.set_register("A6", 5).unwrap();
        cycles_to_reach(&mut shifts, 16, 40);
        assert_eq!(shifts.register('A', 10), 0x0091_a2b3);

        let mut negative = Cpu::new(0);
        negative.write_words(0, &[0x0514_c6b0]);
        negative.set_register("A5", 0x1234_5678).unwrap();
        negative.set_register("A6", (-2i32) as u32).unwrap();
        cycles_to_reach(&mut negative, 16, 40);
        assert_eq!(negative.register('A', 10), 0x48d1_59e0);
    }

    /// SPRU862B §3.4.7's two worked examples, transcribed as streams with the access pattern
    /// its arithmetic assumes. Both loops read 16 bytes per load packet - two parallel LDDWs,
    /// the second eight bytes behind the first - so a 64-byte L1D line lasts four packets and
    /// its miss is neither paired with another in the same cycle nor issued in the cycle after
    /// one, the two cases Table 3-2 discounts. L1P is left in SRAM mode, so `stall_memory`
    /// holds L1D stalls and nothing else.
    const MANUAL_LOAD_PACKET: [u32; 2] = [
        0x0220_5765, // LDDW .D1T1 *A8++[2],A5:A4
        0x0324_57e6, // || LDDW .D2T2 *B9++[2],B7:B6
    ];
    const MANUAL_STORE_PACKET: [u32; 2] = [
        0x032c_5745, // STDW .D1T1 A7:A6,*A11++[2]
        0x0230_57c6, // || STDW .D2T2 B5:B4,*B12++[2]
    ];

    /// The CPU keeps the fetch packet it last read from L1P and issues the rest of it without
    /// asking again. A master overwriting an instruction in that packet makes the buffered copy as
    /// stale as the L1P line, so the coherence check must see the next instruction issue from it.
    #[test]
    fn a_stale_instruction_in_the_buffered_fetch_packet_is_reported() {
        use crate::memory::CoherenceHazard;
        use crate::system::{MasterAccess, Privilege};
        const CODE: u32 = 0x0080_1000;
        let mut cpu = Cpu::new(0);
        cpu.bus_mut().write32(0x0184_0020, 1);
        cpu.write_words(CODE, &[0; 8]);
        cpu.fabric_mut().set_coherence_check(true);
        cpu.set_pc(CODE);
        cpu.step_cycle();
        assert_eq!(cpu.pc(), CODE + 4, "one serial NOP issued from the packet");
        cpu.fabric_mut()
            .write_word(Privilege::ID0_SUPERVISOR, CODE + 4, 0);
        assert_eq!(cpu.fabric_mut().take_coherence_hazards(), []);
        // The fill's stall cycles pass first; then the overwritten NOP issues.
        cycles_to_reach(&mut cpu, CODE + 8, 50);
        assert_eq!(
            cpu.fabric_mut().take_coherence_hazards(),
            [(0, CoherenceHazard::StaleFetch { address: CODE })]
        );
    }

    /// The check observes; it never changes what executes. A packet fetched past a frozen L1P is
    /// not resident, so a master's write to some other, resident L1P line must not make the CPU
    /// fetch it again: the instruction it issues next is the one it buffered, check or no check.
    #[test]
    fn the_coherence_check_leaves_a_packet_fetched_past_a_frozen_l1p_alone() {
        use crate::system::{MasterAccess, Privilege};
        const RESIDENT: u32 = 0x0080_1000;
        const BYPASSED: u32 = 0x0080_1100;
        const MVK_5_A3: u32 = 0x0180_02a8;
        let run = |check: bool| {
            let mut cpu = Cpu::new(0);
            cpu.bus_mut().write32(0x0184_0020, 1);
            cpu.write_words(RESIDENT, &[0; 8]);
            cpu.write_words(BYPASSED, &[0; 8]);
            cpu.fabric_mut().set_coherence_check(check);
            cpu.set_pc(RESIDENT);
            cycles_to_reach(&mut cpu, RESIDENT + 4, 50);
            cpu.bus_mut().write32(0x0184_0024, 1);
            cpu.set_pc(BYPASSED);
            cycles_to_reach(&mut cpu, BYPASSED + 4, 50);
            let master = Privilege::ID0_SUPERVISOR;
            cpu.fabric_mut().write_word(master, BYPASSED + 4, MVK_5_A3);
            cpu.fabric_mut().write_word(master, RESIDENT + 8, 0);
            cycles_to_reach(&mut cpu, BYPASSED + 8, 50);
            cpu.register('A', 3)
        };
        assert_eq!(run(false), 0, "the buffered NOP issues");
        assert_eq!(run(true), run(false));
    }

    #[test]
    fn the_manuals_dotprod_read_miss_total() {
        // "The dotprod routine sees 128 read misses since it accesses 4096 elements. We expect
        // to see 128 misses x 12.5 cycles = 1600 stall cycles." (SPRU862B §3.4.7.) 4096 shorts
        // are 8192 bytes, which is 128 of this device's 64-byte lines, and 128 lines occupy
        // 128 of the 256 sets of a 32K two-way L1D, so nothing evicts anything: the miss count
        // is the line count. The total is the manual's own arithmetic, and it is an integer
        // because 128 misses of an average 12.5 cycles cost the same as 64 twelves and 64
        // thirteens - which is what makes it an expectation rather than a range.
        let mut cpu = Cpu::new(0);
        cpu.bus_mut().write32(0x0184_0040, 4);
        let words: Vec<u32> = std::iter::repeat_n(MANUAL_LOAD_PACKET, 512)
            .flatten()
            .collect();
        cpu.write_words(0, &words);
        cpu.set_register("A8", 0x0080_0000).unwrap();
        cpu.set_register("B9", 0x0080_0008).unwrap();
        let cycles = cycles_to_reach(&mut cpu, 4 * words.len() as u32, 8_000);
        assert_eq!(
            cpu.fabric_ref().megamodule(0).cache_views()[1]
                .stats
                .line_fills,
            128
        );
        assert_eq!(cpu.stats().stall_memory, 1600);
        // 512 load packets issue one cycle each, and every stall cycle is an L1D stall.
        assert_eq!(cycles, 512 + 1600);
    }

    #[test]
    fn the_manuals_vecaddc_read_miss_and_write_buffer_total() {
        // §3.4.7 states vecaddc's first call twice over, once for the read misses alone and
        // once with the write buffer in the picture. "The routine accesses 2048 elements, 4096
        // bytes spanning 64 cache lines. Assuming an L2 type with 1 wait state and 4 x 128-bit
        // banks, we expect to see 64 misses x 12.5 stalls = 800 stall cycles." Then Table 3-4
        // walks the interaction: a read miss on the input array every eight execute cycles,
        // one write buffer entry every two cycles for the output array, and from the second
        // miss onwards "the write buffer still contains A that needs to be drained taking one
        // cycle". "In summary, we expect to see the following number of L1D stall cycles:
        // 12.5 + ((12.5 + 1) x 63) = 863."
        //
        // Both numbers are checked, and their difference is the claim Table 3-4 actually
        // makes: 63 cycles, one per miss after the first, with no fourth-entry stall anywhere
        // because the buffer fills at exactly the rate it drains.
        //
        // The transcription is the loop the manual describes rather than the register
        // allocation of Example 3-10: one load packet reading 16 bytes of in[] alternating
        // with one store packet writing 16 bytes of out[], 256 times, which is the manual's
        // "eight results every 2 cycles" over 2048 elements - 512 execute cycles, 4096 bytes,
        // 64 lines. L1D does not write allocate, so every store pair misses and merges into
        // one 128-bit write buffer entry.
        let run = |words: &[u32]| {
            let mut cpu = Cpu::new(0);
            cpu.bus_mut().write32(0x0184_0040, 4);
            cpu.write_words(0, words);
            for (name, address) in [
                ("A8", 0x0080_0000u32),
                ("B9", 0x0080_0008),
                ("A11", 0x0080_2000),
                ("B12", 0x0080_2008),
            ] {
                cpu.set_register(name, address).unwrap();
            }
            let cycles = cycles_to_reach(&mut cpu, 4 * words.len() as u32, 8_000);
            // in[] is read and allocated; out[] is written and never allocated.
            assert_eq!(
                cpu.fabric_ref().megamodule(0).cache_views()[1]
                    .stats
                    .line_fills,
                64
            );
            (cycles, cpu.stats().stall_memory)
        };

        let loads: Vec<u32> = std::iter::repeat_n(MANUAL_LOAD_PACKET, 256)
            .flatten()
            .collect();
        let (load_cycles, read_miss_stalls) = run(&loads);
        assert_eq!(read_miss_stalls, 800);
        assert_eq!(load_cycles, 256 + 800);

        let kernel: Vec<u32> = std::iter::repeat_n(
            [
                MANUAL_LOAD_PACKET[0],
                MANUAL_LOAD_PACKET[1],
                MANUAL_STORE_PACKET[0],
                MANUAL_STORE_PACKET[1],
            ],
            256,
        )
        .flatten()
        .collect();
        let (kernel_cycles, kernel_stalls) = run(&kernel);
        assert_eq!(kernel_stalls, 863);
        assert_eq!(kernel_stalls - read_miss_stalls, 63);
        assert_eq!(kernel_cycles, 512 + 863);
    }

    #[test]
    fn dot_products_follow_the_manual_examples() {
        // SPRUFE8B DOTPN2 example 1 and 2: the product of the lower halfwords is subtracted
        // from the product of the upper halfwords, which is the opposite order to the sum
        // DOTP2 computes. The DSPLIB IIR kernel is compiled into DOTPN2 pairs at -O2.
        let mut cpu = Cpu::new(0);
        cpu.write_words(0, &[0x0418_a270, 0x0498_a330]);
        cpu.set_register("A5", 0x3629_274a).unwrap();
        cpu.set_register("A6", 0x325c_8036).unwrap();
        cpu.run(8);
        assert_eq!(cpu.register('A', 8) as i32, 507_784_992);
        assert_eq!(cpu.register('A', 9) as i32, -150_289_832);

        let mut second = Cpu::new(0);
        second.write_words(0, &[0x0418_a270]);
        second.set_register("A5", 0x3ff6_5010).unwrap();
        second.set_register("A6", 0xb1c3_0244).unwrap();
        second.run(8);
        assert_eq!(second.register('A', 8) as i32, -339_842_526);
    }

    #[test]
    fn the_compact_single_bit_extu_writes_the_predicate_register() {
        // `EXTU.S2 B6,2,31,B0` in compact form: the destination is fully determined by the
        // functional unit, so it is B0 and the source must survive. The Ethernet firmware
        // polls a CPPI descriptor's ownership bit with exactly this instruction.
        for (value, expected) in [(0xd000_003cu32, 0u32), (0xe000_003c, 1)] {
            let mut cpu = Cpu::new(0);
            let mut words = compact_packet(0x4303, 0);
            words[0] = (words[0] & 0x0000_ffff) | (0x0c6eu32 << 16);
            cpu.write_words(0, &words);
            cpu.set_register("B6", value).unwrap();
            cpu.set_register("B0", 0xdead).unwrap();
            cpu.run(4);
            assert_eq!(cpu.register('B', 0), expected, "value {value:#x}");
            assert_eq!(cpu.register('B', 6), value);
            assert_eq!(cpu.fault(), None);
        }
    }

    #[test]
    fn long_offset_loads_and_stores_use_only_the_second_d_unit() {
        // A packet the compiler emits at -O3: `MV.D1 A16,A4` in parallel with
        // `STW.D2T1 A6,*+B14[0]`. The B14-relative long-offset forms are .D2 only, so their
        // `s` bit selects the data path rather than the unit and this packet is legal.
        let mut cpu = Cpu::new(0);
        cpu.write_words(
            0,
            &[
                0x0218_0fdb, // MV.L2 B6,B4
                0x0344_0fd9, // || MV.L1 A17,A6
                0x0240_08f1, // || MV.D1 A16,A4
                0x0300_007c, // || STW.D2T1 A6,*+B14[0]
            ],
        );
        cpu.set_register("A6", 0x1234).unwrap();
        cpu.set_register("A16", 7).unwrap();
        cpu.set_register("B14", 0x2000).unwrap();
        cpu.run(3);
        assert_eq!(cpu.fault(), None);
        assert_eq!(cpu.register('A', 4), 7);
        assert_eq!(
            u32::from_le_bytes(cpu.host_read_bytes(0x2000, 4).try_into().unwrap()),
            0x1234
        );
    }

    #[test]
    fn the_compact_three_operand_compare_writes_a_low_predicate_register() {
        // `CMPLT.L1 A17,A16,A0` in compact form. Its one-bit destination field is the
        // `reg_nors` form, so the header RS bit must not move the result into the A16-A23
        // window: writing A16 instead of A0 turned a compiled loop bound into zero.
        let mut cpu = Cpu::new(0);
        let mut words = compact_packet(0x2c08, 1 << 19);
        words[0] = (words[0] & 0x0000_ffff) | (0x0c6eu32 << 16);
        cpu.write_words(0, &words);
        cpu.set_register("A17", 5).unwrap();
        cpu.set_register("A16", 64).unwrap();
        cpu.set_register("A0", 0).unwrap();
        cpu.run(4);
        assert_eq!(cpu.register('A', 0), 1);
        assert_eq!(cpu.register('A', 16), 64);
        assert_eq!(cpu.register('A', 17), 5);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn compact_m_unit_multiplies_cover_all_four_halfword_pairings() {
        // `m_m3`: op selects lsb x lsb, msb x msb, lsb x msb and msb x lsb; the destination
        // is the two-bit field doubled, and the header SAT bit turns them into the SMPY forms
        // that double the product and saturate.
        for (op, expected) in [(0u16, 24i32), (1, -15), (2, -20), (3, 18)] {
            let mut cpu = Cpu::new(0);
            let mut words = compact_packet(0x651e | (op << 5), 0);
            words[0] = (words[0] & 0x0000_ffff) | (0x0c6eu32 << 16);
            cpu.write_words(0, &words);
            cpu.set_register("A3", 0x0003_0004).unwrap();
            cpu.set_register("A2", 0xfffb_0006).unwrap();
            cpu.run(6);
            assert_eq!(cpu.register('A', 2) as i32, expected, "op {op}");
            assert_eq!(cpu.fault(), None);
        }

        let mut saturating = Cpu::new(0);
        let mut words = compact_packet(0x651e, 1 << 14);
        words[0] = (words[0] & 0x0000_ffff) | (0x0c6eu32 << 16);
        saturating.write_words(0, &words);
        saturating.set_register("A3", 0x8000_8000).unwrap();
        saturating.set_register("A2", 0x8000_8000).unwrap();
        saturating.run(6);
        assert_eq!(saturating.register('A', 2), 0x7fff_ffff);
    }

    #[test]
    fn comparisons_against_a_forty_bit_long_and_norm_of_an_int() {
        // The L unit compare family has forms whose src2 is a register pair holding a 40-bit
        // long, and NORM has both a long and a 32-bit form. The RTS soft-float and division
        // helpers use them, e.g. `CMPEQ.L1 0,A9:A8,A16` to test a double for zero.
        let mut cpu = Cpu::new(0);
        cpu.write_words(
            0,
            &[
                0x0820_0a18, // CMPEQ.L1 0,A9:A8,A16
                0x0110_0c78, // NORM.L1 A4,A2
                0x08a0_6898, // CMPGT.L1 3,A9:A8,A17
                0x0920_6a98, // CMPLT.L1 3,A9:A8,A18
                0x09a0_6998, // CMPGTU.L1 3,A9:A8,A19
                0x0a20_6b98, // CMPLTU.L1 3,A9:A8,A20
            ],
        );
        // A9:A8 is -1 as a 40-bit long, and A4 has nine redundant sign bits.
        cpu.set_register("A8", 0xffff_ffff).unwrap();
        cpu.set_register("A9", 0xff).unwrap();
        cpu.set_register("A4", 0xffc0_0000).unwrap();
        cpu.run(8);
        assert_eq!(cpu.register('A', 16), 0);
        assert_eq!(cpu.register('A', 2), 9);
        assert_eq!(cpu.register('A', 17), 1);
        assert_eq!(cpu.register('A', 18), 0);
        // Unsigned forms read the same pair as a 40-bit unsigned value, so it is huge.
        assert_eq!(cpu.register('A', 19), 0);
        assert_eq!(cpu.register('A', 20), 1);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn the_unsigned_compares_take_an_unsigned_five_bit_constant() {
        // CMPGTU and CMPLTU are the only constant forms on the L unit whose src1 is unsigned,
        // which is how the compiler realises `index <u 16` as `CMPGTU 16,index,dst`. Reading
        // that constant as signed makes every such loop bound compare against 0xfffffff0 and
        // never terminate; `cl6x -O0` emits exactly this word for a counted loop.
        let mut cpu = Cpu::new(0);
        cpu.write_words(
            0,
            &[
                0x000e09d8, // CMPGTU.L1 16,A3,A0
                0x008e8bd8, // CMPLTU.L1 20,A3,A1
            ],
        );
        cpu.set_register("A3", 16).unwrap();
        cpu.run(4);
        assert_eq!(cpu.register('A', 0), 0);
        assert_eq!(cpu.register('A', 1), 0);
        let mut below = Cpu::new(0);
        below.write_words(0, &[0x000e09d8, 0x008e8bd8]);
        below.set_register("A3", 15).unwrap();
        below.run(4);
        assert_eq!(below.register('A', 0), 1);
        assert_eq!(below.register('A', 1), 0);
        let mut above = Cpu::new(0);
        above.write_words(0, &[0x000e09d8, 0x008e8bd8]);
        above.set_register("A3", 25).unwrap();
        above.run(4);
        assert_eq!(above.register('A', 0), 0);
        assert_eq!(above.register('A', 1), 1);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn norm_of_a_zero_long_saturates_at_the_datapath_width() {
        // NORM counts redundant sign bits, so a zero operand - which has no bit to stop
        // the count - returns the datapath width minus one, 39 for the long form exactly
        // as the 32-bit form returns 31 (SPRU732J examples: NORM of 7 as a long is 36).
        let mut cpu = Cpu::new(0);
        cpu.write_words(
            0,
            &[
                0x0210_0c1b, // NORM.L2 B5:B4,B4
                0x0000_0000,
            ],
        );
        cpu.run(8);
        assert_eq!(cpu.register('B', 4), 39);
        assert_eq!(cpu.fault(), None);
    }

    /// The 32-bit NEG wraps: it is SUB 0,src2 (.L opfield 0000110), and cl6x negates an
    /// unsigned int with it, where C requires -0x80000000u to be 0x80000000u. Zephyr's
    /// cbprintf printed INT64_MIN as -9223372032559808512 while this saturated, because
    /// the high word of a 64-bit negation is exactly this instruction.
    #[test]
    fn neg_of_the_most_negative_int_wraps() {
        let mut cpu = Cpu::new(0);
        cpu.set_register("A4", 0x8000_0000).unwrap();
        cpu.write_words(
            0,
            &[
                0x0210_00d8, // NEG.L1 A4,A4 (cl6x, dis6x)
                0x0000_0000,
            ],
        );
        cpu.run(8);
        assert_eq!(cpu.register('A', 4), 0x8000_0000);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn neg_of_the_most_negative_long_saturates() {
        // NEG is defined by the saturating execution `0 -s src2` even though it assembles
        // as plain SUB (SPRU732J), so negating -2^39 clamps to 2^39-1. Compiled code relies
        // on this when it expands a saturating long subtraction into NEG plus SADD.
        let mut cpu = Cpu::new(0);
        cpu.set_register("B4", 0).unwrap();
        cpu.set_register("B5", 0x80).unwrap();
        cpu.write_words(
            0,
            &[
                0x0210_049a, // NEG.L2 B5:B4,B5:B4
                0x0000_0000,
            ],
        );
        cpu.run(8);
        assert_eq!(cpu.register('B', 4), 0xffff_ffff);
        assert_eq!(cpu.register('B', 5), 0x7f);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn the_eight_bit_compact_bnop_fills_five_delay_slots() {
        // Two fetch packets lifted from a compiled RTS image. The first ends in the compact
        // eight-bit `[!A0] BNOP` form, whose NOP count is hardwired to five, so every packet
        // in its delay slots is replaced by a NOP: the `MVK.L2 1,B10` that follows must not
        // run, and B10 keeps the value the predicated move gave it.
        let mut cpu = Cpu::new(0);
        cpu.write_words(
            0,
            &[
                0x027f_e06b, // MVKH.S2 0xffc00000,B4
                0x850e_a586, // MV.L1 A11,A5 || MV.S1 A10,A4
                0x0010_0fd9, // MV.L1 A4,A0
                0x06bc_06a1, // || MV.S1 A15,A13
                0x05bc_1fda, // || MV.L2X A15,B11
                0xd53c_1fdb, // || [!A0] MV.L2X A15,B10
                0xa586_d23a, // || [!A0] BNOP.S1 +144,5 ; MV.L1 A11,A5
                0xe840_a004, // fetch packet header
                0x0504_a35b, // MVK.L2 1,B10        (first delay slot packet)
                0x0600_192b, // || MVK.S2 0x0032,B12
                0x0ffe_8c10, // || B.S1 (far call)
                0x06a7_8506, // MV.L1 A10,A4 ; MVK.L2 0,B5
                0x0234_1fda, // MV.L2X A13,B4
                0x02a0_126a, // MVKH.S2 0x40240000,B5
                0x0188_0162, // ADDKPC.S2 +32,B3,0
                0xe100_0000, // fetch packet header
            ],
        );
        cpu.set_register("A15", 0x77).unwrap();
        cpu.set_register("A4", 0).unwrap();
        cpu.run(10);
        assert_eq!(cpu.register('B', 10), 0x77);
        assert_eq!(cpu.register('B', 12), 0);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn spmaskr_reloads_a_nested_loop_while_its_selector_says_so() {
        // A predicated `SPLOOP` is a nested loop: when it finishes draining the buffer waits
        // instead of going idle, the post-SPKERNEL code runs, and `SPMASKR` reloads ILC from
        // RILC and replays the body (SPRU732J §7.6). The outer loop branch only redirects the
        // frozen fetch stream, so it must not idle the buffer. The register the `SPLOOP` named
        // decides whether each reload point actually reloads, and the compiler puts the outer
        // loop's own branch in the same execute packet on the same register - so the inner
        // loop runs once per outer pass and not once more.
        let mut cpu = Cpu::new(0);
        cpu.write_words(
            0,
            &[
                0x0208_a35a, // MVK.L2 2,B4
                0x0200_a358, // MVK.L1 0,A4
                0x0088_a35a, // MVK.L2 2,B1     (two outer iterations)
                0x0690_03a2, // MVC.S2 B4,ILC
                0x0710_03a2, // MVC.S2 B4,RILC
                0x0000_6000, // NOP 4           (ILC has a four-cycle load latency)
                0x4283_8000, // [B1] SPLOOP 6
                0x0210_2058, // ADD.L1 1,A4,A4  (the whole loop body)
                0x0000_0000, // NOP
                0x0000_0000, // NOP
                0x0000_0000, // NOP
                0x0000_0000, // NOP
                0x0000_0000, // NOP
                0x0003_4000, // SPKERNEL 0,0
                0x0087_e05a, // outer: SUB.L2 B1,1,B1
                0x0003_2001, // SPMASKR
                0x4fff_ff10, // || [B1] B.S1 outer
                0x0000_8000, // NOP 5
            ],
        );
        cpu.run(400);
        // Two outer passes of two inner iterations each.
        assert_eq!(cpu.register('A', 4), 4);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn set_and_clr_name_an_inclusive_bit_range_and_dint_uses_no_unit() {
        // `SET B8,0,19,B8` fills bits 0-19 and `CLR B8,4,7,B8` clears bits 4-7; `DINT` shares
        // the packet with `MV.D1 A1,A0` and must neither claim a unit nor write a register.
        let mut cpu = Cpu::new(0);
        cpu.write_words(
            0,
            &[
                0x009c_a358,
                0x0400_a35a,
                0x0000_2000,
                0x0004_08f1,
                0x1000_4001,
                0x0420_138a,
                0x0000_8000,
                0x0420_87ca,
                0x1000_6000,
                0x0000_8000,
            ],
        );
        cpu.run(20);
        assert_eq!(cpu.register('B', 8), 0x000f_ff0f);
        assert_eq!(cpu.register('A', 0), 7);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn nonaligned_doubleword_offsets_are_scaled_by_the_sc_bit() {
        // `LDNDW *A4++[1]` and `STNDW *A5++[1]` both set the sc bit, so each advances its
        // pointer by eight bytes; the copy also reads and writes from an odd address.
        let mut cpu = Cpu::new(0);
        cpu.set_register("A4", 0x1001).unwrap();
        cpu.set_register("A5", 0x2003).unwrap();
        cpu.write_bytes(0x1001, &0x1122_3344_5566_7788u64.to_le_bytes());
        cpu.write_words(0, &[0x0390_3724, 0x0000_8000, 0x0394_3774, 0x0000_8000]);
        cpu.run(16);
        assert_eq!(cpu.register('A', 4), 0x1009);
        assert_eq!(cpu.register('A', 5), 0x200b);
        assert_eq!(
            cpu.host_read_bytes(0x2003, 8),
            0x1122_3344_5566_7788u64.to_le_bytes()
        );
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn a_doubleword_puts_its_first_memory_word_in_the_even_register_only_on_little_endian() {
        // SPRU732J LDDW: "In little-endian mode, the first 32-bit word in memory is loaded into
        // the even register. In big-endian mode, the first 32-bit word in memory is loaded into
        // the odd register", and Example 2 loads mem 18h = 4021 3333h, 3333 3333h into
        // A1:A0 = 4021 3333h:3333 3333h in big-endian mode. A push with STDW and a pop with
        // LDDW round-trip either way; what the rule decides is the memory image in between,
        // which a word load of one half of a pushed pair sees.
        const STDW_A11_A10_B15_POSTDEC: u32 = 0x053c_35c4;
        const LDDW_B15_PREINC_A11_A10: u32 = 0x053c_33e4;
        for endian in [Endian::Little, Endian::Big] {
            let mut cpu = Cpu::new_with_endian(0, endian);
            cpu.write_words(
                0,
                &[
                    STDW_A11_A10_B15_POSTDEC,
                    0x0000_8000,
                    LDDW_B15_PREINC_A11_A10,
                    0x0000_8000,
                    0,
                    0,
                    0,
                    0,
                ],
            );
            cpu.set_register("B15", 0x1010).unwrap();
            cpu.set_register("A11", 0x4021_3333).unwrap();
            cpu.set_register("A10", 0x3333_3333).unwrap();
            cpu.run(1);
            cpu.run(5);
            let stored = cpu.host_read_bytes(0x1010, 8);
            let expected: [u8; 8] = match endian {
                Endian::Little => [0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x21, 0x40],
                Endian::Big => [0x40, 0x21, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33],
            };
            assert_eq!(
                stored,
                expected,
                "STDW memory image, big endian: {}",
                endian == Endian::Big
            );
            // Overwrite the image with Example 2's doubleword in this byte order and load it.
            let image: [u8; 8] = match endian {
                Endian::Little => [0x11, 0x11, 0x11, 0x11, 0x22, 0x22, 0x22, 0x22],
                Endian::Big => [0x22, 0x22, 0x22, 0x22, 0x11, 0x11, 0x11, 0x11],
            };
            cpu.write_bytes(0x1010, &image);
            cpu.run(8);
            assert_eq!(cpu.fault(), None);
            assert_eq!(cpu.register('B', 15), 0x1010);
            assert_eq!(
                cpu.register('A', 11),
                0x2222_2222,
                "odd register, big endian: {}",
                endian == Endian::Big
            );
            assert_eq!(
                cpu.register('A', 10),
                0x1111_1111,
                "even register, big endian: {}",
                endian == Endian::Big
            );
        }
    }

    #[test]
    fn compact_nonaligned_doubleword_offsets_count_bytes_not_doublewords() {
        // The compact register-indexed `LDNDW *A4(A6)` (dinddw) adds its offset register
        // unscaled, in bytes - unlike every aligned compact form, whose offsets count
        // access-sized units. The packet is lifted from a compiled _mem8 loop at -O0.
        let mut cpu = Cpu::new(0);
        cpu.write_words(
            0,
            &[
                0x9a33_05a6,
                0xa72b_9d89,
                0x034f_1c28,
                0x0300_4069,
                0x0410_1fd8,
                0xc45c_61c0,
                0x03a8_8ca0,
                0xe474_8000,
            ],
        );
        // A3 is zeroed by the packet's own MVK, so the ADD yields a zero displacement
        // and the load reads the eight bytes at the base register's value. With the
        // offset scaled by eight the address lands far outside the written data.
        cpu.set_register("A3", 0).unwrap();
        cpu.set_register("B4", 60).unwrap();
        cpu.write_bytes(0x0080_9e38, &0xdead_beef_cafe_babeu64.to_le_bytes());
        eprintln!("PRE {:02x?}", cpu.host_read_bytes(0x0080_9e34, 12));
        cpu.run(4);
        eprintln!("MID {:02x?}", cpu.host_read_bytes(0x0080_9e34, 12));
        cpu.run(16);
        eprintln!(
            "DBG A3={:08x} A4={:08x} A5={:08x} A6={:08x}",
            cpu.register('A', 3),
            cpu.register('A', 4),
            cpu.register('A', 5),
            cpu.register('A', 6)
        );
        assert_eq!(cpu.register('A', 4), 0xcafe_babe);
        assert_eq!(cpu.register('A', 5), 0xdead_beef);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn compact_ndw_post_increment_advances_by_access_size() {
        // Packet lifted from DSP_maxval_cn -O2: the updating compact LDNDW walks a short
        // array with an eight-byte stride per iteration (ucst scaled by the access size,
        // SPRU732J Figure C-12 "ucst2 scaled only"), while the Doff4DW constant form
        // counts raw bytes ("ucst4 unscaled only").
        let mut cpu = Cpu::new(0);
        cpu.write_words(
            0,
            &[
                0x0e67_2ce6,
                0x9247_0d5c,
                0x6c6e_1e5b,
                0x8262_e2e2,
                0x0eba_0c6e,
                0x0210_985a,
                0xe7e4_0098,
                0xe7e4_0098,
            ],
        );
        // Halfword pattern 1100, 1102, ... at the base; A6 starts there.
        for i in 0u32..16 {
            let v = (0x1100u32 + i * 0x11).to_le_bytes();
            cpu.write_bytes(0x1000 + i * 2, &v);
        }
        cpu.set_register("A6", 0x1000).unwrap();
        cpu.set_register("B4", 0).unwrap();
        cpu.set_register("A4", 0).unwrap();
        cpu.run(24);
        assert_eq!(cpu.register('A', 6), 0x1008); // one LDNDW chunk = eight bytes
        assert_eq!(cpu.register('A', 4), 0x1100); // low halfword sign-extended
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn bdec_branches_and_decrements_while_the_register_is_positive() {
        // `BDEC target,A0` branches whenever A0 >= 0 and decrements it; the 40-bit
        // `SHL A1,A2,A3:A2` in the packet before it widens an int into a register pair.
        let mut cpu = Cpu::new(0);
        cpu.write_words(
            0,
            &[
                0x000c_a358,
                0x0084_a358,
                0x0114_a358,
                0x0000_2000,
                0x0104_44e0,
                0x0001_1020,
                0x0000_8000,
                0x0224_a358,
                0x029c_a358,
                0x0000_8000,
            ],
        );
        cpu.run(20);
        assert_eq!(cpu.register('A', 2), 32);
        assert_eq!(cpu.register('A', 3), 0);
        assert_eq!(cpu.register('A', 0), 2);
        assert_eq!(cpu.register('A', 5), 7);
        // The branch is taken, so the instruction after its delay slots never runs.
        assert_eq!(cpu.register('A', 4), 0);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn taken_branch_idles_the_loop_buffer() {
        // A conditional return sitting five delay slots ahead of an `SPLOOP` must abort the
        // buffer: TI's memcpy tail relies on the loop never starting when the branch is taken.
        let mut cpu = Cpu::new(0);
        cpu.set_register("B0", 1).unwrap();
        cpu.set_register("B3", 0x100).unwrap();
        cpu.set_register("A4", 7).unwrap();
        cpu.write_words(
            0,
            &[
                0x2000_0362, // [B0] B.S2 B3
                0x0000_0000, // NOP
                0x0000_0000, // NOP
                0x0000_0000, // NOP
                0x0000_0000, // NOP
                0x0083_8000, // SPLOOP 2
                0x0210_2059, // ADD.L1 1,A4,A4  (loop body, must not replay)
                0x0000_4000, // NOP 3
                0x0003_4000, // SPKERNEL 0,0
            ],
        );
        // 0x100: the branch target simply stops fetching new work.
        cpu.write_words(0x100, &[0x0000_8000, 0x0000_8000, 0x0000_8000]);
        cpu.run(24);
        // The SPLOOP itself is the fifth delay slot, so the buffer starts and is then idled
        // by the branch; the loop body one packet later never issues at all.
        assert!(!cpu.loop_active());
        assert_eq!(cpu.register('A', 4), 7);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn compact_stack_pointer_forms_use_opposite_update_modes() {
        // `STW.D2T2 B11,*B15--[2]` post-decrements B15 and `LDW.D2T2 *++B15[2],B13`
        // pre-increments it, so a save/restore pair round-trips through the same slot.
        let mut cpu = Cpu::new(0);
        cpu.set_register("B15", 0x1000).unwrap();
        cpu.set_register("B11", 0xdead_beef).unwrap();
        let mut words = compact_packet(0x35f7, 0);
        words[0] = (words[0] & 0x0000_ffff) | (0x76f7u32 << 16);
        cpu.write_words(0, &words);
        cpu.run(10);
        assert_eq!(cpu.register('B', 15), 0x1000);
        assert_eq!(cpu.register('B', 13), 0xdead_beef);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn protected_loads_insert_four_nop_cycles() {
        // `LDW.D1T1 *A6++[1],A3` followed by `MV.L1 A3,A4`. With the header PROT bit the load
        // is followed by four NOP cycles, so the move already sees the loaded word.
        let build = |flags: u32| {
            let mut words = compact_packet(0x0d3c, flags);
            words[0] = (words[0] & 0x0000_ffff) | (0x81c6u32 << 16);
            words
        };
        let mut protected = Cpu::new(0);
        protected.set_register("A6", 0x1000).unwrap();
        protected.write_bytes(0x1000, &0x4433_2211u32.to_le_bytes());
        protected.write_words(0, &build(1 << 20));
        protected.run(10);
        assert_eq!(protected.register('A', 3), 0x4433_2211);
        assert_eq!(protected.register('A', 4), 0x4433_2211);
        assert_eq!(protected.fault(), None);

        let mut plain = Cpu::new(0);
        plain.set_register("A6", 0x1000).unwrap();
        plain.write_bytes(0x1000, &0x4433_2211u32.to_le_bytes());
        plain.write_words(0, &build(0));
        plain.run(10);
        assert_eq!(plain.register('A', 3), 0x4433_2211);
        assert_eq!(plain.register('A', 4), 0);
    }

    #[test]
    fn protected_compact_loads_of_every_family_insert_their_nop_cycles() {
        // `LDW.D2T2 *B4[0],B5` belongs to the pointer-register compact family, which the PROT
        // rule must cover too: with the header bit set its four NOP cycles let the following
        // `ADD.L2 B5,-1,B0` see the loaded word. Both encodings are lifted from the RTS
        // printf conversion loop, which is what broke while this family was not recognised.
        let build = |flags: u32| {
            let mut words = compact_packet(0x105d, flags);
            words[0] = (words[0] & 0x0000_ffff) | (0xee81u32 << 16);
            words
        };
        let mut protected = Cpu::new(0);
        protected.set_register("B4", 0x1000).unwrap();
        protected.write_bytes(0x1000, &0x2233_4455u32.to_le_bytes());
        protected.write_words(0, &build(1 << 20));
        protected.run(14);
        assert_eq!(protected.register('B', 5), 0x2233_4455);
        assert_eq!(protected.register('B', 0), 0x2233_4454);
        assert_eq!(protected.fault(), None);

        let mut plain = Cpu::new(0);
        plain.set_register("B4", 0x1000).unwrap();
        plain.write_bytes(0x1000, &0x2233_4455u32.to_le_bytes());
        plain.write_words(0, &build(0));
        plain.run(14);
        assert_eq!(plain.register('B', 5), 0x2233_4455);
        assert_eq!(plain.register('B', 0), 0xffff_ffff);
    }

    #[test]
    fn an_execute_packet_spanning_fetch_packets_protects_only_the_loads_of_the_protected_one() {
        // SPRU732J §3.9.2.2: the header's PROT bit describes its own fetch packet. The first
        // fetch packet here is PROTected and ends in a compact NOP whose header p-bit (slot 13) chains it to the
        // `LDW.D1T1 *A6++[1],A3` at the start of the second, whose header is plain - so that
        // load keeps its four delay slots and the next packet's `MV.L1 A3,A4` still sees the
        // old A3. This is the shape TI's 8.3.15 compiler emits for `[B0] BNOP loop,2 ||
        // [B0] LDW` kernels, where the load's delay slots are counted against the branch.
        let mut words = [0x0c6e_0c6eu32; 16];
        words[7] = 0xe000_0000 | 0x0fe0_0000 | (1 << 20) | (1 << 13);
        words[8] = 0x81c6_0d3c;
        words[15] = 0xe000_0000 | 0x0fe0_0000;
        let mut cpu = Cpu::new(0);
        cpu.set_register("A6", 0x1000).unwrap();
        cpu.write_bytes(0x1000, &0x4433_2211u32.to_le_bytes());
        cpu.write_words(0, &words);
        cpu.run(30);
        assert_eq!(cpu.register('A', 3), 0x4433_2211, "the load still lands");
        assert_eq!(cpu.register('A', 4), 0, "no PROT NOP cycles follow it");
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn predicated_off_bnop_still_idles_its_nop_cycles() {
        // `[B0] BNOP bnp,1` with B0 == 0 does not branch, but its NOP cycle is generated in
        // decode, so the following MV runs two cycles after the MPY and sees the product.
        let mut cpu = Cpu::new(0);
        cpu.write_words(
            0,
            &[
                0x0000_a35a,
                0x000c_a358,
                0x0094_a358,
                0x0000_2000,
                0x0104_0c80,
                0x2000_2120,
                0x0188_0fd8,
                0x0000_8000,
            ],
        );
        cpu.run(10);
        assert_eq!(cpu.register('A', 2), 15);
        assert_eq!(cpu.register('A', 3), 15);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn sploopw_exits_when_its_predicate_goes_false() {
        // `[B1] SPLOOPW 2` is a do-while: ILC is unused. The predicate B1 is decremented in the
        // first stage, so it reads zero from cycle 5. Termination is the predicate as it stood
        // three cycles before a stage boundary (SPRU732J 7.10.2): the boundary at cycle 8 sees
        // cycle 5, so the iteration that starts at cycle 6 runs and the body runs four times.
        let mut cpu = Cpu::new(0);
        cpu.write_words(
            0,
            &[
                0x008c_a35a,
                0x0200_a358,
                0x0000_2000,
                0x4083_e000,
                0x0210_2059,
                0x0087_e05a,
                0x0000_4000,
                0x0000_0000,
                0x0003_4001,
                0x0290_06a0,
                0x0000_8000,
            ],
        );
        cpu.run(24);
        assert_eq!(cpu.register('A', 4), 4);
        // The iteration that starts at cycle 6 decrements B1 once more: zero becomes -1.
        assert_eq!(cpu.register('B', 1), 0xffff_ffff);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn a_compact_compare_writes_a_predicate_register_even_with_rs_set() {
        // `CMPLT.L2 0,B22,B0` in the `l_lx1c` form: the destination field is one bit wide and
        // names a predicate register, so the header RS bit must not remap it into B16-B23.
        // The compiler predicates the following packet on B0, which only works this way.
        let mut cpu = Cpu::new(0);
        cpu.set_register("B22", 4).unwrap();
        cpu.write_words(0, &compact_packet(0x1327, 1 << 19));
        cpu.run(6);
        assert_eq!(cpu.register('B', 0), 1);
        assert_eq!(cpu.register('B', 16), 0);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn compact_register_fields_follow_the_header_register_set_bit() {
        // `MV.L2 B4,B18` only reaches B18 when the fetch packet header sets RS.
        let mut cpu = Cpu::new(0);
        cpu.set_register("B4", 0x1234).unwrap();
        cpu.write_words(0, &compact_packet(0x4207, 1 << 19));
        cpu.run(6);
        assert_eq!(cpu.register('B', 18), 0x1234);
        assert_eq!(cpu.register('B', 2), 0);
        assert_eq!(cpu.fault(), None);

        let mut plain = Cpu::new(0);
        plain.set_register("B4", 0x1234).unwrap();
        plain.write_words(0, &compact_packet(0x4207, 0));
        plain.run(6);
        assert_eq!(plain.register('B', 2), 0x1234);
        assert_eq!(plain.register('B', 18), 0);
    }

    #[test]
    fn compact_add_constant_field_zero_means_eight() {
        // `ADD.L2 B5,8,B5`: the `scst_l3i` field is zero because zero would be a no-op.
        let mut cpu = Cpu::new(0);
        cpu.set_register("B5", 0x100).unwrap();
        cpu.write_words(0, &compact_packet(0x06d1, 0));
        cpu.run(6);
        assert_eq!(cpu.register('B', 5), 0x108);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn parallel_packet_reads_one_register_snapshot() {
        let mut cpu = Cpu::new(0);
        cpu.write_words(
            0,
            &[
                0x0004_a358,
                0x0088_a358,
                0x0080_0fd9,
                0x0004_06a0,
                0x0000_8000,
            ],
        );
        cpu.run(8);
        assert_eq!(cpu.register('A', 0), 2);
        assert_eq!(cpu.register('A', 1), 1);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn load_result_appears_after_four_delay_slots() {
        let mut cpu = Cpu::new(0);
        cpu.set_register("A0", 0x1000).unwrap();
        cpu.write_bytes(0x1000, &0x4433_2211u32.to_le_bytes());
        cpu.write_words(
            0,
            &[
                0x0080_0264,
                0x0104_0fd8,
                0x0000_4000,
                0x0184_0fd8,
                0x0000_8000,
            ],
        );
        cpu.run(4);
        assert_eq!(cpu.register('A', 1), 0);
        cpu.run(2);
        assert_eq!(cpu.register('A', 1), 0x4433_2211);
        assert_eq!(cpu.register('A', 3), 0x4433_2211);
    }

    #[test]
    fn sploop_overlays_four_iterations_at_ii_one() {
        let mut cpu = Cpu::new(0);
        cpu.write_words(
            0,
            &[
                0x0000_022a,
                0x0680_03a2,
                0x0000_6000,
                0x0000_a358,
                0x0003_8000,
                0x0000_0000,
                0x0003_4001,
                0x0000_2058,
                0x0000_0000,
            ],
        );
        cpu.run(13);
        assert_eq!(cpu.register('A', 0), 4);
        assert_eq!(cpu.cycles(), 13);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn same_functional_unit_in_parallel_faults() {
        let mut cpu = Cpu::new(0);
        cpu.write_words(0, &[0x0080_1079, 0x0100_10f8]);
        cpu.run(1);
        assert_eq!(
            cpu.fault(),
            Some("functional unit conflict on .L1 at 0x00000000 and 0x00000004")
        );
    }

    /// A packet that fails validation is annulled, but the instructions already past E1 are
    /// not: they complete as the pipeline drains into the exception (SPRU732J 6.3), so a load
    /// issued four cycles earlier lands on the cycle the bad packet would have executed.
    #[test]
    fn a_rejected_packet_does_not_hold_back_writes_already_in_flight() {
        let mut cpu = Cpu::new(0);
        cpu.control.write(0x1a, 1 << 2); // TSR.GEE
        cpu.control.write(0x05, 0x400); // ISTP
        cpu.set_register("A0", 0x1000).unwrap();
        cpu.write_bytes(0x1000, &0x7654_3210u32.to_le_bytes());
        // LDW *A0,B1, three NOPs, then two instructions claiming .L1 in one packet on the
        // cycle the load's result is written.
        cpu.write_words(0, &[0x0080_0066, 0, 0, 0, 0x0080_1079, 0x0100_10f8]);
        cpu.run(5);
        assert_eq!(cpu.fault(), None);
        assert_eq!(cpu.register('B', 1), 0x7654_3210);
        assert_eq!(cpu.control.read(0x1d, 0), 1 << 1); // EFR.IXF: the exception was taken
    }
    #[test]
    fn register_branch_executes_five_delay_slots() {
        let mut cpu = Cpu::new(0);
        cpu.set_register("B3", 0x20).unwrap();
        cpu.write_words(
            0,
            &[0x000c_0362, 0, 0, 0, 0, 0, 0x02a4_a358, 0, 0x039c_a358],
        );
        cpu.run(7);
        assert_eq!(cpu.register('A', 5), 0);
        assert_eq!(cpu.register('A', 7), 7);
        assert_eq!(cpu.fault(), None);
    }

    #[test]
    fn data_path_is_independent_of_address_side() {
        let mut cpu = Cpu::new(0);
        cpu.set_register("A0", 0x1000).unwrap();
        cpu.write_bytes(0x1000, &0x7654_3210u32.to_le_bytes());
        cpu.write_words(0, &[0x0080_0066, 0x0000_8000]);
        cpu.run(6);
        assert_eq!(cpu.register('A', 1), 0);
        assert_eq!(cpu.register('B', 1), 0x7654_3210);
        assert_eq!(cpu.fault(), None);
    }

    /// An exception recognised while `TSR.EXC` is set cannot save its context without
    /// destroying the outer one, so the CPU leaves NTSR and NRP alone, saves into the
    /// interrupt pair, clears NMIE and vectors through the service table's RESET entry
    /// (SPRU732J 6.3.4).
    #[test]
    fn a_nested_exception_saves_into_the_interrupt_pair_and_vectors_to_reset() {
        let mut cpu = Cpu::new(0);
        cpu.control.write(0x1a, 1 << 2); // TSR.GEE
        cpu.control.write(0x05, 0x400); // ISTP
        cpu.control.write(0x04, 1 << 1); // IER.NMIE
                                         // An illegal opcode, and another one in the exception vector it lands on.
        cpu.write_words(0, &[0x5645_4314]);
        cpu.write_words(0x420, &[0x5645_4314]);
        cpu.write_words(0x400, &[0x0480_03a8]); // MVK 7,A9 at the RESET entry
        cpu.run(40);
        assert_eq!(cpu.fault(), None);
        assert_eq!(cpu.register('A', 9), 7);
        // The first exception's context survives the second one.
        assert_eq!(cpu.control.read(0x07, 0), 0); // NRP names the first faulting packet
        assert_eq!(cpu.control.read(0x1c, 0), 0x0001_0004); // NTSR: GEE plus HWE
                                                            // The nested one saved the handler's state into IRP/ITSR and disabled NMIE.
        assert_eq!(cpu.control.read(0x06, 0), 0x420);
        assert_eq!(cpu.control.read(0x1b, 0), 0x0000_0404); // ITSR: GEE and EXC
        assert_eq!(cpu.control.read(0x04, 0) & 2, 0);
    }

    /// `SWE` is a service request, not a fault: it completes, so NRP names the packet after
    /// it and `B NRP` resumes there. With exceptions disabled it does nothing at all.
    #[test]
    fn swe_vectors_and_resumes_after_the_instruction() {
        let mut cpu = Cpu::new(0);
        cpu.control.write(0x1a, 1 << 2);
        cpu.control.write(0x05, 0x400);
        cpu.write_words(0, &[0x1000_0000, 0x0480_03a8]); // SWE then MVK 7,A9
        cpu.run(20);
        assert_eq!(cpu.fault(), None);
        assert_eq!(cpu.control.read(0x07, 0), 4); // NRP: the packet after the SWE
        assert_eq!(cpu.control.read(0x1d, 0), 1); // EFR.SXF, sticky until ECR clears it
        assert_eq!(cpu.control.read(0x1c, 0), 4); // NTSR: GEE, HWE clear for software
        assert_eq!(cpu.register('A', 9), 0); // the packet after SWE was annulled

        let mut disabled = Cpu::new(0);
        disabled.write_words(0, &[0x1000_0000, 0x0480_03a8]);
        disabled.run(20);
        assert_eq!(disabled.control.read(0x1d, 0), 0);
        assert_eq!(disabled.register('A', 9), 7);
    }

    /// Returning into User mode installs the mode at the branch target: the delay slots of
    /// the `B NRP` still run in Supervisor, so a restricted instruction there is legal and
    /// the same instruction one packet later is not (SPRU732J 8.1).
    #[test]
    fn the_restored_mode_takes_effect_at_the_branch_target() {
        let mut cpu = Cpu::new(0);
        cpu.control.write(0x1a, 1 << 2); // TSR.GEE
        cpu.control.write(0x05, 0x400); // ISTP
        cpu.control.write(0x1c, 0x44); // NTSR: GEE and CXM, the User context
        cpu.control.write(0x07, 0x40); // NRP
        cpu.write_words(
            0,
            &[
                0x001c_00e2, // B .S2 NRP
                0x0001_e000, // IDLE, in the first delay slot: still Supervisor
                0,
                0,
                0,
                0,
            ],
        );
        // The IDLE in the delay slot parks the CPU rather than faulting.
        cpu.run(3);
        assert_eq!(cpu.fault(), None);
        assert_eq!(cpu.control.read(0x1f, 0), 0); // IERR: nothing reported
        assert!(cpu.idling);

        // The same instruction at the branch target is a privilege violation.
        let mut user = Cpu::new(0);
        user.control.write(0x1a, 1 << 2);
        user.control.write(0x05, 0x400);
        user.control.write(0x1c, 0x44);
        user.control.write(0x07, 0x40);
        user.write_words(0, &[0x001c_00e2, 0, 0, 0, 0, 0]);
        user.write_words(0x40, &[0x0001_e000]); // IDLE at the target
        user.run(12);
        assert_eq!(user.fault(), None);
        assert_eq!(user.control.read(0x1f, 0), (1 << 3) | (1 << 6)); // OPX and PRX
        assert_eq!(user.control.read(0x1d, 0), 1 << 1); // EFR.IXF
        assert!(!user.idling);
    }

    #[test]
    fn compact_naming_and_resources_agree_over_the_whole_word_space() {
        // The permanent invariant: the decoder, the naming projection and the resource
        // projection must agree on which compact words are valid instructions, over every
        // 16-bit word and every header bit family identification reads (SAT 14, BR 15,
        // DSZ 16-18, RS 19). The differential only ever sees the ~38K compact instructions
        // in the built images; this sweep sees the other 60K words, where the old cascades
        // disagreed on the nameless `MvkCond` op/unit combinations until the merge made
        // them invalid encodings by construction.
        let mut disagreements = Vec::new();
        for raw in 0u32..=0xffff {
            for sat in 0..=1 {
                for br in 0..=1 {
                    for dsz in 0..8 {
                        for rs in 0..=1 {
                            let header = (sat << 14) | (br << 15) | (dsz << 16) | (rs << 19);
                            let word = raw as u16;
                            let named = crate::decoder::decode_compact(word, header).is_some();
                            let resources_ok =
                                crate::view::Instruction::new(0, u32::from(word), Some(header))
                                    .resources()
                                    .is_ok();
                            if named != resources_ok && disagreements.len() < 8 {
                                disagreements.push((word, header, named, resources_ok));
                            }
                        }
                    }
                }
            }
        }
        assert!(
            disagreements.is_empty(),
            "naming and resources disagree: {disagreements:?}"
        );
    }

    #[test]
    fn compact_family_counts_are_stable() {
        // Classification snapshot over the whole compact space: the per-family counts must
        // not drift when the decoder is edited, so an identification change outside the
        // differential's coverage still fails the suite. Values frozen at the merge.
        use std::collections::BTreeMap;
        let mut counts: BTreeMap<&'static str, u64> = BTreeMap::new();
        for raw in 0u32..=0xffff {
            for sat in 0..=1 {
                for br in 0..=1 {
                    for dsz in 0..8 {
                        for rs in 0..=1 {
                            let header = (sat << 14) | (br << 15) | (dsz << 16) | (rs << 19);
                            let name = crate::decoder::decode_compact(raw as u16, header)
                                .map(crate::decoder::CompactOpcode::mnemonic)
                                .unwrap_or("<none>");
                            *counts.entry(name).or_default() += 1;
                        }
                    }
                }
            }
        }
        let frozen: BTreeMap<&str, u64> = BTreeMap::from([
            ("<none>", 407_040),
            ("add", 396_288),
            ("addaw", 36_864),
            ("addk", 32_768),
            ("and", 32_768),
            ("bnop", 212_992),
            ("callp", 65_536),
            ("clr", 32_768),
            ("cmpeq", 49_152),
            ("cmpgt", 36_864),
            ("cmpgtu", 36_864),
            ("cmplt", 36_864),
            ("cmpltu", 36_864),
            ("ext", 16_384),
            ("extu", 49_152),
            ("ldb", 57_344),
            ("ldbu", 28_672),
            ("lddw", 61_440),
            ("ldh", 57_344),
            ("ldhu", 28_672),
            ("ldndw", 57_344),
            ("ldnw", 28_672),
            ("ldw", 212_992),
            ("mpy", 32_768),
            ("mpyh", 32_768),
            ("mpyhl", 32_768),
            ("mpylh", 32_768),
            ("mv", 393_216),
            ("mvc", 1_024),
            ("mvk", 325_632),
            ("neg", 2_048),
            ("nop", 512),
            ("or", 32_768),
            ("sadd", 98_304),
            ("set", 32_768),
            ("shl", 90_112),
            ("shr", 90_112),
            ("shru", 24_576),
            ("smpy", 32_768),
            ("smpyh", 32_768),
            ("smpyhl", 32_768),
            ("smpylh", 32_768),
            ("spkernel", 4_096),
            ("sploop", 1_024),
            ("sploopd", 3_072),
            ("spmask", 4_096),
            ("spmaskr", 4_096),
            ("sshl", 8_192),
            ("ssub", 65_536),
            ("stb", 86_016),
            ("stdw", 61_440),
            ("sth", 86_016),
            ("stndw", 57_344),
            ("stnw", 28_672),
            ("stw", 212_992),
            ("sub", 166_912),
            ("subaw", 4_096),
            ("xor", 35_840),
        ]);
        assert_eq!(counts, frozen, "compact family classification drifted");
    }
}
