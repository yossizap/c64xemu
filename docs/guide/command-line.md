# Command line

The exhaustive reference for the three subcommands `c64xemu` exposes — `decode`, `run` and
`debug` — every option they take, and the artefacts a batch run produces: the state report, the
`--dump` windows, the `--perf` report and the process exit code.

## Subcommands

`src/main.rs` defines one `Cli` with a required subcommand and nothing else; there are no global
options beyond clap's own.

| Command | Argument | What it does |
|---|---|---|
| `decode` | `WORD[/HEADER]...` | Names instruction words through the instruction view without loading an image. |
| `run` | `IMAGE` | Loads an ELF image, executes it to its exit point, writes the state report. |
| `debug` | `IMAGE` | Loads an ELF image and drives it interactively: console REPL, browser dashboard, or GDB stub. |
| `help` | `[COMMAND]` | clap's own help subcommand. |

`-h`/`--help` and `-V`/`--version` work on the top level and, because the command carries
`propagate_version`, on every subcommand.

`run` and `debug` share the same `MachineArgs` (`src/machine.rs`), flattened into both under the
`Machine` help heading. That is deliberate: an option that describes the machine — how memory is
loaded, what the device is strapped to, what is wired to its Ethernet and HPI, what is traced —
means exactly the same thing in a batch run as in a debugging session. Only the `Batch run` group
(`src/runner.rs`) and the `Session` group (`src/debugger.rs`) differ.

## Value syntaxes

| Syntax | Parser | Accepted forms |
|---|---|---|
| `ADDR` | `parse_address`, `src/main.rs` | Decimal, or hexadecimal with a `0x`/`0X` prefix. Nothing else: no `$`, no trailing `h`. |
| `ADDR\|SYM` | `Machine::resolve`, `src/machine.rs` | An ELF symbol name, `.` for the current PC, or an `ADDR`. The symbol table is consulted first, so a symbol named `0x100` would win. |
| `FILE:ADDR` | `parse_file_address`, `src/machine.rs` | A path, `:`, and an `ADDR`. Split on the first `:`. |
| `ADDR:LEN` | `parse_dump`, `src/runner.rs` | An `ADDR`, `:`, and a decimal byte count. The length is decimal only — `0x` is rejected. |
| `MAC` | `parse_mac`, `src/machine.rs` | Six colon-separated hexadecimal byte pairs, `AA:BB:CC:DD:EE:FF`. |
| `WORD[/HEADER]` | `parse_instruction`, `src/decode_cli.rs` | Hexadecimal always, `0x` prefix optional. `/HEADER` marks the word as a compact instruction and is refused if the word exceeds 16 bits. |
| `[ADDR]` on `--ui`/`--gdb` | clap `num_args = 0..=1` | A `host:port` string passed to `TcpListener::bind`. Omitting the value takes the `default_missing_value`; omitting the flag leaves the option unset. |

Poll intervals (`--net-poll`, `--hpi-poll`, `--utopia-poll`) go through `non_zero_cycles`, which rejects `0` with
`poll interval must be at least one cycle` — a zero interval would poll the socket on every
emulated cycle rather than never.

## decode

```
c64xemu decode <WORD[/HEADER]>...
```

At least one word is required; each is decoded independently at address 0 and printed as one line
of JSON. There is no image, no memory and no device: `src/decode_cli.rs` constructs
`view::Instruction::new(0, word, header)` and serialises it, which is the same value and the same
serialisation the differential against TI's `dis6x` consumes. See
[Instruction decoding](../internals/decoding.md) for what the fields mean.

A 32-bit word, with and without a predicate:

```console
$ c64xemu decode 0x0217e5f8 0x2217e5f8 0x3217e5f8
{"mnemonic":"subu","operands":"a31,a5,a5:a4","unit":"l1"}
{"mnemonic":"subu","operands":"a31,a5,a5:a4","predicate":"b0","unit":"l1"}
{"mnemonic":"subu","operands":"a31,a5,a5:a4","predicate":"!b0","unit":"l1"}
```

A compact instruction cannot be named without its fetch-packet header, so `decode` takes the
header alongside the word. The same 16 bits mean different things under different headers:

```console
$ c64xemu decode 0x1e1e 0x1e1e/0x4000 0xb407/0xe0080001
{"address":"*+b14[30]","data_path":"t2","mnemonic":"ldbu","unit":"d2"}
{"mnemonic":"smpy","operands":"a0,b4,a6","unit":"m1x"}
{"mnemonic":"mv","operands":"a8,b21","unit":"l2x"}
```

`0x1e1e` with no header is read as a 32-bit encoding — a `ldbu` off B14. With header `0x4000`,
whose bit 14 — the SAT bit `decode_compact` reads — is set, the same halfword is the compact
`smpy`. The third word needs bit 19 of header `0xe0080001`, the register-set bit
`compact_register` reads, to know that its destination is B21.

Field presence is meaningful. `Instruction`'s `Serialize` (`src/view.rs`) emits only the keys that
apply, in a fixed order: `address`, `data_path`, `mnemonic`, `operands`, `predicate`, `unit`. An
encoding the decoder does not name serialises as an empty object, and one that claims no
functional unit has no `unit` key:

```console
$ c64xemu decode 0x00000000 0x21042 0xf0a/0x8000
{"mnemonic":"nop"}
{}
{"mnemonic":"bnop"}
```

`decode` always exits 0 once its arguments parse.

## Machine options

Shared by `run` and `debug`.

| Option | Default | What it does |
|---|---|---|
| `--device NAME` | `c6455` | Selects the device profile by name; `DeviceProfile::named` refuses anything else with `unknown device NAME`. `c6455` is single-core, `c6472` has six cores. See [The six-core device](multicore.md). |
| `--boot-mode MODE` | `0` | The BOOTMODE pins. Mode 0 installs ELF segments and starts every core. Mode 1 requires `--hpi-unix`: the C6455 host releases core 0 with DSPINT; the C6472 host sets `BOOT_COMPLETE_STAT`. C6455 mode 5 and C6472 modes 2 (EEPROM at 0x50) and 3 (0x51) are the master I2C boot and require `--i2c-eeprom`. C6472 modes 5-8 require `--utopia-unix` (8/16-bit UTOPIA, paired PLL x10/x20), and modes 9/10 boot over MAC 0/1. Other modes run core 0 and leave it to release the rest. See [Booting an image](booting.md). |
| `--i2c-eeprom FILE` | none | The serial EEPROM image on the I2C bus that the master I2C boot reads its boot parameters and boot table from; refused with any other boot mode. Supplying it makes the load `Wire`: nothing comes from the ELF's segments. A C6472 EEPROM whose parameters hand over to UTOPIA also needs `--utopia-unix`. |
| `--boot-param-index N` | `0` | Which boot parameter block the I2C boot loads: CFGGP[2:0] on the C6455 (0-7), CFG[3:0] on the C6472 (0-15). The block sits at `0x80 * N`. |
| `--i2c-pll-x19` | off | Straps CFG[4] high for a C6472 I2C boot, so the boot sets PLL1 to x19 rather than x9 (SPRUEC6G Table 22). Refused elsewhere. |
| `--mac-id MAC` | `02:00:00:00:64:72` | The address fused into the MAC ID words, which an Ethernet boot names itself by. |
| `--boot-udp-port PORT` | `9` | The UDP port an Ethernet boot accepts boot-table frames on. |
| `--stop-pc ADDR\|SYM` | `C$$EXIT`, else `abort` | The program's exit point. Resolved once at load; if neither symbol is present and the option is absent, the machine has no stop address and `run` refuses to start. |
| `--break ADDR\|SYM` | none | Halt before this address executes. Repeatable. Resolved once at load and handed to the front end as its initial breakpoint set; the machine folds it and the exit point into the halt rule (`Machine::arm_halt`), so a branch landing directly on the address still stops. |
| `--boot-table FILE:ADDR` | none | Writes the file at `ADDR` and walks it as a ROM boot table instead of installing the ELF's segments; the table's entry point replaces the ELF's. The ELF is then only a symbol source ([SPRUEC6G](../reference/spruec6g.pdf) §6.2.2). Rejected together with a ROM boot mode: two boot sources are one too many. |
| `--boot-config FILE:ADDR` | none | A boot configuration table applied before anything else, as the ROM would. |
| `--pcap-in FILE` | none | Read at load and injected as received Ethernet frames. Refused with an Ethernet boot: the boot code resets the MAC before configuring it, so frames handed over earlier are gone. |
| `--pcap-out FILE` | none | Capture transmitted frames to a pcap file, written when the run finishes. |
| `--net-unix PATH` | none | Bind a Unix datagram socket for live Ethernet. An existing socket file is removed first, so a crashed emulator needs no cleanup (`Bridge::bind`, `src/netio.rs`). |
| `--net-peer PATH` | none | The Unix datagram socket transmitted frames are sent to. |
| `--net-poll CYCLES` | `1024` | How often the run loop looks at the Ethernet socket. Polling happens on a cycle boundary inside the run loop rather than on a thread, because the loop owns the MAC. |
| `--net-port PORT` | MAC 0, or the strapped boot port | Which of the device's MACs the backend's frames are delivered to. A port beyond the profile's MAC count is refused. |
| `--hpi-unix PATH` | none | Bind a Unix datagram socket for the HPI host side. |
| `--hpi-peer PATH` | none | The Unix datagram socket HPI replies are sent to. Host requests carry an HCNTL code as [SPRU969B](../reference/spru969b.pdf) Table 4 numbers it. |
| `--hpi-poll CYCLES` | `1024` | How often the run loop looks for host accesses. |
| `--utopia-unix PATH` | none | Bind the external ATM master's Unix datagram socket: addressed polls, cell writes and cell reads, with replies to the sender; see [The UTOPIA ATM master](networking-and-host-io.md#the-utopia-atm-master). Required for C6472 boot modes 5-8. |
| `--utopia-poll CYCLES` | `1024` | How often the run loop services ATM master requests; at least one cycle. |
| `--utopia-phy N` | `0` | C6472 UTOPIA boot's CFG[4:0] and UTOPIA SLID, 0-31. Address 31 is the unreachable null PHY. Refused outside `--device c6472 --boot-mode 5`-`8`. |
| `--trace FILE` | none | Write an execution trace. Without it every emit site is short-circuited by `enabled()` and nothing is allocated. |
| `--trace-format text\|json` | `text` | Aligned text for humans, JSON Lines for machines. |
| `--trace-ring N` | no ring | Keep only the last N events in memory and flush them at exit — the window before a fault or a hang, without paying to record the whole run. Without it every event is written as it happens. |
| `--trace-start CYCLE` | `0` | Drop events before this cycle (`Trace::emit`, `src/trace.rs`). |
| `--trace-exceptions` | off | Print `exception at 0xADDR: CAUSE` on stderr for every internal exception left to the firmware. Note that it also opts the run out of the fast resume path described below. |
| `--profile` | off | Keep the per-address packet histogram. Adds `profile` and `cache_stats` to the state report and feeds the debugger's hotspot views. |
| `--cache` | off | Program every core's L1P, L1D and L2 to the largest configuration a C64x+ megamodule has — 32 KB, 32 KB and 256 KB — by writing mode 4 into `L1PCFG`, `L1DCFG` and `L2CFG` as firmware would ([SPRU871K](../reference/spru871k.pdf)). Nothing else changes: MAR cacheability stays at its reset values and the caches come up cold, so the program answers the same and takes more cycles doing it. See [Memory and caches](../internals/memory-and-caches.md). |
| `--coherence-check` | off | Report on stderr, once per line, every DMA transfer that disagrees with what a core caches where the hardware keeps no coherence: a read or fetch of a line a master has since overwritten, a master reading past a dirty line, a dirty line written back over a master's data. Each report names the core, cycle, PC and the cache operation that was missing. What the program sees is unchanged. Opts the run out of the fast resume path. See [Memory and caches](../internals/memory-and-caches.md#the-coherence-check). |

The trace and network options are covered in use by [Debugging](debugging.md) and
[Networking and host I/O](networking-and-host-io.md).

Some of these change the shape of the run loop. `Machine::resume` takes a fast path,
`resume_budget`, when the only stop condition is a cycle budget and neither Ethernet history,
`--trace-exceptions` nor `--coherence-check` is active; those two therefore cost a per-cycle
check even on cycles that raise nothing.

## Batch run options

`run` takes the image as its one positional argument.

```
c64xemu run [OPTIONS] <IMAGE>
```

| Option | Default | What it does |
|---|---|---|
| `--max-cycles CYCLES` | `10000000` | Cycle budget for the whole run; `0` runs without a limit. Measured against the device cycle while a ROM boot is running and against the focused core's cycle counter afterwards (`run_clock`), because a held core retires nothing and could not pace a budget. Exhausting it is an error, not a clean finish. |
| `--state-out FILE` | stdout | Write the state report here instead of printing it. Dumps, the perf report and diagnostics stay on stderr either way. |
| `--dump ADDR:LEN` | none | Print a memory window on stderr when the run ends. Repeatable; windows print in the order given, before the state report. |
| `--stack-window START:END` | none | Observe the focused core's B15 and committed stores in a word-aligned stack window. Adds `stack_observation` to the state JSON. |
| `--stack-window-all START:END` | none | Observe B15 and committed stores on every core; adds ordered `stack_observations` entries to the state JSON. Mutually exclusive with `--stack-window`. |
| `--pc-trail N` | `0` | Keep the last N execute-packet addresses, collapsing runs of the same address. Printed under `last N execute packets:` when the run faults, and used to annotate a `--trap-oob` stop. Zero keeps nothing: the trail costs a push per cycle, so a plain batch run does not install the hook at all. |
| `--trap-oob` | off | Stop as soon as the program counter leaves every loaded segment. The check runs once per cycle against the ranges `Machine::loaded_ranges` reports, prints `fetch left the image at ...` plus up to twelve trail entries on stderr, and stops with `OutOfBounds`. An image whose memory came off the wire reports no ranges at all, so on an Ethernet or I2C boot the first cycle already looks out of bounds. |
| `--perf` | off | Print the performance report on stderr after the state report. |

Stack observation is passive and detects stores from CPU writeback when a store overlaps the
configured window. Stores entirely outside it are not flagged, and DMA or global memory aliases
are not covered. All-core mode includes cores held in reset; their entries remain uninitialized
until they execute a B15 write.

`--pc-trail` and `--trap-oob` share one stop-condition closure, so asking for the trail is what
makes a `--trap-oob` diagnostic useful.

## Session options

`debug` takes the image as its one positional argument and picks one of three front ends. They are
exclusive, not concurrent: both the dashboard and the stub own the breakpoint set and the run loop.

```
c64xemu debug [OPTIONS] <IMAGE>
```

| Option | Default | What it does |
|---|---|---|
| `--max-cycles CYCLES` | `1000000000` | Cycle budget for one `continue`, not for the session; `0` runs without a limit. Used by the console REPL and passed to the GDB stub. The dashboard does not take it. |
| `--ui [ADDR]` | `127.0.0.1:8080` when bare | Serve the browser dashboard. Binds a TCP listener and prints `c64xemu dashboard on http://ADDR`, then serves connections until killed. The dashboard's hotspot table is part of the page, so this session keeps the histogram whether or not `--profile` was given, and it enables the Ethernet history the live packet view reads. Conflicts with `--gdb`. |
| `--gdb [ADDR]` | `127.0.0.1:1234` when bare | Serve the GDB remote serial protocol. Binds a TCP listener, prints `GDB listening on ADDR`, accepts exactly one connection, prints `connection from PEER`, and serves it. |

With neither flag, `debug` runs the console REPL; it turns on the issue view so each stop can show
which functional unit issued what. All three front ends are documented in
[Debugging](debugging.md).

## The state report

`run` always writes one JSON object describing the finished machine — the same JSON the console
debugger's `state` command writes, because both are projections of one `MachineSnapshot`
(`Machine::state_json`). Keys are sorted.

```console
$ c64xemu run tests/firmware/bios_kernel/build/bios_kernel.out --max-cycles 20000000
bios kernel consumed 64 relayed 64 outer 22 inner 22 nested 22 swi 22 woken 16 failures 0 ticks 4 slept 2 letters 16 masked 1 events 4 held 1 released 1 checksum 3410629677
{
  "cio_output": "bios kernel consumed 64 relayed 64 outer 22 inner 22 nested 22 swi 22 woken 16 failures 0 ticks 4 slept 2 letters 16 masked 1 events 4 held 1 released 1 checksum 3410629677\n",
  "cores": [
    {
      "core": 0,
      "cycles": 4120027,
      "fault": null,
      "focused": true,
      "halted": true,
      "held_in_reset": false,
      "pc": 8473408
    }
  ],
  "cycles": 4120027,
  "device": "c6455",
  "exit_code": 0,
  "fault": null,
  "halted": true,
  "loop_active": false,
  "net_dropped": 0,
  "pc": 8473408,
  "registers": {
    "A0": 0,
    ...
    "B9": 16
  },
  "stats": {
    "branches_taken": 398049,
    "faults": 0,
    "halt_cycles": 0,
    "idle_cycles": 0,
    "interrupt_wait_cycles": 897,
    "interrupt_waited": 46,
    "interrupts": 56,
    "packets": 2126903,
    "replay_packets": 95132,
    "retired": 3308382,
    "saturations": 0,
    "stall_interrupt": 504,
    "stall_memory": 12927,
    "stall_nop": 1903406,
    "stores": 427175
  },
  "tx_frames": []
}
```

The first line is not part of the report: CIO writes are echoed as the program makes them, to
stdout or stderr according to the file descriptor the request named, and also accumulated into
`cio_output`.

| Field | Meaning |
|---|---|
| `halted` | The snapshot's `at_stop`, not the CPU's halt flag: some core that is not held in reset is parked ready to fetch the exit point. This is what "the program finished" means, and `run` treats anything else as a failure. |
| `fault` | The focused core's fault message, or `null`. |
| `pc` | The focused core's program counter, decimal. |
| `cycles` | The focused core's cycle counter. |
| `loop_active` | The SPLOOP buffer is live, which `TSR.SPLX` mirrors. |
| `exit_code` | The low byte of A4, the EABI return register. |
| `registers` | The focused core's two register files as `A0`…`A31`, `B0`…`B31`, decimal. |
| `cio_output` | Everything the program wrote through the CIO handshake. |
| `tx_frames` | Frames drained from the MAC that no backend took, hex-encoded. |
| `net_dropped` | Frames the Ethernet backend dropped. |
| `device` | The device profile's name. |
| `cores` | One entry per core: index, cycle counter, `pc`, `halted`, `held_in_reset`, `fault`, and `focused` for the core the front ends drive. On a C6455 there is exactly one. |
| `stats` | The CPU's counters, which the perf report renders. |
| `ethernet_boot` | Present only when the ROM's Ethernet boot ran: `frames_accepted`, `frames_discarded`, `table_bytes`, `entry`. The only way to tell a boot that completed from one still waiting for a frame. |
| `i2c_boot` | Present only when the ROM's I2C boot ran: `parameter_blocks`, `config_blocks`, `table_blocks`, `table_bytes`, `checksum_failures`, `entry`, `error`. `entry` stays null when a C6472 parameter block handed the boot over to UTOPIA. |
| `utopia_boot` | Present for C6472 UTOPIA boot, cold or handed over from an I2C boot: `cells_accepted`, `cells_discarded`, `table_bytes` (32 per accepted cell, including final padding), `entry`, `error`. |
| `hpi_boot` | Present when strapped to HPI boot: `entries` holds each core's release entry, null until released; `entry` is core 0's entry. The C6455 entry is fixed at `0x00800000`; C6472 entries come from `DSP_BOOT_ADDRn`. |
| `profile` | Present with `--profile`: the 32 hottest execute-packet addresses as `{pc, where, packets}`, `where` being the covering symbol as `name+0x1c`. |
| `cache_stats` | Present with `--profile`: per cache, its name, `enabled`, `frozen`, `capacity_bytes`, `line_size`, `ways`, `sets`, `resident_lines`, `dirty_lines` and its hit/miss/fill/eviction counters. |

## Memory dumps

Each `--dump ADDR:LEN` prints a header line and the window as little-endian 32-bit words, four to
a row, on stderr. The window is read through the host view, so dumping neither fills a cache line
nor moves a replacement clock.

```console
$ c64xemu run tests/firmware/bios_kernel/build/bios_kernel.out --max-cycles 20000000 --dump 0x800000:32 >/dev/null
dump 0x00800000:
  0x00800000  10294810 07fee053 4646b407 028088fe
  0x00800010  02a80266 d3476747 030089fe e4800020
```

## The performance report

`--perf` renders the `stats` counters as a `perf stat`-style summary on stderr. Unlike host
`perf`, every number is exactly what the modelled pipeline retired, so the report is reproducible.
One cache block follows per cache the memory hierarchy exposes, whether or not `--profile` was
given.

```console
$ c64xemu run tests/firmware/bios_kernel/build/bios_kernel.out --max-cycles 20000000 --perf >/dev/null
Performance counter report for the emulated C64x+
         4120027 cycles           100.0% of emulated time
         2222035 executing         53.9%  (packets issued, incl. SPLOOP replays)
               0 idle               0.0%  (IDLE waiting for an interrupt)
         1916837 stall             46.5%  (pipeline frozen)
           12927   memory          (L1D/L1P miss, bank conflict, write buffer)
         1903406   nop             (multicycle NOP)
             504   interrupt       (pipe-down into a service routine)
               0 halted             0.0%  (parked on a stop address)
               0 unaccounted        0.0%  (faulted cycles and rounding)
         4120027 total cycles (including stalls)

  Instructions
         3308382 instructions retired
         2126903 packets issued
           95132 SPLOOP replays
            1.25 cycles per instruction (CPI)
            0.80 instructions per cycle (IPC)

  Branches and interrupts
          398049 branches taken
              56 interrupts taken
              46 interrupts with a wait
             897 cycles spent waiting to take a ready interrupt
           19.50 average wait cycles per deferred interrupt
             504 cycles of 9-cycle pipe-down (56 total)

  Memory and data
          427175 stores committed
               0 functional-unit saturations
               0 faults

  L1P  1024 lines, 32768 bytes, 1 ways, enabled=true
           100.0% read hit  (    16307811 hits,         3645 misses)
             0.0% write hit (           0 hits,            0 misses)
            3645 line fills,         2355 evictions,            0 writebacks,          539 invalidations

  L1D  512 lines, 32768 bytes, 2 ways, enabled=true
           100.0% read hit  (     1640823 hits,          153 misses)
            96.0% write hit (     1417334 hits,        58605 misses)
             153 line fills,           16 evictions,            7 writebacks,            4 invalidations

  L2  0 lines, 0 bytes, 4 ways, enabled=false
             0.0% read hit  (           0 hits,            0 misses)
             0.0% write hit (           0 hits,            0 misses)
               0 line fills,            0 evictions,            0 writebacks,            0 invalidations
```

`executing`, `idle`, `stall`, `halted` and `unaccounted` partition the cycle count: `unaccounted`
is what the other four do not cover, which is faulted cycles and rounding. The three indented
lines under `stall` break the stall total down. `cycles of 9-cycle pipe-down` is `interrupts * 9`,
the fixed pipe-down cost, not an independently measured counter. The average wait line appears only
when at least one interrupt was deferred. This report needs no `--profile`; the caches above show
enabled geometry because that firmware programs its own caches, not because of `--cache`.

## Exit codes and diagnostics

`main` maps the subcommand's result: `Ok(code)` becomes the process exit code, `Err(message)`
prints `c64xemu: MESSAGE` on stderr and exits 1.

| Code | Meaning |
|---|---|
| `0` | `decode` succeeded; `debug` finished; or `run` reached its exit point with A4's low byte zero. |
| `1` | Any error message, or `run` reached its exit point with A4's low byte 1. The stderr line distinguishes them. |
| other `0`–`255` | `run` reached its exit point and the program left this value in A4. |
| `2` | clap rejected the command line — an unknown option, a missing argument, or a value the parser refused. |

The state report is always written before the failure is reported, so a failed run is still
described. The messages `run` can fail with:

| Message | Cause |
|---|---|
| `ELF has no C$$EXIT symbol; pass --stop-pc` | The image has neither `C$$EXIT` nor `abort` and no `--stop-pc`, so there is nothing to stop at. Refused before any cycle runs. |
| `stopped at breakpoint 0x… NAME` | A `--break` address was reached. |
| `cycle limit N reached at PC 0x…` | `--max-cycles` was exhausted. |
| `fetch left the image at PC 0x…` | `--trap-oob` fired; the stderr lines above it name the trail. |
| the fault text | The CPU refused to execute something. Preceded on stderr by `fault: …` and, with `--pc-trail`, the trail. |

```console
$ c64xemu run tests/firmware/bios_kernel/build/bios_kernel.out --max-cycles 1000 >/dev/null
c64xemu: cycle limit 1000 reached at PC 0x00812e98
$ echo $?
1
$ c64xemu run tests/firmware/bios_kernel/build/bios_kernel.out --break main --max-cycles 20000000 >/dev/null
c64xemu: stopped at breakpoint 0x00802884 main
$ echo $?
1
```

SIGTERM is the exception. `run` installs a handler (`install_termination_handler`,
`src/main.rs`); the flag it sets stops the run at the next batch boundary, prints
`c64xemu: terminated by SIGTERM`, and still writes the state report and the perf report before
exiting with the program's `exit_code`. SIGINT is deliberately left at its default so Ctrl-C keeps
its meaning, and only the batch runner installs the handler — the interactive front ends keep
their own control of the terminal.

For a first run end to end see [Getting started](getting-started.md); for what the machine, the
run loop and the halt rule are, see [Architecture](../internals/architecture.md).
