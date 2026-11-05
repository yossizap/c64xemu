//! Batch execution: load an image, run it to its exit point, report what the machine did.
//!
//! Everything about the machine itself - how memory is loaded, the Ethernet and HPI backends,
//! tracing - comes from `MachineArgs`, so this module is only the loop and the report.

use crate::machine::{Machine, MachineArgs, StopCondition, StopReason};
use clap::Args;
use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;

#[derive(Args)]
pub struct RunArgs {
    /// ELF image to execute
    pub image: PathBuf,
    #[command(flatten)]
    pub machine: MachineArgs,
    /// Give up after this many simulated cycles; 0 runs without a limit
    #[arg(
        long,
        default_value_t = 10_000_000,
        value_name = "CYCLES",
        help_heading = "Batch run"
    )]
    pub max_cycles: u64,
    /// Write the final machine state here instead of on stdout
    #[arg(long, value_name = "FILE", help_heading = "Batch run")]
    pub state_out: Option<PathBuf>,
    /// Print this memory window when the run ends; repeatable
    #[arg(long, value_parser = parse_dump, value_name = "ADDR:LEN", help_heading = "Batch run")]
    pub dump: Vec<(u32, usize)>,
    /// Observe B15 and committed stores within START:END without writing a guest watermark
    #[arg(long, value_parser = parse_stack_window, value_name = "START:END", help_heading = "Batch run")]
    pub stack_window: Option<(u32, u32)>,
    /// Observe B15 and committed stores on every core within START:END
    #[arg(long, value_parser = parse_stack_window, value_name = "START:END", conflicts_with = "stack_window", help_heading = "Batch run")]
    pub stack_window_all: Option<(u32, u32)>,
    /// Keep this many recent execute-packet addresses for fault diagnosis
    #[arg(
        long,
        default_value_t = 0,
        value_name = "N",
        help_heading = "Batch run"
    )]
    pub pc_trail: usize,
    /// Stop as soon as the program counter leaves every loaded segment
    #[arg(long, help_heading = "Batch run")]
    pub trap_oob: bool,
    /// Print an emulated-CPU performance report when the run ends
    #[arg(long, help_heading = "Batch run")]
    pub perf: bool,
}

pub fn run(args: RunArgs) -> Result<u8, String> {
    crate::install_termination_handler();
    let mut machine = Machine::load(&args.image, &args.machine)?;
    if let Some((start, end)) = args.stack_window {
        machine.configure_stack_observation(start, end)?;
    }
    if let Some((start, end)) = args.stack_window_all {
        machine.configure_stack_observation_all(start, end)?;
    }
    if machine.stop_address().is_none() {
        return Err("ELF has no C$$EXIT symbol; pass --stop-pc".into());
    }
    let breakpoints = machine.initial_breakpoints().to_vec();
    let loaded_ranges = machine.loaded_ranges();
    // A short history of executed packet addresses, so a fault can be traced back through the
    // call path that reached it. Only kept when asked for: it costs a push per cycle. The
    // check closure runs once per cycle inside `run`, so it records the trail and watches
    // for a fetch leaving the image in the same hook.
    let mut pc_trail: VecDeque<u32> = VecDeque::new();
    machine.arm_halt(&breakpoints);
    let reason = if args.pc_trail == 0 && !args.trap_oob {
        // The normal batch path only has a cycle budget. Do not dispatch an inert diagnostic
        // closure on every emulated cycle.
        let mut conditions = [StopCondition::Budget(args.max_cycles)];
        machine.run(&mut conditions)?
    } else {
        let mut conditions = [
            StopCondition::Budget(args.max_cycles),
            StopCondition::Check(&mut |machine: &Machine| {
                if args.pc_trail > 0 {
                    if pc_trail.len() == args.pc_trail {
                        pc_trail.pop_front();
                    }
                    let pc = machine.pc();
                    if pc_trail.back() != Some(&pc) {
                        pc_trail.push_back(pc);
                    }
                }
                if args.trap_oob {
                    let pc = machine.pc();
                    let inside = loaded_ranges
                        .iter()
                        .any(|(start, end)| pc >= *start && pc < *end);
                    if !inside {
                        eprintln!("fetch left the image at {}", machine.describe(pc));
                        if let Some(previous) = pc_trail.back() {
                            eprintln!("  came from {}", machine.describe(*previous));
                        }
                        for entry in pc_trail.iter().rev().take(12) {
                            eprintln!("  trail {}", machine.describe(*entry));
                        }
                        return Some(StopReason::OutOfBounds);
                    }
                }
                None
            }),
        ];
        machine.run(&mut conditions)?
    };
    for (address, length) in &args.dump {
        let bytes = machine.read_memory(*address, *length);
        eprintln!("dump 0x{address:08x}:");
        for (index, chunk) in bytes.chunks(16).enumerate() {
            let words: Vec<String> = chunk
                .chunks(4)
                .map(|word| {
                    let mut value = 0u32;
                    for (position, byte) in word.iter().enumerate() {
                        value |= u32::from(*byte) << (8 * position);
                    }
                    format!("{value:08x}")
                })
                .collect();
            eprintln!(
                "  0x{:08x}  {}",
                address + (index * 16) as u32,
                words.join(" ")
            );
        }
    }
    let halted = reason == StopReason::Exit;
    let fault = machine.fault().map(str::to_owned);
    if let Some(reason) = &fault {
        eprintln!("fault: {reason}");
        if !pc_trail.is_empty() {
            eprintln!("last {} execute packets:", pc_trail.len());
            for pc in &pc_trail {
                eprintln!("  {}", machine.describe(*pc));
            }
        }
    }
    let exit_code = machine.exit_code();
    machine.finish()?;
    let mut state = machine.state_json(args.machine.profile);
    if args.stack_window_all.is_some() {
        state.as_object_mut().unwrap().remove("stack_observation");
        state["stack_observations"] = serde_json::json!(machine.stack_observations());
    }
    let rendered = serde_json::to_vec_pretty(&state).map_err(|error| error.to_string())?;
    if let Some(path) = &args.state_out {
        fs::write(path, rendered).map_err(|error| format!("{}: {error}", path.display()))?;
    } else {
        println!("{}", String::from_utf8(rendered).unwrap());
    }
    if args.perf {
        eprintln!("{}", machine.perf_report());
    }
    if let Some(fault) = fault {
        return Err(fault);
    }
    if !halted {
        // The state dump is already written, so the message only has to say why it stopped.
        match reason {
            StopReason::Breakpoint(hit) => {
                return Err(format!(
                    "stopped at breakpoint {} on core {}",
                    machine.describe(hit.address),
                    hit.core
                ));
            }
            StopReason::OutOfBounds => {
                return Err(format!("fetch left the image at PC {:#010x}", machine.pc()));
            }
            StopReason::Terminated => {
                eprintln!("c64xemu: terminated by SIGTERM");
            }
            _ => {
                return Err(format!(
                    "cycle limit {} reached at PC {:#010x}",
                    args.max_cycles,
                    machine.pc()
                ));
            }
        }
    }
    Ok(exit_code)
}

/// `ADDR:LEN`: a memory window to print when the run ends.
fn parse_dump(text: &str) -> Result<(u32, usize), String> {
    let (address, length) = text
        .split_once(':')
        .ok_or_else(|| "expected ADDR:LEN".to_string())?;
    Ok((
        crate::parse_address(address)?,
        length
            .parse()
            .map_err(|_| format!("invalid length {length}"))?,
    ))
}

fn parse_stack_window(text: &str) -> Result<(u32, u32), String> {
    let (start, end) = text
        .split_once(':')
        .ok_or_else(|| "expected START:END".to_string())?;
    let start = crate::parse_address(start)?;
    let end = crate::parse_address(end)?;
    if start >= end || start & 3 != 0 || end & 3 != 0 {
        return Err("stack window must be non-empty and word aligned".into());
    }
    Ok((start, end))
}
