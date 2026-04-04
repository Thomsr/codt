use std::{collections::HashSet, path::PathBuf};

use clap::{ArgAction, Args, Parser, Subcommand};
use codt::search::solver::{LowerBoundStrategy, SearchStrategyEnum, UpperboundStrategy};

use crate::clap_enum_variants;

use super::value_parser::RangedF64ValueParser;

pub fn get_cli_args() -> CliParams {
    CliParams::parse()
}

#[derive(Parser)]
#[command(
    version,
    disable_help_flag(true),
    next_line_help(true),
    flatten_help(true)
)]
pub struct CliParams {
    /// Determines if the solver should print logging information to standard output.
    #[arg(short, long, action=ArgAction::Set, default_value_t=false)]
    pub verbose: bool,

    /// Location of the (training) dataset.
    #[arg(short, long)]
    pub file: PathBuf,

    /// Optionally, the maximum amount of seconds to run, after which the best found solution is returned.
    #[arg(short, long, default_value_t = 60)]
    pub timeout: u64,

    /// Optionally, the maximum memory used in bytes, if the limit is hit, the best found solution is returned.
    #[arg(short, long, default_value_t=4 * 1024 * 1024 * 1024)]
    pub memory_limit: u64,

    #[arg(
        short,
        long,
        value_parser = clap_enum_variants!(LowerBoundStrategy),
        value_delimiter = ',',
        num_args = 1..,
        default_value = "class-count"
    )]
    pub lowerbound: HashSet<LowerBoundStrategy>,

    #[arg(short, long, value_enum, default_value = "for-remaining-interval")]
    pub upperbound: UpperboundStrategy,

    /// Determines if the solver should track intermediate solutions.
    #[arg(long, action=ArgAction::Set, default_value_t=false)]
    pub intermediates: bool,

    /// The search strategy to use.
    #[arg(short, long, value_parser=clap_enum_variants!(SearchStrategyEnum), default_value_t=SearchStrategyEnum::BfsBalanceSmallLb)]
    pub strategy: SearchStrategyEnum,
}

#[derive(Subcommand)]
pub enum OptimizationTaskEnum {
    /// Optimizes classification accuracy
    Accuracy(AccuracyParams),
}

#[derive(Args)]
#[command(next_help_heading = "Task Parameters")]
pub struct AccuracyParams {
    /// The cost for adding an extra node to the tree. 0.01 means one extra node is only jusitified if it results in at least one percent better training score.
    #[arg(long, value_parser=RangedF64ValueParser::<f64>::default().range(0.0..=1.0), default_value_t=0.0)]
    pub complexity_cost: f64,
}

#[derive(Args)]
#[command(next_help_heading = "Task Parameters")]
pub struct SquaredErrorParams {
    /// The cost for adding an extra node to the tree. 0.01 means one extra node is only jusitified if it results in at least one percent better training score.
    #[arg(long, value_parser=RangedF64ValueParser::<f64>::default().range(0.0..=1.0), default_value_t=0.0)]
    pub complexity_cost: f64,
}
