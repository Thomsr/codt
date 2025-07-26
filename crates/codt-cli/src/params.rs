use std::path::PathBuf;

use clap::{ArgAction, Args, Parser, Subcommand, value_parser};
use codt::search::solver::{
    BranchRelaxation, SearchStrategyEnum, TerminalSolver, UpperboundStrategy,
};

use crate::clap_enum_variants;

use super::value_parser::RangedF64ValueParser;

pub fn get_cli_args() -> CliParams {
    CliParams::parse()
}

#[derive(Parser)]
#[command(
    version,
    subcommand_required(true),
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

    /// Maximum allowed depth of the tree, where the depth is defined as the largest number of *branching nodes* from the root to any leaf. Depth greater than four is usually time consuming.
    #[arg(short('d'), long, default_value_t=3, value_parser=value_parser!(u32).range(0..20))]
    pub max_depth: u32,

    /// Optionally, the maximum amount of seconds to run, after which the best found solution is returned.
    #[arg(short, long)]
    pub timeout: Option<u64>,

    /// Optionally, the maximum memory used in bytes, if the limit is hit, the best found solution is returned.
    #[arg(short, long)]
    pub memory_limit: Option<u64>,

    #[arg(short, long, value_enum, default_value = "solutions-only")]
    pub upperbound: UpperboundStrategy,

    #[arg(long, value_enum, default_value = "left-right")]
    pub terminal_solver: TerminalSolver,

    /// Determines if the solver should use an immediate lowerbound for a node (e.g. kmeans for regression).
    #[arg(long, value_enum, default_value = "lowerbound")]
    pub branch_relaxation: BranchRelaxation,

    /// Determines if the solver should track intermediate solutions.
    #[arg(long, action=ArgAction::Set, default_value_t=false)]
    pub intermediates: bool,

    /// The search strategy to use.
    #[arg(short, long, value_parser=clap_enum_variants!(SearchStrategyEnum), default_value_t=SearchStrategyEnum::BfsGosdt)]
    pub strategy: SearchStrategyEnum,

    /// The task to optimize.
    #[command(subcommand)]
    pub task: OptimizationTaskEnum,
}

#[derive(Subcommand)]
pub enum OptimizationTaskEnum {
    /// Optimizes classification accuracy
    Accuracy(AccuracyParams),
    /// Optimizes squared error
    SquaredError(SquaredErrorParams),
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
