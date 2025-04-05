use std::path::PathBuf;

use clap::{ArgAction, Args, Parser, Subcommand, value_parser};

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
    #[arg(short, long, action=ArgAction::Set, default_value_t=true)]
    pub verbose: bool,

    /// Location of the (training) dataset.
    #[arg(short, long)]
    pub file: PathBuf,

    /// Maximum allowed depth of the tree, where the depth is defined as the largest number of *decision/feature nodes* from the root to any leaf. Depth greater than four is usually time consuming.
    #[arg(short('d'), long, default_value_t=3, value_parser=value_parser!(u32).range(0..20))]
    pub max_depth: u32,

    /// The task to optimize.
    #[command(subcommand)]
    pub task: OptimizationTaskEnum,
}

#[derive(Subcommand)]
pub enum OptimizationTaskEnum {
    /// Optimizes classification accuracy
    Accuracy,
    /// Optimizes squared error
    Regression(RegressionParams),
}

#[derive(Args)]
#[command(next_help_heading = "Task Parameters")]
pub struct RegressionParams {
    /// The cost for adding an extra node to the tree. 0.01 means one extra node is only jusitified if it results in at least one percent better training score than a constant predictor.
    #[arg(long, value_parser=RangedF64ValueParser::<f64>::new().range(0.0..=1.0), default_value_t=0.0)]
    pub complexity_penalty: f64,
}
