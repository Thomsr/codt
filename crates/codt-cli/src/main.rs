use std::{
    collections::HashSet,
    path::PathBuf,
    time::{Duration, Instant},
};

use codt::{
    model::{dataset::DataSet, dataview::DataView, reduction::DataReductionStrategy},
    search::solver::{
        DataReductionOption, SearchStrategyEnum, SolveStatus, SolverOptions, solver_with_strategy,
    },
    tasks::{OptimizationTask, accuracy::AccuracyTask},
};
use file_reader::read_from_file;
use log::info;
use params::get_cli_args;

mod file_reader;
mod params;
mod value_parser;

fn run_solver_for_task<T: OptimizationTask>(
    file: &PathBuf,
    options: SolverOptions,
    reduction_strategy: DataReductionStrategy,
    task: T,
    strategy: SearchStrategyEnum,
) where
    T::LabelType: PartialEq,
{
    info!("Starting solve with options:\n{:?}\n", options);

    let before_read = Instant::now();
    let mut dataset = DataSet::<T::InstanceType>::default();
    read_from_file(&mut dataset, file).unwrap();
    let read_time = before_read.elapsed().as_secs_f64();

    let before_solve = Instant::now();
    T::preprocess_dataset(&mut dataset);
    let full_view = DataView::from_dataset_with_reduction(&dataset, reduction_strategy);
    let mut solver = solver_with_strategy(task, full_view, strategy);
    let result = solver.solve(options);

    let solve_time = before_solve.elapsed().as_secs_f64();

    info!("Read time (s): {}", read_time);
    info!("Solve time (s): {}", solve_time);
    info!(
        "Max memory usage (MB): {:.2}",
        result.memory_usage_bytes as f64 / 1024.0 / 1024.0
    );
    info!("Graph expansions: {}", result.graph_expansions);
    match result.status {
        SolveStatus::PerfectTreeFound => {
            if let Some(tree) = result.tree {
                info!("Tree: {}", tree);
            }
            if let Some(cost) = result.cost_str {
                println!("{}", cost);
            }
        }
        SolveStatus::NoPerfectTree => {
            println!("No perfect tree exists for the given data and constraints.");
        }
    }
}

fn main() {
    let args = get_cli_args();

    let mut log_builer = env_logger::builder();
    if args.verbose {
        log_builer.filter_level(log::LevelFilter::max());
    } else {
        log_builer.filter_level(log::LevelFilter::Info);
    }
    log_builer.init();

    let options = SolverOptions {
        lb_strategy: args.lowerbound.into_iter().collect::<HashSet<_>>(),
        ub_strategy: args.upperbound,
        cart_ub_strategy: args.cart_upperbound,
        data_reduction: args.data_reduction,
        memory_limit: Some(args.memory_limit),
        timeout: Some(Duration::from_secs(args.timeout)),
        track_intermediates: args.intermediates,
    };

    let task = AccuracyTask::new();
    let reduction_strategy = match args.data_reduction {
        DataReductionOption::Disabled => DataReductionStrategy::Disabled,
        DataReductionOption::Enabled => DataReductionStrategy::Enabled,
    };
    run_solver_for_task(&args.file, options, reduction_strategy, task, args.strategy);
}
