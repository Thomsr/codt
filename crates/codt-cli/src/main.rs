use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use codt::{
    model::{dataset::DataSet, dataview::DataView},
    search::solver::{SearchStrategyEnum, SolverOptions, solver_with_strategy},
    tasks::{OptimizationTask, accuracy::AccuracyTask, squared_error::SquaredErrorTask},
};
use file_reader::read_from_file;
use log::info;
use params::{OptimizationTaskEnum, get_cli_args};

mod file_reader;
mod params;
mod value_parser;

fn run_solver_for_task<T: OptimizationTask>(
    file: &PathBuf,
    options: SolverOptions,
    task: T,
    strategy: SearchStrategyEnum,
) {
    let before_read = Instant::now();
    let mut dataset = DataSet::<T::InstanceType>::default();
    read_from_file(&mut dataset, file).unwrap();
    let read_time = before_read.elapsed().as_secs_f64();

    let before_solve = Instant::now();
    T::preprocess_dataset(&mut dataset);
    let full_view = DataView::from_dataset(&dataset);
    let mut solver = solver_with_strategy(task, full_view, strategy);
    let result = solver.solve(options);

    let solve_time = before_solve.elapsed().as_secs_f64();

    info!("Read time (s): {}", read_time);
    info!("Solve time (s): {}", solve_time);
    info!("Graph expansions: {}", result.graph_expansions);
    info!("Tree: {}", result.tree);
    println!("{}", result.cost_str);
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
        max_depth: args.max_depth,
        ub_strategy: args.upperbound,
        terminal_solver: args.terminal_solver,
        node_lowerbound: args.node_lowerbound,
        memory_limit: args.memory_limit,
        timeout: args.timeout.map(Duration::from_secs),
        track_intermediates: args.intermediates,
    };

    match args.task {
        OptimizationTaskEnum::Accuracy(params) => {
            let task = AccuracyTask::new(params.complexity_cost);
            run_solver_for_task(&args.file, options, task, args.strategy);
        }
        OptimizationTaskEnum::SquaredError(params) => {
            let task = SquaredErrorTask::new(params.complexity_cost);
            run_solver_for_task(&args.file, options, task, args.strategy);
        }
    }
}
