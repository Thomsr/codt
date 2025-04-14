use std::{path::PathBuf, time::Instant};

use codt::{
    model::{dataset::DataSet, dataview::DataView},
    search::{
        solver::Solver,
        strategy::{SearchStrategy, andor::AndOrSearchStrategy, dfs::DfsSearchStrategy},
    },
    tasks::{OptimizationTask, accuracy::AccuracyTask, regression::RegressionTask},
};
use file_reader::read_from_file;
use log::info;
use params::{OptimizationTaskEnum, get_cli_args};

mod file_reader;
mod params;
mod value_parser;

fn run_solver_for_task<T: OptimizationTask, SS: SearchStrategy>(
    file: &PathBuf,
    max_depth: u32,
    task: T,
) {
    let before_read = Instant::now();
    let mut dataset = DataSet::<T::InstanceType>::default();
    read_from_file(&mut dataset, file).unwrap();
    let read_time = before_read.elapsed().as_secs_f64();

    let before_solve = Instant::now();
    T::preprocess_dataset(&mut dataset);
    let full_view = DataView::from_dataset(&dataset);
    let mut solver: Solver<'_, T, SS> = Solver::new(task, full_view);
    let result = solver.solve(max_depth);

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

    match args.task {
        OptimizationTaskEnum::Accuracy => {
            let task = AccuracyTask::default();
            match args.strategy {
                params::SearchStrategy::Dfs => {
                    run_solver_for_task::<_, DfsSearchStrategy>(&args.file, args.max_depth, task)
                }
                params::SearchStrategy::AndOr => {
                    run_solver_for_task::<_, AndOrSearchStrategy>(&args.file, args.max_depth, task)
                }
            };
        }
        OptimizationTaskEnum::Regression(params) => {
            let _ = params;
            let task = RegressionTask::default();
            match args.strategy {
                params::SearchStrategy::Dfs => {
                    run_solver_for_task::<_, DfsSearchStrategy>(&args.file, args.max_depth, task)
                }
                params::SearchStrategy::AndOr => {
                    run_solver_for_task::<_, AndOrSearchStrategy>(&args.file, args.max_depth, task)
                }
            };
        }
    }
}
