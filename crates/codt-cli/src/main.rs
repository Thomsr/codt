use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use codt::{
    model::{dataset::DataSet, dataview::DataView},
    search::{
        solver::{Solver, SolverOptions, TerminalSolver, UpperboundStrategy},
        strategy::{
            SearchStrategy,
            andor::AndOrSearchStrategy,
            bfs::{BfsSearchStrategy, CuriosityHeuristic, GOSDTHeuristic, LBHeuristic},
            dfs::DfsSearchStrategy,
            dfsprio::DfsPrioSearchStrategy,
        },
    },
    tasks::{OptimizationTask, accuracy::AccuracyTask, squared_error::SquaredErrorTask},
};
use file_reader::read_from_file;
use log::info;
use params::{OptimizationTaskEnum, get_cli_args};

mod file_reader;
mod params;
mod value_parser;

fn run_solver_for_task<T: OptimizationTask, SS: SearchStrategy>(
    file: &PathBuf,
    options: SolverOptions,
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
    let result = solver.solve(options);

    let solve_time = before_solve.elapsed().as_secs_f64();

    info!("Read time (s): {}", read_time);
    info!("Solve time (s): {}", solve_time);
    info!("Graph expansions: {}", result.graph_expansions);
    info!("Tree: {}", result.tree);
    println!("{}", result.cost_str);
}

fn run_with_strategy<T: OptimizationTask>(
    file: &PathBuf,
    options: SolverOptions,
    task: T,
    strategy: params::SearchStrategy,
) {
    match strategy {
        params::SearchStrategy::Dfs => {
            run_solver_for_task::<_, DfsSearchStrategy>(file, options, task)
        }
        params::SearchStrategy::AndOr => {
            run_solver_for_task::<_, AndOrSearchStrategy>(file, options, task)
        }
        params::SearchStrategy::DfsPrio => {
            run_solver_for_task::<_, DfsPrioSearchStrategy>(file, options, task)
        }
        params::SearchStrategy::BfsLb => {
            run_solver_for_task::<_, BfsSearchStrategy<LBHeuristic>>(file, options, task)
        }
        params::SearchStrategy::BfsCuriosity => {
            run_solver_for_task::<_, BfsSearchStrategy<CuriosityHeuristic>>(file, options, task)
        }
        params::SearchStrategy::BfsGosdt => {
            run_solver_for_task::<_, BfsSearchStrategy<GOSDTHeuristic>>(file, options, task)
        }
    };
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
        ub_strategy: match args.upperbound {
            params::UpperboundStrategy::SolutionsOnly => UpperboundStrategy::SolutionsOnly,
            params::UpperboundStrategy::TightFromSibling => UpperboundStrategy::TightFromSibling,
            params::UpperboundStrategy::ForRemainingInterval => {
                UpperboundStrategy::ForRemainingInterval
            }
        },
        terminal_solver: match args.terminal_solver {
            params::TerminalSolver::Leaf => TerminalSolver::Leaf,
            params::TerminalSolver::LeftRight => TerminalSolver::LeftRight,
            params::TerminalSolver::D2 => TerminalSolver::D2,
        },
        timeout: args.timeout.map(Duration::from_secs),
        track_intermediates: args.intermediates,
    };

    match args.task {
        OptimizationTaskEnum::Accuracy(params) => {
            let task = AccuracyTask::new(params.complexity_cost);
            run_with_strategy(&args.file, options, task, args.strategy);
        }
        OptimizationTaskEnum::SquaredError(params) => {
            let task = SquaredErrorTask::new(params.complexity_cost);
            run_with_strategy(&args.file, options, task, args.strategy);
        }
    }
}
