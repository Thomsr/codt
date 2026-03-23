use std::{sync::Arc, time::Duration};

use strum_macros::{Display, EnumString, IntoStaticStr, VariantNames};

use crate::{
    model::{dataview::DataView, tree::Tree},
    search::{
        solvers::{
            optimal_solver::OptimalSolverImpl,
            perfect_solver::PerfectSolverImpl,
        },
        strategy::{
            andor::AndOrSearchStrategy,
            bfs::{
                BfsSearchStrategy, CuriosityHeuristic, LBSupportHeuristic,
                LeastDiscrepancyHeuristic, RandomHeuristic,
            },
            dfs::DfsSearchStrategy,
            dfsprio::DfsPrioSearchStrategy,
            random::RandomDfsSearchStrategy,
        },
    },
    tasks::OptimizationTask,
};

pub struct SolveResult<OT: OptimizationTask> {
    pub tree: Arc<Tree<OT>>,
    pub cost_str: String,
    pub graph_expansions: i32,
    pub memory_usage_bytes: i64,
    pub intermediate_lbs: Vec<(OT::CostType, i32, f64)>,
    pub intermediate_ubs: Vec<(OT::CostType, i32, f64)>,
}

pub trait Solver<OT: OptimizationTask> {
    fn solve(&mut self, options: SolverOptions) -> SolveResult<OT>;
}


macro_rules! solver_impl_for {
    ($task:ident, $dataview:ident, $strat:ty, $solver:ident) => {
        match $solver {
            SolverEnum::Optimal => Box::new(OptimalSolverImpl::<OT, $strat>::new($task, $dataview)),
            SolverEnum::Perfect => Box::new(PerfectSolverImpl::<OT, $strat>::new($task, $dataview)),
        }
    };
}

pub fn solver_with_strategy<'a, OT: OptimizationTask + 'a>(
    task: OT,
    dataview: DataView<'a, OT>,
    strategy: SearchStrategyEnum,
    solver: SolverEnum,
) -> Box<dyn Solver<OT> + 'a> {
    match strategy {
        SearchStrategyEnum::AndOr => {
            solver_impl_for!(task, dataview, AndOrSearchStrategy, solver)
        }
        SearchStrategyEnum::Dfs => {
            solver_impl_for!(task, dataview, DfsSearchStrategy, solver)
        }
        SearchStrategyEnum::DfsPrio => {
            solver_impl_for!(task, dataview, DfsPrioSearchStrategy, solver)
        }
        SearchStrategyEnum::DfsRandom => {
            solver_impl_for!(task, dataview, RandomDfsSearchStrategy, solver)
        }
        SearchStrategyEnum::BfsLb => {
            solver_impl_for!(task, dataview, BfsSearchStrategy<LBSupportHeuristic<1, 0>>, solver)
        }
        SearchStrategyEnum::BfsCuriosity => {
            solver_impl_for!(task, dataview, BfsSearchStrategy<CuriosityHeuristic>, solver)
        }
        SearchStrategyEnum::BfsLbTiebreakSmall => solver_impl_for!(
            task,
            dataview,
            BfsSearchStrategy<LBSupportHeuristic<100000, 1>>,
            solver
        ),
        SearchStrategyEnum::BfsLbTiebreakBig => solver_impl_for!(
            task,
            dataview,
            BfsSearchStrategy<LBSupportHeuristic<100000, -1>>,
            solver
        ),
        SearchStrategyEnum::BfsSmall => {
            solver_impl_for!(task, dataview, BfsSearchStrategy<LBSupportHeuristic<0, 1>>, solver)
        }
        SearchStrategyEnum::BfsBig => {
            solver_impl_for!(task, dataview, BfsSearchStrategy<LBSupportHeuristic<0, -1>>, solver)
        }
        SearchStrategyEnum::BfsSmallTiebreakLb => solver_impl_for!(
            task,
            dataview,
            BfsSearchStrategy<LBSupportHeuristic<1, 100000>>,
            solver
        ),
        SearchStrategyEnum::BfsBigTiebreakLb => solver_impl_for!(
            task,
            dataview,
            BfsSearchStrategy<LBSupportHeuristic<1, -100000>>,
            solver
        ),
        SearchStrategyEnum::BfsBalanceSmallLb => {
            solver_impl_for!(task, dataview, BfsSearchStrategy<LBSupportHeuristic<1, 1>>, solver)
        }
        SearchStrategyEnum::BfsBalanceBigLb => {
            solver_impl_for!(task, dataview, BfsSearchStrategy<LBSupportHeuristic<1, -1>>, solver)
        }
        SearchStrategyEnum::BfsRandom => {
            solver_impl_for!(task, dataview, BfsSearchStrategy<RandomHeuristic>, solver)
        }
        SearchStrategyEnum::BfsLds => {
            solver_impl_for!(task, dataview, BfsSearchStrategy<LeastDiscrepancyHeuristic>, solver)
        }
    }
}

/// The search strategy to use. This is used to select the appropriate search strategy at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString, VariantNames, IntoStaticStr, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum SearchStrategyEnum {
    /// Exhaustive exact solver without pruning bounds.
    AndOr,
    Dfs,
    DfsPrio,
    DfsRandom,
    BfsLb,
    BfsCuriosity,
    BfsLbTiebreakSmall,
    BfsLbTiebreakBig,
    BfsSmall,
    BfsBig,
    BfsSmallTiebreakLb,
    BfsBigTiebreakLb,
    BfsBalanceSmallLb,
    BfsBalanceBigLb, // This is the same heuristic as GOSDT
    BfsRandom,
    BfsLds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString, VariantNames, IntoStaticStr, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum SolverEnum {
    Optimal,
    Perfect
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString, VariantNames, IntoStaticStr, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum UpperboundStrategy {
    /// Only use actual solutions as upper bounds
    SolutionsOnly,
    /// Use bounds of parent and sibling to calculate an upper bound
    TightFromSibling,
    /// Similar to `TightFromSibling`, but also leave a margin so that when a solution is found the whole interval can be pruned.
    ForRemainingInterval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString, VariantNames, IntoStaticStr, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum TerminalSolver {
    /// No exhaustive search, search terminates at leaf nodes
    Leaf,
    /// Exhaustive search on subtrees of depth one
    D1,
    /// Start exhaustive search when only a left/right depth one tree remains
    LeftRight,
    /// Exhaustive search on subtrees of depth two
    D2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString, VariantNames, IntoStaticStr, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum BranchRelaxation {
    /// Do not use branch relaxation
    None,
    /// Use branch relaxation to get a lower bound on the cost of the remaining instances
    Lowerbound,
    /// Also check if the splits for branch relaxation can be made, so it is an exact cost
    Exact,
}

impl BranchRelaxation {
    pub fn should_compute(&self) -> bool {
        matches!(self, BranchRelaxation::Lowerbound | BranchRelaxation::Exact)
    }
}

#[derive(Debug)]
pub struct SolverOptions {
    pub ub_strategy: UpperboundStrategy,
    pub terminal_solver: TerminalSolver,
    pub branch_relaxation: BranchRelaxation,
    pub track_intermediates: bool,
    pub max_depth: u32,
    pub timeout: Option<Duration>,
    pub memory_limit: Option<u64>,
}
