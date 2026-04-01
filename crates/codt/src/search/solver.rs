use std::{sync::Arc, time::Duration};

use strum_macros::{Display, EnumString, IntoStaticStr, VariantNames};

use crate::{
    model::{dataview::DataView, tree::Tree},
    search::{
        solver_impl::SolverImpl,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    PerfectTreeFound,
    NoPerfectTree,
}

pub struct SolveResult<OT: OptimizationTask> {
    pub status: SolveStatus,
    pub tree: Option<Arc<Tree<OT>>>,
    pub cost_str: Option<String>,
    pub graph_expansions: i32,
    pub memory_usage_bytes: i64,
    pub intermediate_lbs: Vec<(OT::CostType, i32, f64)>,
    pub intermediate_ubs: Vec<(OT::CostType, i32, f64)>,
}

pub trait Solver<OT: OptimizationTask> {
    fn solve(&mut self, options: SolverOptions) -> SolveResult<OT>;
    fn d0d1_lowerbound(&mut self) -> (OT::CostType, OT::CostType);
}

macro_rules! solver_impl_for {
    ($task:ident, $dataview:ident, $strat:ty) => {
        Box::new(SolverImpl::<OT, $strat>::new($task, $dataview))
    };
}

pub fn solver_with_strategy<'a, OT: OptimizationTask + 'a>(
    task: OT,
    dataview: DataView<'a, OT>,
    strategy: SearchStrategyEnum,
) -> Box<dyn Solver<OT> + 'a> {
    match strategy {
        SearchStrategyEnum::AndOr => {
            solver_impl_for!(task, dataview, AndOrSearchStrategy)
        }
        SearchStrategyEnum::Dfs => {
            solver_impl_for!(task, dataview, DfsSearchStrategy)
        }
        SearchStrategyEnum::DfsPrio => {
            solver_impl_for!(task, dataview, DfsPrioSearchStrategy)
        }
        SearchStrategyEnum::DfsRandom => {
            solver_impl_for!(task, dataview, RandomDfsSearchStrategy)
        }
        SearchStrategyEnum::BfsLb => {
            solver_impl_for!(task, dataview, BfsSearchStrategy<LBSupportHeuristic<1, 0>>)
        }
        SearchStrategyEnum::BfsCuriosity => {
            solver_impl_for!(task, dataview, BfsSearchStrategy<CuriosityHeuristic>)
        }
        SearchStrategyEnum::BfsLbTiebreakSmall => solver_impl_for!(
            task,
            dataview,
            BfsSearchStrategy<LBSupportHeuristic<100000, 1>>
        ),
        SearchStrategyEnum::BfsLbTiebreakBig => solver_impl_for!(
            task,
            dataview,
            BfsSearchStrategy<LBSupportHeuristic<100000, -1>>
        ),
        SearchStrategyEnum::BfsSmall => {
            solver_impl_for!(task, dataview, BfsSearchStrategy<LBSupportHeuristic<0, 1>>)
        }
        SearchStrategyEnum::BfsBig => {
            solver_impl_for!(task, dataview, BfsSearchStrategy<LBSupportHeuristic<0, -1>>)
        }
        SearchStrategyEnum::BfsSmallTiebreakLb => solver_impl_for!(
            task,
            dataview,
            BfsSearchStrategy<LBSupportHeuristic<1, 100000>>
        ),
        SearchStrategyEnum::BfsBigTiebreakLb => solver_impl_for!(
            task,
            dataview,
            BfsSearchStrategy<LBSupportHeuristic<1, -100000>>
        ),
        SearchStrategyEnum::BfsBalanceSmallLb => {
            solver_impl_for!(task, dataview, BfsSearchStrategy<LBSupportHeuristic<1, 1>>)
        }
        SearchStrategyEnum::BfsBalanceBigLb => {
            solver_impl_for!(task, dataview, BfsSearchStrategy<LBSupportHeuristic<1, -1>>)
        }
        SearchStrategyEnum::BfsRandom => {
            solver_impl_for!(task, dataview, BfsSearchStrategy<RandomHeuristic>)
        }
        SearchStrategyEnum::BfsLds => {
            solver_impl_for!(task, dataview, BfsSearchStrategy<LeastDiscrepancyHeuristic>)
        }
    }
}

/// The search strategy to use. This is used to select the appropriate search strategy at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString, VariantNames, IntoStaticStr, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum SearchStrategyEnum {
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
pub enum UpperboundStrategy {
    /// Only use actual solutions as upper bounds
    SolutionsOnly,
    /// Use bounds of parent and sibling to calculate an upper bound
    TightFromSibling,
    /// Similar to `TightFromSibling`, but also leave a margin so that when a solution is found the whole interval can be pruned.
    ForRemainingInterval,
    /// Run the CART algorithm on the remaining dataview to get an upper bound on the size needed.
    Cart 
}

#[derive(Debug)]
pub struct SolverOptions {
    pub ub_strategy: UpperboundStrategy,
    pub track_intermediates: bool,
    pub timeout: Option<Duration>,
    pub memory_limit: Option<u64>,
}
