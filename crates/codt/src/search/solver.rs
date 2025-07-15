use std::{sync::Arc, time::Duration};

use strum_macros::{Display, EnumString, IntoStaticStr, VariantNames};

use crate::{
    model::{dataview::DataView, tree::Tree},
    search::{
        solver_impl::SolverImpl,
        strategy::{
            self,
            andor::AndOrSearchStrategy,
            bfs::{BfsSearchStrategy, LBHeuristic},
            dfs::DfsSearchStrategy,
        },
    },
    tasks::OptimizationTask,
};

pub struct SolveResult<OT: OptimizationTask> {
    pub tree: Arc<Tree<OT>>,
    pub cost_str: String,
    pub graph_expansions: i32,
    pub intermediate_lbs: Vec<(OT::CostType, i32, f64)>,
    pub intermediate_ubs: Vec<(OT::CostType, i32, f64)>,
}

pub trait Solver<OT: OptimizationTask> {
    fn solve(&mut self, options: SolverOptions) -> SolveResult<OT>;
}

pub fn solver_with_strategy<'a, OT: OptimizationTask + 'a>(
    task: OT,
    dataview: DataView<'a, OT>,
    strategy: SearchStrategyEnum,
) -> Box<dyn Solver<OT> + 'a> {
    match strategy {
        SearchStrategyEnum::Dfs => {
            Box::new(SolverImpl::<OT, DfsSearchStrategy>::new(task, dataview))
        }
        SearchStrategyEnum::AndOr => {
            Box::new(SolverImpl::<OT, AndOrSearchStrategy>::new(task, dataview))
        }
        SearchStrategyEnum::BfsLb => Box::new(
            SolverImpl::<OT, BfsSearchStrategy<LBHeuristic>>::new(task, dataview),
        ),
        SearchStrategyEnum::BfsCuriosity => Box::new(SolverImpl::<
            OT,
            BfsSearchStrategy<strategy::bfs::CuriosityHeuristic>,
        >::new(task, dataview)),
        SearchStrategyEnum::BfsGosdt => Box::new(SolverImpl::<
            OT,
            BfsSearchStrategy<strategy::bfs::GOSDTHeuristic>,
        >::new(task, dataview)),
        SearchStrategyEnum::BfsRandom => Box::new(SolverImpl::<
            OT,
            BfsSearchStrategy<strategy::bfs::RandomHeuristic>,
        >::new(task, dataview)),
        SearchStrategyEnum::DfsPrio => {
            Box::new(SolverImpl::<OT, DfsSearchStrategy>::new(task, dataview))
        }
        SearchStrategyEnum::DfsRandom => {
            Box::new(SolverImpl::<OT, DfsSearchStrategy>::new(task, dataview))
        }
    }
}

/// The search strategy to use. This is used to select the appropriate search strategy at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString, VariantNames, IntoStaticStr, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum SearchStrategyEnum {
    Dfs,
    AndOr,
    BfsLb,
    BfsCuriosity,
    BfsGosdt,
    BfsRandom,
    DfsPrio,
    DfsRandom,
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

#[derive(Debug)]
pub struct SolverOptions {
    pub ub_strategy: UpperboundStrategy,
    pub terminal_solver: TerminalSolver,
    pub track_intermediates: bool,
    pub node_lowerbound: bool,
    pub max_depth: u32,
    pub timeout: Option<Duration>,
    pub memory_limit: Option<u64>,
}
