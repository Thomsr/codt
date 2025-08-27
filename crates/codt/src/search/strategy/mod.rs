use std::cmp::Ordering;

use crate::{
    search::node::{FeatureTest, Node},
    tasks::OptimizationTask,
};

// Expand each interval at the best heuristic in k quantiles.
// pub mod quant;
pub mod andor;
pub mod bfs;
pub mod dfs;
pub mod dfsprio;
pub mod random;

pub trait SearchStrategy {
    /// The order in which the items in the queue should be handled. Items are handled in ascending order.
    fn cmp_item<'a, OT: OptimizationTask, SS: SearchStrategy>(
        a: &FeatureTest<'a, OT, SS>,
        b: &FeatureTest<'a, OT, SS>,
    ) -> Ordering;

    /// After backtracking, decide which of the child nodes should be explored next.
    fn child_priority<'a, OT: OptimizationTask, SS: SearchStrategy>(
        item: &FeatureTest<'a, OT, SS>,
        children: &[Node<'a, OT, SS>; 2],
    ) -> usize;

    /// In some search strategies, the front of the queue is the lowest possible
    /// lowest bound, and can be used as a lower bound for the node.
    fn item_front_of_queue_is_lowest_lb<OT: OptimizationTask, SS: SearchStrategy>(
        item: &FeatureTest<OT, SS>,
    ) -> bool;

    fn heuristic<OT: OptimizationTask, SS: SearchStrategy>(_item: &FeatureTest<OT, SS>) -> f64 {
        0.0 // Only used for global best first search strategy
    }

    fn generate_random_value() -> bool {
        false // Only used for random search strategies. Generation of random values introduces small slowdown, so make it conditional.
    }

    fn should_greedily_split() -> bool {
        false // Only used for least discrepancy search, where we want to do the first spit at the heuristically best point.
    }
}
