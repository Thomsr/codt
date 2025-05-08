use std::cmp::Ordering;

use crate::tasks::OptimizationTask;

use super::node::{Node, QueueItem};

// Todo:
// add priority attribute, set the priority to the lowest of its two children. If no children, then lower bound.
// pub mod bfs;
// AndOr search, but limited to k-quantiles at a time. Essentially Quant-BnB but best-first.
// pub mod quant;
// ------
pub mod andor;
pub mod bfs;
pub mod dfs;
pub mod dfsprio;

pub trait SearchStrategy {
    /// The order in which the items in the queue should be handled. Items are handled in asceding order.
    fn cmp_item<'a, OT: OptimizationTask, SS: SearchStrategy>(
        a: &QueueItem<'a, OT, SS>,
        b: &QueueItem<'a, OT, SS>,
    ) -> Ordering;

    /// After backtracking, decide which of the child nodes should be explored next.
    fn child_priority<'a, OT: OptimizationTask, SS: SearchStrategy>(
        a: &Node<'a, OT, SS>,
        b: &Node<'a, OT, SS>,
    ) -> usize;

    /// In some search strategies, the front of the queue is the lowest possible
    /// lowest bound, and can be used as a lower bound for the node.
    fn item_front_of_queue_is_lowest_lb<OT: OptimizationTask, SS: SearchStrategy>(
        item: &QueueItem<OT, SS>,
    ) -> bool;

    fn heuristic_from_lb_and_support<OT: OptimizationTask>(lb: OT::CostType, support: usize)
    -> f64;
}
