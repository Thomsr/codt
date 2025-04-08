use std::cmp::Ordering;

use crate::tasks::OptimizationTask;

use super::node::{Node, QueueItem};

pub mod andor;
pub mod dfs;

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

    /// Reinsert an updated item into the queue of a node. Optionally, return a queue item that should then also be reinserted.
    fn backtrack_item<'a, OT: OptimizationTask, SS: SearchStrategy>(
        node: &mut Node<'a, OT, SS>,
        item: QueueItem<'a, OT, SS>,
    ) -> Option<QueueItem<'a, OT, SS>>;
}
