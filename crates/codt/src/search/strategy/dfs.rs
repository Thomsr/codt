use std::cmp::Ordering;

use crate::{search::node::QueueItem, tasks::OptimizationTask};

use super::SearchStrategy;

pub struct DfsSearchStrategy;

impl SearchStrategy for DfsSearchStrategy {
    const FRONT_OF_QUEUE_IS_LOWEST_LB: bool = false;

    fn cmp_item<'a, OT: OptimizationTask, SS: SearchStrategy>(
        a: &QueueItem<'a, OT, SS>,
        b: &QueueItem<'a, OT, SS>,
    ) -> Ordering {
        // For DFS, we want a consistent ordering that cannot change dynamically. The selected
        // search node should keep being selected until it is complete.
        a.feature
            .cmp(&b.feature)
            // Crucially, order by length of the range. This means any expanded
            // node (which has a range of a single value) goes first.
            .then(a.split_points.len().cmp(&b.split_points.len()))
            // Break ties deterministically by the unique start of the interval.
            .then(a.split_points.start.cmp(&b.split_points.start))
    }

    fn child_priority<'a, OT: OptimizationTask, SS: SearchStrategy>(
        _a: &crate::search::node::Node<'a, OT, SS>,
        _b: &crate::search::node::Node<'a, OT, SS>,
    ) -> usize {
        0
    }
}
