use std::cmp::Ordering;

use crate::{search::node::QueueItem, tasks::OptimizationTask};

use super::SearchStrategy;

pub struct AndOrSearchStrategy;

impl SearchStrategy for AndOrSearchStrategy {
    const FRONT_OF_QUEUE_IS_LOWEST_LB: bool = true;

    fn cmp_item<'a, OT: OptimizationTask, SS: SearchStrategy>(
        a: &QueueItem<'a, OT, SS>,
        b: &QueueItem<'a, OT, SS>,
    ) -> Ordering {
        // First ordered by the objective value, so more promising nodes are explored first.
        // Then by completeness, if the most promising node is also complete, then we are done.
        // Then by expanded, so we expand the least number of nodes possible.
        // Then by interval size, so we get a good spread for bounds.
        // Then by feature and interval start, for a deterministic ordering.
        a.cost_lower_bound
            .partial_cmp(&b.cost_lower_bound)
            .unwrap_or(Ordering::Equal)
            .then(a.is_complete().cmp(&b.is_complete()))
            .then(a.is_expanded().cmp(&b.is_expanded()))
            .then(a.split_points.len().cmp(&b.split_points.len()))
            .then(a.feature.cmp(&b.feature))
            .then(a.split_points.start.cmp(&b.split_points.start))
    }

    fn child_priority<'a, OT: OptimizationTask, SS: SearchStrategy>(
        a: &crate::search::node::Node<'a, OT, SS>,
        b: &crate::search::node::Node<'a, OT, SS>,
    ) -> usize {
        // We choose the path in the graph as the most promising
        // solution (lowest lower bound), but when choosing which
        // 'and' node to expand, we choose the node most likely to
        // change the estimate (highest lower bound).
        if a.cost_lower_bound >= b.cost_lower_bound {
            0
        } else {
            1
        }
    }
}
