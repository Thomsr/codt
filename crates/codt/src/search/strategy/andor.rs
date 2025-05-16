use std::cmp::Ordering;

use crate::{search::node::QueueItem, tasks::OptimizationTask};

use super::SearchStrategy;

pub struct AndOrSearchStrategy;

impl SearchStrategy for AndOrSearchStrategy {
    fn cmp_item<'a, OT: OptimizationTask, SS: SearchStrategy>(
        a: &QueueItem<'a, OT, SS>,
        b: &QueueItem<'a, OT, SS>,
    ) -> Ordering {
        // First ordered by the objective value, so more promising nodes are explored first.
        // Then by expanded, so we expand the least number of nodes possible.
        // Then by interval size, so we get a good spread for bounds.
        // Then by feature and interval start, for a deterministic ordering.
        a.cost_lower_bound
            .partial_cmp(&b.cost_lower_bound)
            .unwrap_or(Ordering::Equal)
            .then(a.is_expanded().cmp(&b.is_expanded()).reverse())
            .then(a.split_points.len().cmp(&b.split_points.len())) // TODO: reverse?
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
        // change the estimate (highest upper bound).
        if a.cost_upper_bound >= b.cost_upper_bound {
            0
        } else {
            1
        }
    }

    fn item_front_of_queue_is_lowest_lb<OT: OptimizationTask, SS: SearchStrategy>(
        _item: &QueueItem<OT, SS>,
    ) -> bool {
        true
    }

    fn heuristic_from_lb_and_support<OT: OptimizationTask>(
        _lb: OT::CostType,
        _support: usize,
    ) -> f64 {
        0.0 // Not used
    }
}
