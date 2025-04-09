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
        // Then by completeness, if the most promising node is also complete, then we are done.
        // Then by expanded, so we expand the least number of nodes possible.
        // Then by feature, so we focus on each feature individually. Bounds do not propagate between features.
        // Then by interval size, so we get a good spread for bounds.
        // Then by interval start, for a deterministic ordering.
        a.cost_lower_bound
            .partial_cmp(&b.cost_lower_bound)
            .unwrap_or(Ordering::Equal)
            .then(a.is_complete().cmp(&b.is_complete()))
            .then(a.is_expanded().cmp(&b.is_expanded()))
            .then(a.feature.cmp(&b.feature))
            .then(a.split_points.len().cmp(&b.split_points.len()))
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

    fn backtrack_item<'a, OT: OptimizationTask, SS: SearchStrategy>(
        node: &mut crate::search::node::Node<'a, OT, SS>,
        mut item: QueueItem<'a, OT, SS>,
    ) -> Option<QueueItem<'a, OT, SS>> {
        node.recalculate_item_lb(&mut item);
        let prev_lb = item.cost_lower_bound;

        // Only revisit this node if it is not yet fully explored.
        if !item.is_complete() {
            node.queue.push(item);
        }

        if let Some(next_in_line) = node.queue.peek() {
            let lowest_lb = next_in_line.cost_lower_bound;

            // Lower bound of the node is at least the minimum lower bound in the queue.
            if lowest_lb > node.cost_lower_bound {
                node.cost_lower_bound = lowest_lb;
            }

            // Early exit check
            if node.cost_lower_bound > node.cost_upper_bound {
                node.queue.clear();
                return None;
            }

            // If the front of the queue has a lesser lower bound than the
            // one we just reinserted, then we need to continue updating.
            if lowest_lb < prev_lb {
                node.queue.pop()
            } else {
                None
            }
        } else {
            None
        }
    }
}
