use std::cmp::Ordering;

use crate::{
    search::node::{Node, QueueItem},
    tasks::{Cost, OptimizationTask},
};

use super::SearchStrategy;

pub struct DfsPrioSearchStrategy;

impl SearchStrategy for DfsPrioSearchStrategy {
    fn cmp_item<'a, OT: OptimizationTask, SS: SearchStrategy>(
        a: &QueueItem<'a, OT, SS>,
        b: &QueueItem<'a, OT, SS>,
    ) -> Ordering {
        // First ordered by expanded, so we expand the least number of nodes possible.
        // Then by the objective value, so more promising nodes are explored first.
        // Then by interval size, so we get a good spread for bounds.
        // Then by feature and interval start, for a deterministic ordering.
        a.is_expanded()
            .cmp(&b.is_expanded())
            .reverse()
            .then(
                a.cost_lower_bound
                    .to_order()
                    .cmp(&b.cost_lower_bound.to_order()),
            )
            .then(a.split_points.len().cmp(&b.split_points.len())) // TODO reverse?
            .then(a.feature.cmp(&b.feature))
            .then(a.split_points.start.cmp(&b.split_points.start))
    }

    fn child_priority<'a, OT: OptimizationTask, SS: SearchStrategy>(
        _item: &QueueItem<'a, OT, SS>,
        children: &[Node<'a, OT, SS>; 2],
    ) -> usize {
        // We choose the path in the graph as the most promising
        // solution (lowest lower bound), but when choosing which
        // 'and' node to expand, we choose the node most likely to
        // change the estimate (highest upper bound).
        if children[0]
            .cost_upper_bound
            .greater_or_not_much_less_than(&children[1].cost_upper_bound)
        {
            0
        } else {
            1
        }
    }

    fn item_front_of_queue_is_lowest_lb<OT: OptimizationTask, SS: SearchStrategy>(
        item: &QueueItem<OT, SS>,
    ) -> bool {
        // If the item front of queue is not expanded, then there are no other
        // expanded in queue, and the next ordering is by lower bound.
        !item.is_expanded()
    }
}
