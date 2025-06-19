use std::cmp::Ordering;

use crate::{
    search::{
        node::{Node, QueueItem},
        strategy::SearchStrategy,
    },
    tasks::{Cost, CostSum, OptimizationTask},
};

pub struct DfsSearchStrategy;

impl SearchStrategy for DfsSearchStrategy {
    fn cmp_item<'a, OT: OptimizationTask, SS: SearchStrategy>(
        a: &QueueItem<'a, OT, SS>,
        b: &QueueItem<'a, OT, SS>,
    ) -> Ordering {
        // For DFS, we want a consistent ordering that cannot change dynamically. The selected
        // search node should keep being selected until it is complete.
        //
        // First ordered by expanded, so we always continue with previously selected nodes.
        // Then by feature, to prefer features with a better heuristic.
        // Then by the size of the interval, so we get a good spread for bounds.
        // Then by the unique start of the interval, to break ties deterministically.
        a.is_expanded()
            .cmp(&b.is_expanded())
            .reverse()
            .then(a.feature_rank.cmp(&b.feature_rank))
            .then(a.split_points.len().cmp(&b.split_points.len()).reverse())
            .then(a.split_points.start.cmp(&b.split_points.start))
    }

    fn child_priority<'a, OT: OptimizationTask, SS: SearchStrategy>(
        _item: &QueueItem<'a, OT, SS>,
        children: &[Node<'a, OT, SS>; 2],
    ) -> usize {
        // For DFS, only use information available when starting the search.
        // This is a proxy for the upper bound.
        if children[0]
            .dataview
            .cost_summer
            .cost()
            .greater_or_not_much_less_than(&children[1].dataview.cost_summer.cost())
        {
            0
        } else {
            1
        }
    }

    fn item_front_of_queue_is_lowest_lb<OT: OptimizationTask, SS: SearchStrategy>(
        _item: &QueueItem<OT, SS>,
    ) -> bool {
        false
    }
}
