use std::cmp::Ordering;

use crate::{
    search::node::{Node, QueueItem},
    tasks::{CostSum, OptimizationTask},
};

use super::SearchStrategy;

pub struct DfsSearchStrategy;

impl SearchStrategy for DfsSearchStrategy {
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
        _item: &QueueItem<'a, OT, SS>,
        children: &[Node<'a, OT, SS>; 2],
    ) -> usize {
        // For DFS, only use information available when starting the search.
        // This is a proxy for the upper bound.
        if children[0].dataview.cost_summer.cost() >= children[1].dataview.cost_summer.cost() {
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
