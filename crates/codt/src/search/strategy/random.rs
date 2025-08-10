use std::cmp::Ordering;

use crate::{
    search::{
        node::{FeatureTest, Node},
        strategy::{SearchStrategy, dfs::DfsSearchStrategy},
    },
    tasks::OptimizationTask,
};

pub struct RandomDfsSearchStrategy;

impl SearchStrategy for RandomDfsSearchStrategy {
    fn cmp_item<'a, OT: OptimizationTask, SS: SearchStrategy>(
        a: &FeatureTest<'a, OT, SS>,
        b: &FeatureTest<'a, OT, SS>,
    ) -> Ordering {
        // For dfs, use random ordering, but stick with the same node once picked.
        a.random_value
            .cmp(&b.random_value)
            // Crucially, order by length of the range. This means any expanded
            // node (which has a range of a single value) goes first.
            .then(DfsSearchStrategy::cmp_item(a, b))
    }

    fn child_priority<'a, OT: OptimizationTask, SS: SearchStrategy>(
        item: &FeatureTest<'a, OT, SS>,
        _children: &[Node<'a, OT, SS>; 2],
    ) -> usize {
        // Choose the left or right branch randomly
        (item.random_value & 1).try_into().unwrap()
    }

    fn item_front_of_queue_is_lowest_lb<OT: OptimizationTask, SS: SearchStrategy>(
        _item: &FeatureTest<OT, SS>,
    ) -> bool {
        false
    }
}
