use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    ops::{Bound, Range},
};

use rustc_hash::FxHashMap;

use crate::{model::dataview::DataView, tasks::OptimizationTask};

pub struct QueueItem<'a, OT: OptimizationTask> {
    pub cost_lower_bound: OT::CostType,

    /// The branching test for this node is one of `feature <= s` where s in split_points.
    pub feature: usize,
    /// The branching test for this node is one of `feature <= s` where s in split_points.
    pub split_points: Range<i32>,

    // Child nodes can only be initiated once the size of the `split_points` range is one.
    pub left_child: Option<Node<'a, OT>>,
    pub right_child: Option<Node<'a, OT>>,
}

impl<OT: OptimizationTask> PartialEq for QueueItem<'_, OT> {
    fn eq(&self, other: &Self) -> bool {
        self.cost_lower_bound == other.cost_lower_bound && self.feature == other.feature
    }
}

impl<OT: OptimizationTask> Eq for QueueItem<'_, OT> {}

impl<OT: OptimizationTask> PartialOrd for QueueItem<'_, OT> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<OT: OptimizationTask> Ord for QueueItem<'_, OT> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost_lower_bound
            .partial_cmp(&other.cost_lower_bound)
            .unwrap_or(Ordering::Equal)
            .then(self.feature.cmp(&other.feature))
    }
}

/// Represents a search node for a concrete feature test.
pub struct Node<'a, OT: OptimizationTask> {
    pub cost_lower_bound: Bound<OT::CostType>,
    pub cost_upper_bound: Bound<OT::CostType>,

    pub dataview: DataView<'a, OT::InstanceType>,

    pub queue: BinaryHeap<QueueItem<'a, OT>>,

    /// Best upper bound of feature test, split point, and cost.
    pub best: (i32, i32, OT::CostType),

    /// For each feature, the split points for which there is an upper and lower bound known for the left subtree.
    pub tried_solutions_l: Vec<FxHashMap<i32, (i32, i32)>>,
    /// For each feature, the split points for which there is an upper and lower bound known for the right subtree.
    pub tried_solutions_r: Vec<FxHashMap<i32, (i32, i32)>>,
}

impl<'a, OT: OptimizationTask> Node<'a, OT> {
    pub fn new(task: &OT, dataview: DataView<'a, OT::InstanceType>) -> Self {
        let ub = task.leaf_cost(&dataview);
        Node {
            cost_lower_bound: Bound::Included(OT::MIN_COST),
            cost_upper_bound: Bound::Included(ub),
            dataview,
            queue: BinaryHeap::new(),
            best: (-1, -1, ub),
            tried_solutions_l: Vec::new(),
            tried_solutions_r: Vec::new(),
        }
    }

    pub fn split(&self, task: &OT, feature: usize, split: i32) -> (Self, Self) {
        let (left_view, right_view) = self.dataview.split(feature, split);

        let ub_left = task.leaf_cost(&left_view);
        let ub_right = task.leaf_cost(&right_view);

        let left_node = Node {
            cost_lower_bound: Bound::Included(OT::MIN_COST),
            cost_upper_bound: Bound::Included(ub_left),
            dataview: left_view,
            queue: BinaryHeap::new(),
            best: (-1, -1, ub_left),
            tried_solutions_l: Vec::new(),
            tried_solutions_r: Vec::new(),
        };

        let right_node = Node {
            cost_lower_bound: Bound::Included(OT::MIN_COST),
            cost_upper_bound: Bound::Included(ub_right),
            dataview: right_view,
            queue: BinaryHeap::new(),
            best: (-1, -1, ub_right),
            tried_solutions_l: Vec::new(),
            tried_solutions_r: Vec::new(),
        };

        (left_node, right_node)
    }
}
