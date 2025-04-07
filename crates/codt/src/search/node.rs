use std::{cmp::Ordering, collections::BinaryHeap, fmt::Debug, ops::Range};

use log::info;
use rustc_hash::FxHashMap;

use crate::{model::dataview::DataView, tasks::OptimizationTask};

pub struct QueueItem<'a, OT: OptimizationTask> {
    pub cost_lower_bound: OT::CostType,

    /// The branching test for this node is one of `feature <= s` where s in split_points.
    pub feature: usize,
    /// The branching test for this node is one of `feature <= s` where s in split_points.
    pub split_points: Range<i32>,

    // Child nodes can only be initiated once the size of the `split_points` range is one.
    pub children: Option<[Node<'a, OT>; 2]>,

    /// The index of the child currently under consideration. E.g. for best first search, the child with the lowest lower bound.
    pub current_child: usize,
}

impl<OT: OptimizationTask> Debug for QueueItem<'_, OT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueueItem")
            .field("cost_lower_bound", &self.cost_lower_bound)
            .field("feature", &self.feature)
            .field("split_points", &self.split_points)
            .field("children", &self.children.is_some())
            .field("current_child", &self.current_child)
            .finish()
    }
}

impl<OT: OptimizationTask> PartialEq for QueueItem<'_, OT> {
    fn eq(&self, other: &Self) -> bool {
        self.cost_lower_bound == other.cost_lower_bound
            && self.feature == other.feature
            && self.split_points.start == other.split_points.start
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
            .then(self.is_complete().cmp(&other.is_complete()))
            .then(self.is_expanded().cmp(&other.is_expanded()))
            .then(self.feature.cmp(&other.feature))
            .then(self.split_points.start.cmp(&other.split_points.start))
            .reverse() // BinaryHeap is a max-heap. But we want the minumum, so reverse.
    }
}

impl<'a, OT: OptimizationTask> QueueItem<'a, OT> {
    fn new(feature: usize, split_points: Range<i32>) -> Self {
        Self {
            cost_lower_bound: OT::MIN_COST,
            feature,
            split_points,
            children: None,
            current_child: 0,
        }
    }

    pub fn split_at(&mut self, split: i32) -> (Option<Self>, Option<Self>) {
        let mut left = None;
        let mut right = None;
        if split - self.split_points.start > 0 {
            left = Some(Self {
                cost_lower_bound: self.cost_lower_bound,
                feature: self.feature,
                split_points: self.split_points.start..split,
                children: None,
                current_child: 0,
            })
        }
        if self.split_points.end - 1 - split > 0 {
            right = Some(Self {
                cost_lower_bound: self.cost_lower_bound,
                feature: self.feature,
                split_points: (split + 1)..self.split_points.end,
                children: None,
                current_child: 0,
            })
        }
        self.split_points = split..(split + 1);

        (left, right)
    }

    pub fn is_expanded(&self) -> bool {
        self.children.is_some()
    }

    pub fn is_complete(&self) -> bool {
        if let Some(children) = &self.children {
            children[0].is_complete() && children[1].is_complete()
        } else {
            false
        }
    }

    pub fn get_upper_and_update_lower_bound_from_children(&mut self) -> Option<OT::CostType> {
        if let Some(children) = &self.children {
            let lb = children[0].cost_lower_bound + children[1].cost_lower_bound;
            if self.cost_lower_bound < lb {
                self.cost_lower_bound = lb
            }

            if children[0].cost_lower_bound <= children[1].cost_lower_bound {
                self.current_child = 0;
            } else {
                self.current_child = 1;
            }

            if children[self.current_child].is_complete() {
                self.current_child = 1 - self.current_child;
            }

            Some(children[0].cost_upper_bound + children[1].cost_upper_bound)
        } else {
            None
        }
    }

    pub fn current_node(&self) -> Option<&Node<'a, OT>> {
        self.children.as_ref().map(|c| &c[self.current_child])
    }

    pub fn current_node_mut(&mut self) -> Option<&mut Node<'a, OT>> {
        self.children.as_mut().map(|c| &mut c[self.current_child])
    }
}

/// Represents a search node for a concrete feature test.
#[derive(Debug)]
pub struct Node<'a, OT: OptimizationTask> {
    pub cost_lower_bound: OT::CostType,
    pub cost_upper_bound: OT::CostType,

    pub remaining_depth_budget: u32,

    pub dataview: DataView<'a, OT::InstanceType>,

    pub queue: BinaryHeap<QueueItem<'a, OT>>,

    /// Best upper bound of feature test, split point, and cost.
    pub best: (i32, Range<i32>, OT::CostType),

    /// For each feature, the split points for which there is an upper and lower bound known for the left subtree.
    pub tried_solutions_l: Vec<FxHashMap<i32, (i32, i32)>>,
    /// For each feature, the split points for which there is an upper and lower bound known for the right subtree.
    pub tried_solutions_r: Vec<FxHashMap<i32, (i32, i32)>>,
}

impl<'a, OT: OptimizationTask> Node<'a, OT> {
    pub fn new(task: &OT, dataview: DataView<'a, OT::InstanceType>, max_depth: u32) -> Self {
        let ub = task.leaf_cost(&dataview);
        let lb = if max_depth == 0 { ub } else { OT::MIN_COST };

        let mut queue = BinaryHeap::new();

        if max_depth > 0 {
            for (feature, values) in dataview.feature_values_sorted.iter().enumerate() {
                match (values.first(), values.last()) {
                    (Some(x), Some(y)) => {
                        // For each feature, consider all useful splitting points. Note: last feature value is
                        // not a useful splitting point because all instances would go to the left. The range
                        // excludes the endpoint.
                        assert!(x.feature_value <= y.feature_value);
                        if x.feature_value != y.feature_value {
                            queue.push(QueueItem::new(feature, x.feature_value..y.feature_value));
                        }
                    }
                    // No branching decisions for this feature, nothing to add to the queue.
                    _ => {}
                }
            }
        }

        Node {
            cost_lower_bound: lb,
            cost_upper_bound: ub,
            remaining_depth_budget: max_depth,
            dataview,
            queue,
            best: (-1, -1..0, ub),
            tried_solutions_l: Vec::new(),
            tried_solutions_r: Vec::new(),
        }
    }

    pub fn is_complete(&self) -> bool {
        if let Some(item) = self.queue.peek() {
            item.cost_lower_bound == self.cost_upper_bound
        } else {
            true
        }
    }

    pub fn split(&self, task: &OT, feature: usize, split: i32) -> (Self, Self) {
        let (left_view, right_view) = self.dataview.split(feature, split);

        let left = Self::new(task, left_view, self.remaining_depth_budget - 1);
        let right = Self::new(task, right_view, self.remaining_depth_budget - 1);

        (left, right)
    }

    pub fn backtrack_item(&mut self, mut item: QueueItem<'a, OT>) {
        if let Some(ub) = item.get_upper_and_update_lower_bound_from_children() {
            info!("New UB: {:?}", ub);
            if self.cost_upper_bound > ub {
                self.cost_upper_bound = ub;
                self.best = (item.feature as i32, item.split_points.clone(), ub)
            }
        }

        if !item.is_complete() {
            self.queue.push(item);
        } else if self.queue.is_empty() {
            self.cost_lower_bound = self.cost_upper_bound;
        }
    }
}
