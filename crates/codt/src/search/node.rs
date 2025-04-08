use std::{
    cmp::Ordering,
    collections::{BTreeMap, BinaryHeap},
    fmt::Debug,
    ops::{Bound, Range},
};

use log::info;

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
        // While the Ord checks many more attributes, there should never be an item in
        // the queue for the same feature and an overlapping interval.
        self.feature == other.feature && self.split_points.start == other.split_points.start
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
        // First ordered by the objective value, so more promising nodes are explored first.
        // Then by completeness, if the most promising node is also complete, then we are done.
        // Then by expanded, so we expand the least number of nodes possible.
        // Then by feature, so we focus on each feature individually. Bounds do not propagate between features.
        // Then by interval size, so we get a good spread for bounds.
        // Then by interval start, for a deterministic ordering.
        self.cost_lower_bound
            .partial_cmp(&other.cost_lower_bound)
            .unwrap_or(Ordering::Equal)
            .then(self.is_complete().cmp(&other.is_complete()))
            .then(self.is_expanded().cmp(&other.is_expanded()))
            .then(self.feature.cmp(&other.feature))
            .then(self.split_points.len().cmp(&other.split_points.len()))
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

    /// Lookup for finding lower bounds. Note: sorted increasing in threshold
    /// and increasing in cost.
    /// Presence of tuple (threshold, cost) means that from 0 to
    /// threshold inclusive, cost is the highest lower bound for
    /// the left subtree found so far.
    pub best_left_subtree_left_of: Vec<BTreeMap<i32, OT::CostType>>,

    /// Lookup for finding lower bounds. Note: sorted increasing in threshold
    /// and decreasing in cost.
    /// Presence of tuple (threshold, cost) means that from threshold to
    /// inf inclusive, cost is the highest lower bound for
    /// the right subtree found so far.
    pub best_right_subtree_right_of: Vec<BTreeMap<i32, OT::CostType>>,
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
            best_left_subtree_left_of: vec![BTreeMap::new(); dataview.num_features()],
            best_right_subtree_right_of: vec![BTreeMap::new(); dataview.num_features()],
            dataview,
            queue,
            best: (-1, -1..0, ub),
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

    fn insert_left_subtree(&mut self, feature: usize, threshold: i32, lb: OT::CostType) {
        // Find point to insert
        let mut cursor =
            self.best_left_subtree_left_of[feature].upper_bound_mut(Bound::Included(&threshold));

        // Update or insert the new lower bound.
        let needs_insert = match cursor.peek_prev() {
            Some((&k, v)) => {
                if *v >= lb {
                    return;
                } else if k == threshold {
                    *v = lb;
                    false
                } else {
                    true
                }
            }
            None => true,
        };

        if needs_insert {
            cursor
                .insert_before(threshold, lb)
                .expect("Order should have been preserved by inserting at the correct index");
        }

        // Remove all thresholds after this that have a worse or equal lower bound.
        while let Some((_, v)) = cursor.next() {
            if *v <= lb {
                cursor.remove_prev();
            } else {
                // If we see a better lower bound, all subsequent ones must also be better.
                break;
            }
        }
    }

    fn insert_right_subtree(&mut self, feature: usize, threshold: i32, lb: OT::CostType) {
        // Find point to insert
        let mut cursor =
            self.best_right_subtree_right_of[feature].lower_bound_mut(Bound::Included(&threshold));

        // Update or insert the new lower bound.
        let needs_insert = match cursor.peek_next() {
            Some((&k, v)) => {
                if *v >= lb {
                    return;
                } else if k == threshold {
                    *v = lb;
                    false
                } else {
                    true
                }
            }
            None => true,
        };

        if needs_insert {
            cursor
                .insert_after(threshold, lb)
                .expect("Order should have been preserved by inserting at the correct index");
        }

        // Remove all thresholds after this that have a worse or equal lower bound.
        while let Some((_, v)) = cursor.prev() {
            if *v <= lb {
                cursor.remove_next();
            } else {
                // If we see a better lower bound, all subsequent ones must also be better.
                break;
            }
        }
    }

    fn get_upper_and_update_lower_bound_from_children(
        &mut self,
        item: &mut QueueItem<'a, OT>,
    ) -> Option<OT::CostType> {
        if let Some(children) = &item.children {
            self.insert_left_subtree(
                item.feature,
                item.split_points.start,
                children[0].cost_lower_bound,
            );
            self.insert_right_subtree(
                item.feature,
                item.split_points.end,
                children[1].cost_lower_bound,
            );
            let lb = children[0].cost_lower_bound + children[1].cost_lower_bound;
            if item.cost_lower_bound < lb {
                item.cost_lower_bound = lb
            }

            if children[0].cost_lower_bound <= children[1].cost_lower_bound {
                item.current_child = 0;
            } else {
                item.current_child = 1;
            }

            if children[item.current_child].is_complete() {
                item.current_child = 1 - item.current_child;
            }

            Some(children[0].cost_upper_bound + children[1].cost_upper_bound)
        } else {
            None
        }
    }

    fn recalculate_item_lb(&mut self, item: &mut QueueItem<'a, OT>) {
        let lb_left = self.best_left_subtree_left_of[item.feature]
            .range(..(item.split_points.start - 1))
            .last();
        let lb_right = self.best_right_subtree_right_of[item.feature]
            .range((item.split_points.end + 1)..)
            .next();

        let lb_total = match (lb_left, lb_right) {
            (Some((_, &l)), Some((_, &r))) => l + r,
            (Some((_, &l)), None) => l,
            (None, Some((_, &r))) => r,
            (None, None) => OT::MIN_COST,
        };

        if lb_total > item.cost_lower_bound {
            item.cost_lower_bound = lb_total;
        }
    }

    pub fn backtrack_item(&mut self, mut item: QueueItem<'a, OT>) {
        if let Some(ub) = self.get_upper_and_update_lower_bound_from_children(&mut item) {
            info!("New UB: {:?}", ub);
            if self.cost_upper_bound > ub {
                self.cost_upper_bound = ub;
                self.best = (item.feature as i32, item.split_points.clone(), ub)
            }
        }

        let mut next_o = Some(item);
        while let Some(mut next) = next_o.take() {
            self.recalculate_item_lb(&mut next);
            let prev_lb = next.cost_lower_bound;

            // Only revisit this node if it is not yet fully explored.
            if !next.is_complete() {
                self.queue.push(next);
            }

            if let Some(next_in_line) = self.queue.peek() {
                let lowest_lb = next_in_line.cost_lower_bound;

                // Lower bound of the node is at least the minimum lower bound in the queue.
                if lowest_lb > self.cost_lower_bound {
                    self.cost_lower_bound = lowest_lb;
                }

                // If the front of the queue has a lesser lower bound than the
                // one we just reinserted, then we need to continue updating.
                if lowest_lb < prev_lb {
                    next_o = self.queue.pop();
                }
            } else {
                // Queue empty, we are done.
                self.cost_lower_bound = self.cost_upper_bound;
            }
        }
    }
}
