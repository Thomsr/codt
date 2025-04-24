use std::{
    cmp::Ordering, collections::BinaryHeap, fmt::Debug, marker::PhantomData, ops::Range, sync::Arc,
};

use crate::{
    model::{dataview::DataView, tree::Tree},
    tasks::{CostSum, OptimizationTask},
};

use super::{pruner::Pruner, strategy::SearchStrategy};

pub struct QueueItem<'a, OT: OptimizationTask, SS: SearchStrategy> {
    /// The current lower bound on the error for this item in the queue.
    pub cost_lower_bound: OT::CostType,

    /// The fraction of remaining instances from the original dataset in the dataview of this item.
    /// Used by some search heuristics.
    pub remaining_fraction: f64,

    /// The branching test for this node is one of `feature <= possible_split_points[s]` where s in split_points.
    pub feature: usize,
    /// The branching test for this node is one of `feature <= possible_split_points[s]` where s in split_points.
    pub split_points: Range<usize>,

    // Child nodes can only be initiated once the size of the `split_points` range is one.
    pub children: Option<[Node<'a, OT, SS>; 2]>,

    /// The index of the child currently under consideration. E.g. for best first search, the child with the lowest lower bound.
    pub current_child: usize,

    _ss: PhantomData<SS>,
}

impl<OT: OptimizationTask, SS: SearchStrategy> Debug for QueueItem<'_, OT, SS> {
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

impl<OT: OptimizationTask, SS: SearchStrategy> PartialEq for QueueItem<'_, OT, SS> {
    fn eq(&self, other: &Self) -> bool {
        // While the Ord checks many more attributes, there should never be an item in
        // the queue for the same feature and an overlapping interval.
        self.feature == other.feature && self.split_points.start == other.split_points.start
    }
}

impl<OT: OptimizationTask, SS: SearchStrategy> Eq for QueueItem<'_, OT, SS> {}

impl<OT: OptimizationTask, SS: SearchStrategy> PartialOrd for QueueItem<'_, OT, SS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<OT: OptimizationTask, SS: SearchStrategy> Ord for QueueItem<'_, OT, SS> {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap. But we want the minumum, so reverse.
        SS::cmp_item(self, other).reverse()
    }
}

impl<'a, OT: OptimizationTask, SS: SearchStrategy> QueueItem<'a, OT, SS> {
    fn new(feature: usize, split_points: Range<usize>, remaining_fraction: f64) -> Self {
        Self {
            cost_lower_bound: OT::MIN_COST,
            remaining_fraction,
            feature,
            split_points,
            children: None,
            current_child: 0,
            _ss: PhantomData,
        }
    }

    /// Sets this item to split at an exact point. Returns the QueueItems left
    /// after splitting at this concrete splitting point. Only returns an item
    /// if there are remaining splits in that interval.
    pub fn split_at(&mut self, split: usize) -> (Option<Self>, Option<Self>) {
        let mut left = None;
        let mut right = None;
        if split - self.split_points.start > 0 {
            left = Some(Self {
                cost_lower_bound: self.cost_lower_bound,
                remaining_fraction: self.remaining_fraction,
                feature: self.feature,
                split_points: self.split_points.start..split,
                children: None,
                current_child: 0,
                _ss: PhantomData,
            })
        }
        if self.split_points.end - 1 - split > 0 {
            right = Some(Self {
                cost_lower_bound: self.cost_lower_bound,
                remaining_fraction: self.remaining_fraction,
                feature: self.feature,
                split_points: (split + 1)..self.split_points.end,
                children: None,
                current_child: 0,
                _ss: PhantomData,
            })
        }
        self.split_points = split..(split + 1);

        (left, right)
    }

    pub fn is_expanded(&self) -> bool {
        self.children.is_some()
    }

    /// A queue item is complete if it is guaranteed that it has expanded its best solution,
    /// or that this node is not part of the optimal solution (lower bound > upper bound).
    pub fn is_complete(&self) -> bool {
        if let Some(children) = &self.children {
            children[0].is_complete() && children[1].is_complete()
        } else {
            false
        }
    }

    pub fn current_node(&self) -> Option<&Node<'a, OT, SS>> {
        self.children.as_ref().map(|c| &c[self.current_child])
    }

    pub fn current_node_mut(&mut self) -> Option<&mut Node<'a, OT, SS>> {
        self.children.as_mut().map(|c| &mut c[self.current_child])
    }

    /// The lowest heuristic value of an unexpanded node descendant. Own lower bound if
    /// not expanded, otherwise the lowest of its children.
    pub fn lowest_descendant_heuristic(&self) -> f64 {
        if let Some(children) = &self.children {
            children[0]
                .lowest_descendant_heuristic
                .min(children[1].lowest_descendant_heuristic)
        } else {
            SS::heuristic_from_lb_and_remaining_fraction::<OT>(
                self.cost_lower_bound,
                self.remaining_fraction,
            )
        }
    }
}

/// Represents a search node for a concrete feature test.
pub struct Node<'a, OT: OptimizationTask, SS: SearchStrategy> {
    /// The lower bound on the cost for this node.
    pub cost_lower_bound: OT::CostType,
    /// The lowest heuristic value (search strategy dependent) of any
    /// descendant of this node (including the node itself). This value
    /// is used to emulate a global queue for some heuristic selection methods.
    pub lowest_descendant_heuristic: f64,
    /// The remaining depth for the tree rooted at this node.
    pub remaining_depth_budget: u32,
    /// The view of the dataset containing each remaining instance.
    pub dataview: DataView<'a, OT>,
    /// The remaining candidate children for an optimal solution.
    pub queue: BinaryHeap<QueueItem<'a, OT, SS>>,
    /// Best tree found so far.
    pub best: Arc<Tree<OT>>,

    pruner: Pruner<OT>,
}

impl<'a, OT: OptimizationTask, SS: SearchStrategy> Node<'a, OT, SS> {
    pub fn new(dataview: DataView<'a, OT>, max_depth: u32) -> Self {
        let ub = dataview.cost_summer.cost();

        let mut queue = BinaryHeap::new();

        if max_depth > 0 {
            for feature in 0..dataview.num_features() {
                let n_splitpoints = dataview.possible_split_values[feature].len();
                if n_splitpoints > 0 {
                    queue.push(QueueItem::new(
                        feature,
                        0..n_splitpoints,
                        dataview.remaining_fraction(),
                    ));
                }
            }
        }

        let lb = if max_depth == 0 || queue.is_empty() {
            ub
        } else {
            OT::MIN_COST
        };

        Node {
            cost_lower_bound: lb,
            lowest_descendant_heuristic: SS::heuristic_from_lb_and_remaining_fraction::<OT>(
                lb,
                dataview.remaining_fraction(),
            ),
            remaining_depth_budget: max_depth,
            pruner: Pruner::new(dataview.num_features()),
            best: Arc::new(Tree::new_leaf(dataview.cost_summer.label(), ub)),
            dataview,
            queue,
        }
    }

    /// A search node is complete if it is guaranteed that it has expanded its best solution,
    /// or that this node is not part of the optimal solution (lower bound > upper bound).
    pub fn is_complete(&self) -> bool {
        self.cost_lower_bound >= self.best.cost()
    }

    pub fn split(&self, feature: usize, split: usize) -> (Self, Self) {
        let (left_view, right_view) = self.dataview.split(feature, split);

        let left = Self::new(left_view, self.remaining_depth_budget - 1);
        let right = Self::new(right_view, self.remaining_depth_budget - 1);

        (left, right)
    }

    pub fn find_lowest_cost_split(&self, _feature: usize, range: &Range<usize>) -> usize {
        // get rs = right solution
        // get ls = left solution
        // for x = distance from a solution:
        // lower bound for left = max(ls.left, rs.left-worst*x)
        // lower bound for right = max(rs.right, ls.right-worst*x)
        // find minimum index of left_lb + right_lb. If in a flat area, pick the center of that area.
        assert!(range.start < range.end);
        (range.start + range.end) / 2
    }

    fn get_upper_and_update_lower_bound_from_children(
        &mut self,
        item: &mut QueueItem<'a, OT, SS>,
    ) -> Option<OT::CostType> {
        if let Some(children) = &item.children {
            // No actual LB update here, add the solution to the pruner. The LB will be updated by it later.
            self.pruner.insert_left_subtree(
                item.feature,
                item.split_points.start,
                children[0].cost_lower_bound,
            );
            self.pruner.insert_right_subtree(
                item.feature,
                item.split_points.end - 1,
                children[1].cost_lower_bound,
            );

            item.current_child = SS::child_priority(&children[0], &children[1]);

            if children[item.current_child].is_complete() {
                item.current_child = 1 - item.current_child;
            }

            Some(children[0].best.cost() + children[1].best.cost())
        } else {
            None
        }
    }

    pub fn recalculate_item_lb(&mut self, item: &mut QueueItem<'a, OT, SS>) {
        let lb = self.pruner.lb_for(item.feature, &item.split_points);
        OT::update_lowerbound(&mut item.cost_lower_bound, &lb);
    }

    pub fn backtrack_item(&mut self, mut item: QueueItem<'a, OT, SS>) {
        // Update our upper bound if the item is the current best.
        if let Some(ub) = self.get_upper_and_update_lower_bound_from_children(&mut item) {
            if ub < self.best.cost() {
                assert!(item.split_points.start == item.split_points.end - 1);
                if ub == OT::MIN_COST {
                    // We cannot find a better solution: quickly clear the queue.
                    self.queue.clear();
                }
                let children = item
                    .children
                    .as_ref()
                    .expect("An item can only be backtracked once it is expanded.");
                self.best = Arc::new(Tree::new_branch(
                    item.feature,
                    self.dataview
                        .threshold_from_split(item.feature, item.split_points.start),
                    children[0].best.clone(),
                    children[1].best.clone(),
                ))
            }
        }

        // Reinsert the item in the queue, and ensure the item in the front
        // of the queue has updated lower bounds for the next iteration.
        let mut maybe_item = Some(item);
        while let Some(mut item) = maybe_item.take() {
            self.recalculate_item_lb(&mut item);

            let update_needed = if item.is_complete() || item.cost_lower_bound >= self.best.cost() {
                // Don't add it back to the queue if the item cannot further improve the solution.
                // Should update the lower bounds of the next queue item.
                true
            } else {
                let update_needed = if let Some(next) = self.queue.peek() {
                    // Update needed if the current item will not be returned to the front of the queue.
                    next > &item
                } else {
                    // This is the last item in the queue, no further updates needed.
                    false
                };

                self.queue.push(item);
                update_needed
            };

            maybe_item = if update_needed {
                self.queue.pop()
            } else {
                None
            }
        }

        if let Some(next) = self.queue.peek() {
            if SS::item_front_of_queue_is_lowest_lb(next) {
                OT::update_lowerbound(&mut self.cost_lower_bound, &next.cost_lower_bound);
            }
            // Self or front of queue, since for bfs the front of queue will contain the lowest heuristic value.
            self.lowest_descendant_heuristic = SS::heuristic_from_lb_and_remaining_fraction::<OT>(
                self.cost_lower_bound,
                self.dataview.remaining_fraction(),
            )
            .min(next.lowest_descendant_heuristic())
        } else {
            // Queue empty, we are done.
            OT::update_lowerbound(&mut self.cost_lower_bound, &self.best.cost());
            // Set heuristic to max, so partially unfinished queue items take the lower bound from its sibling, and not from a completed node.
            self.lowest_descendant_heuristic = f64::MAX;
        }
    }
}
