use std::{cmp::Ordering, collections::BinaryHeap, fmt::Debug, marker::PhantomData, ops::Range};

use crate::{model::dataview::DataView, tasks::OptimizationTask};

use super::{pruner::Pruner, strategy::SearchStrategy};

pub struct QueueItem<'a, OT: OptimizationTask, SS: SearchStrategy> {
    pub cost_lower_bound: OT::CostType,

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
    fn new(feature: usize, split_points: Range<usize>) -> Self {
        Self {
            cost_lower_bound: OT::MIN_COST,
            feature,
            split_points,
            children: None,
            current_child: 0,
            _ss: PhantomData,
        }
    }

    pub fn split_at(&mut self, split: usize) -> (Option<Self>, Option<Self>) {
        let mut left = None;
        let mut right = None;
        if split - self.split_points.start > 0 {
            left = Some(Self {
                cost_lower_bound: self.cost_lower_bound,
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
}

/// Represents a search node for a concrete feature test.
pub struct Node<'a, OT: OptimizationTask, SS: SearchStrategy> {
    pub cost_lower_bound: OT::CostType,
    pub cost_upper_bound: OT::CostType,

    pub remaining_depth_budget: u32,

    pub dataview: DataView<'a, OT>,

    pub queue: BinaryHeap<QueueItem<'a, OT, SS>>,

    /// Best upper bound of feature test, split point, and cost.
    /// TODO
    pub best: (i32, Range<i32>, OT::CostType),

    pruner: Pruner<OT>,
}

impl<'a, OT: OptimizationTask, SS: SearchStrategy> Node<'a, OT, SS> {
    pub fn new(dataview: DataView<'a, OT>, max_depth: u32) -> Self {
        let ub = (&dataview.cost_summer).into();
        let lb = if max_depth == 0 { ub } else { OT::MIN_COST };

        let mut queue = BinaryHeap::new();

        if max_depth > 0 {
            for feature in 0..dataview.num_features() {
                let n_splitpoints = dataview.possible_split_values[feature].len();
                if n_splitpoints > 0 {
                    queue.push(QueueItem::new(feature, 0..n_splitpoints));
                }
            }
        }

        Node {
            cost_lower_bound: lb,
            cost_upper_bound: ub,
            remaining_depth_budget: max_depth,
            pruner: Pruner::new(dataview.num_features()),
            dataview,
            queue,
            best: (-1, -1..0, ub),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.cost_lower_bound >= self.cost_upper_bound
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

        (range.start + range.end) / 2
    }

    fn get_upper_and_update_lower_bound_from_children(
        &mut self,
        item: &mut QueueItem<'a, OT, SS>,
    ) -> Option<OT::CostType> {
        if let Some(children) = &item.children {
            self.pruner.insert_left_subtree(
                item.feature,
                item.split_points.start,
                children[0].cost_lower_bound,
            );
            self.pruner.insert_right_subtree(
                item.feature,
                item.split_points.end,
                children[1].cost_lower_bound,
            );
            let lb = children[0].cost_lower_bound + children[1].cost_lower_bound;
            if item.cost_lower_bound < lb {
                item.cost_lower_bound = lb
            }

            item.current_child = SS::child_priority(&children[0], &children[1]);

            if children[item.current_child].is_complete() {
                item.current_child = 1 - item.current_child;
            }

            Some(children[0].cost_upper_bound + children[1].cost_upper_bound)
        } else {
            None
        }
    }

    pub fn recalculate_item_lb(&mut self, item: &mut QueueItem<'a, OT, SS>) {
        let lb = self.pruner.lb_for(item.feature, &item.split_points);

        if lb > item.cost_lower_bound {
            item.cost_lower_bound = lb;
        }
    }

    pub fn backtrack_item(&mut self, mut item: QueueItem<'a, OT, SS>) {
        if let Some(ub) = self.get_upper_and_update_lower_bound_from_children(&mut item) {
            OT::update_upperbound(&mut self.cost_upper_bound, &ub);
        }

        let mut next_o = Some(item);
        while let Some(next) = next_o.take() {
            next_o = SS::backtrack_item(self, next)
        }

        while let Some(next) = self.queue.peek() {
            if next.cost_lower_bound > self.cost_upper_bound {
                self.queue.pop();
            } else {
                break;
            }
        }

        if self.queue.is_empty() {
            // Queue empty, we are done.
            OT::update_lowerbound(&mut self.cost_lower_bound, &self.cost_upper_bound);
        }
    }
}
