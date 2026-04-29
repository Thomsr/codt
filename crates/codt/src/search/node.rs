use std::{
    cmp::Ordering, collections::VecDeque, fmt::Debug, marker::PhantomData, ops::Range, sync::Arc,
};

use crate::{
    model::{
        dataview::DataView,
        tree::{BranchNode, LeafNode, Tree},
    },
    search::{
        lower_bounds::{
            class_count::class_count_lower_bound, improvement::improvement_lower_bound,
        },
        pruner::Pruner,
        queue::{BinaryHeapQueue, PQ},
        solver::{LowerBoundStrategy, UpperboundStrategy},
        solver_impl::SolveContext,
        strategy::SearchStrategy,
        upper_bounds::cart::cart_upper_bound_with_subset_seed,
    },
    tasks::{Cost, CostSum, OptimizationTask},
};

pub enum ExpandedQueueItem<'a, OT: OptimizationTask, SS: SearchStrategy> {
    Children([Node<'a, OT, SS>; 2]),
    Solution(Arc<Tree<OT>>),
}

impl<OT: OptimizationTask, SS: SearchStrategy> ExpandedQueueItem<'_, OT, SS> {
    fn lower_bound_left(&self) -> OT::CostType {
        match self {
            Self::Children(children) => children[0].cost_lower_bound,
            Self::Solution(tree) => match (*tree).as_ref() {
                Tree::Branch(branch) => branch.left_child.cost(),
                _ => unreachable!(),
            },
        }
    }

    fn lower_bound_right(&self) -> OT::CostType {
        match self {
            Self::Children(children) => children[1].cost_lower_bound,
            Self::Solution(tree) => match (*tree).as_ref() {
                Tree::Branch(branch) => branch.right_child.cost(),
                _ => unreachable!(),
            },
        }
    }

    fn upper_bound_left(&self) -> OT::CostType {
        match self {
            Self::Children(children) => children[0].cost_upper_bound,
            Self::Solution(tree) => match (*tree).as_ref() {
                Tree::Branch(branch) => branch.left_child.cost(),
                _ => unreachable!(),
            },
        }
    }

    fn upper_bound_right(&self) -> OT::CostType {
        match self {
            Self::Children(children) => children[1].cost_upper_bound,
            Self::Solution(tree) => match (*tree).as_ref() {
                Tree::Branch(branch) => branch.right_child.cost(),
                _ => unreachable!(),
            },
        }
    }

    fn upper_bound(&self, context: &SolveContext<'_, OT, SS>) -> OT::CostType {
        match self {
            Self::Children(children) => {
                children[0].cost_upper_bound
                    + children[1].cost_upper_bound
                    + context.task.branching_cost()
            }
            Self::Solution(tree) => tree.cost(),
        }
    }

    fn best_cost(&self, context: &SolveContext<'_, OT, SS>) -> OT::CostType {
        match self {
            Self::Children(children) => {
                children[0].best.cost() + children[1].best.cost() + context.task.branching_cost()
            }
            Self::Solution(tree) => tree.cost(),
        }
    }
}

pub struct FeatureTest<'a, OT: OptimizationTask, SS: SearchStrategy> {
    /// The current lower bound on the error for this item in the queue. This is including possible branching costs
    pub cost_lower_bound: OT::CostType,

    /// The number of instances that reach this node.
    /// Used by some search heuristics.
    pub support: usize,

    /// The feature index of this feature test.
    pub feature: usize,
    /// The range of threshold indices that are ideally pruned by searching for this feature test.
    pub split_points: Range<usize>,

    /// The exact point at which to split the feature. May change until expanded.
    pub split_point: usize,

    /// Either a left and right node for this feature test, or an optimal tree. Set once the search expands this feature test.
    expanded: Option<ExpandedQueueItem<'a, OT, SS>>,

    /// A pseudorandom value for the node, if the search strategy requires it
    pub random_value: u64,

    /// The rank of the feature based on the greedy heuristic value, used by some search strategies to prioritize items.
    pub feature_rank: i32,

    /// The number of discrepancies from the heuristically best solution.
    /// This is the discrepancies of the parent + feature_rank + the number of subdivisions this interval has gone through.
    pub discrepancies: i32,

    _ss: PhantomData<SS>,
}

impl<OT: OptimizationTask, SS: SearchStrategy> Debug for FeatureTest<'_, OT, SS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueueItem")
            .field("cost_lower_bound", &self.cost_lower_bound)
            .field("feature", &self.feature)
            .field("split_points", &self.split_points)
            .field("children", &self.expanded.is_some())
            .finish()
    }
}

impl<OT: OptimizationTask, SS: SearchStrategy> PartialEq for FeatureTest<'_, OT, SS> {
    fn eq(&self, other: &Self) -> bool {
        // While the Ord checks many more attributes, there should never be an item in
        // the queue for the same feature test.
        self.feature == other.feature && self.split_point == other.split_point
    }
}

impl<OT: OptimizationTask, SS: SearchStrategy> Eq for FeatureTest<'_, OT, SS> {}

impl<OT: OptimizationTask, SS: SearchStrategy> PartialOrd for FeatureTest<'_, OT, SS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<OT: OptimizationTask, SS: SearchStrategy> Ord for FeatureTest<'_, OT, SS> {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap. But we want the minimum, so reverse.
        SS::cmp_item(self, other).reverse()
    }
}

impl<'a, OT: OptimizationTask, SS: SearchStrategy> FeatureTest<'a, OT, SS> {
    fn new(
        context: &SolveContext<OT, SS>,
        feature: usize,
        split_points: Range<usize>,
        split_point: usize,
        support: usize,
        feature_rank: i32,
        discrepancies: i32,
    ) -> Self {
        Self {
            cost_lower_bound: context.task.branching_cost(),
            support,
            feature,
            split_points,
            split_point,
            expanded: None,
            random_value: if SS::generate_random_value() {
                rand::random()
            } else {
                0
            },
            feature_rank,
            discrepancies: discrepancies + feature_rank,
            _ss: PhantomData,
        }
    }

    /// Sets this item to split at an exact point. Returns the QueueItems left
    /// after splitting at this concrete splitting point. Only returns an item
    /// if there are remaining splits in that interval.
    fn split_off(&self) -> (Option<Self>, Option<Self>) {
        let mut left = None;
        let mut right = None;
        let split = self.split_point;
        if split - self.split_points.start > 0 {
            left = Some(Self {
                cost_lower_bound: self.cost_lower_bound,
                support: self.support,
                feature: self.feature,
                split_points: self.split_points.start..split,
                split_point: (self.split_points.start + split) / 2,
                expanded: None,
                random_value: if SS::generate_random_value() {
                    rand::random()
                } else {
                    0
                },
                feature_rank: self.feature_rank,
                discrepancies: self.discrepancies + 1,
                _ss: PhantomData,
            })
        }
        if self.split_points.end - 1 - split > 0 {
            right = Some(Self {
                cost_lower_bound: self.cost_lower_bound,
                support: self.support,
                feature: self.feature,
                split_points: (split + 1)..self.split_points.end,
                split_point: (split + 1 + self.split_points.end) / 2,
                expanded: None,
                random_value: if SS::generate_random_value() {
                    rand::random()
                } else {
                    0
                },
                feature_rank: self.feature_rank,
                discrepancies: self.discrepancies + 1,
                _ss: PhantomData,
            })
        }

        (left, right)
    }

    pub fn is_expanded(&self) -> bool {
        self.expanded.is_some()
    }

    /// A queue item is complete if it is guaranteed that it has expanded its best solution,
    /// or that this node is not part of the optimal solution (lower bound > upper bound).
    pub fn is_complete(&self) -> bool {
        if let Some(expanded) = &self.expanded {
            match expanded {
                ExpandedQueueItem::Children(children) => {
                    children[0].is_complete() && children[1].is_complete()
                }
                ExpandedQueueItem::Solution(_) => true,
            }
        } else {
            false
        }
    }

    pub fn child_by_idx(&mut self, idx: usize) -> Option<&mut Node<'a, OT, SS>> {
        self.expanded.as_mut().and_then(|children| match children {
            ExpandedQueueItem::Children(children) => Some(&mut children[idx]),
            ExpandedQueueItem::Solution(_) => None,
        })
    }

    /// If expanded, returns either the left or right child which should be explored next.
    pub fn current_node(&mut self) -> Option<usize> {
        return if let Some(ExpandedQueueItem::Children(children)) = self.expanded.take() {
            let heuristic_child = SS::child_priority(self, &children);

            let selected_child = if children[heuristic_child].is_complete() {
                1 - heuristic_child
            } else {
                heuristic_child
            };

            self.expanded = Some(ExpandedQueueItem::Children(children));
            Some(selected_child)
        } else {
            None
        };
    }

    /// The lowest heuristic value of an unexpanded node descendant. Own lower bound if
    /// not expanded, otherwise the lowest of its children.
    pub fn lowest_descendant_heuristic(&self) -> f64 {
        if let Some(expanded) = &self.expanded {
            match expanded {
                ExpandedQueueItem::Children(children) => children[0]
                    .lowest_descendant_heuristic
                    .min(children[1].lowest_descendant_heuristic),
                // Set heuristic to max, so partially unfinished queue items take the lower bound from its sibling, and not from a completed node.
                ExpandedQueueItem::Solution(_) => f64::MAX,
            }
        } else {
            SS::heuristic(self)
        }
    }

    /// If the item is expanded, it can be pruned as soon as its split point is infeasible.
    /// If the item is not expanded, we need to consider the full range before it is pruned.
    pub fn lb_range(&self) -> Range<usize> {
        if self.is_expanded() {
            self.split_point..self.split_point + 1
        } else {
            self.split_points.clone()
        }
    }
}

/// Represents a search node for a concrete feature test.
pub struct Node<'a, OT: OptimizationTask, SS: SearchStrategy> {
    /// The upper bound on the cost for this node.
    pub cost_upper_bound: OT::CostType,
    /// The lower bound on the cost for this node.
    pub cost_lower_bound: OT::CostType,
    /// The lowest heuristic value (search strategy dependent) of any
    /// descendant of this node (including the node itself). This value
    /// is used to emulate a global queue for some heuristic selection methods.
    pub lowest_descendant_heuristic: f64,
    /// The view of the dataset containing each remaining instance.
    pub dataview: DataView<'a, OT>,
    /// The remaining candidate children for an optimal solution.
    pub queue: BinaryHeapQueue<FeatureTest<'a, OT, SS>>,
    /// Best tree found so far.
    pub best: Arc<Tree<OT>>,
    /// Keeps track of found lower bounds for pruning similar items
    pruner: Pruner<OT>,
    /// The threshold range we are still interested in per feature. Initially the full range, but shrunk when a zero solution is found.
    pub interesting_solutions_range: Vec<Range<usize>>,
}

impl<'a, OT: OptimizationTask, SS: SearchStrategy> Node<'a, OT, SS> {
    pub fn new(
        context: &SolveContext<OT, SS>,
        dataview: DataView<'a, OT>,
        discrepancies: i32,
    ) -> Self {
        // Replaced by full range if we are actually searching
        let mut interesting_solutions_range = vec![0..0; dataview.num_features()];

        let leaf_cost = dataview.cost_summer.cost();
        let mut ub = leaf_cost;
        let mut best = Arc::new(Tree::Leaf(LeafNode {
            cost: leaf_cost,
            label: dataview.cost_summer.label(),
        }));

        let mut queue = BinaryHeapQueue::default();

        if dataview.num_instances() > 1 {
            for (feature, interesting_solutions_range) in
                interesting_solutions_range.iter_mut().enumerate()
            {
                let n_splitpoints = dataview.possible_split_values[feature].len();
                *interesting_solutions_range = 0..n_splitpoints;

                if n_splitpoints == 0 {
                    continue;
                }

                let mut feature_test = FeatureTest::new(
                    context,
                    feature,
                    0..n_splitpoints,
                    n_splitpoints / 2,
                    dataview.num_instances(),
                    dataview.feature_ranking[feature],
                    discrepancies,
                );

                if SS::should_greedily_split() {
                    feature_test.split_point =
                        dataview.best_greedy_splits[feature].split_value_index;
                    let (left, right) = feature_test.split_off();
                    if let Some(left) = left {
                        queue.push(left);
                    }
                    if let Some(right) = right {
                        queue.push(right);
                    }
                }

                queue.push(feature_test);
            }
        }

        if context.cart_ub {
            let max_iterations = 1;
            for seed in 42..(42 + max_iterations as u64) {
                let cart_tree =
                    cart_upper_bound_with_subset_seed(context.task, &dataview, Some(false), seed);
                let cart_ub = cart_tree.cost();
                if cart_ub.less_or_not_much_greater_than(&ub) {
                    ub = cart_ub;
                    best = cart_tree;
                }
            }
        }

        let use_class_count_lb = context
            .lb_strategy
            .contains(&LowerBoundStrategy::ClassCount);

        let use_improvement_lb = context
            .lb_strategy
            .contains(&LowerBoundStrategy::Improvement);

        let lb = match (queue.is_empty(), use_class_count_lb, use_improvement_lb) {
            (true, _, _) => ub,
            (false, true, false) => class_count_lower_bound::<OT>(dataview.num_unique_labels()),
            (false, false, true) => improvement_lower_bound::<OT>(&dataview),
            (false, true, true) => {
                let class_count_lb = class_count_lower_bound::<OT>(dataview.num_unique_labels());
                let improvement_lb = improvement_lower_bound::<OT>(&dataview);
                if class_count_lb.greater_or_not_much_less_than(&improvement_lb) {
                    class_count_lb
                } else {
                    improvement_lb
                }
            }
            (false, false, false) => context.task.branching_cost(),
        };

        Node {
            cost_upper_bound: ub,
            cost_lower_bound: lb,
            lowest_descendant_heuristic: queue
                .peek()
                .map_or(f64::MAX, |item| item.lowest_descendant_heuristic()),
            pruner: Pruner::new(dataview.num_features()),
            best,
            dataview,
            queue,
            interesting_solutions_range,
        }
    }

    /// A search node is complete if it is guaranteed that it has expanded its best solution,
    /// or that this node is not part of the optimal solution (lower bound > upper bound).
    pub fn is_complete(&self) -> bool {
        self.cost_lower_bound
            .greater_or_not_much_less_than(&self.best.cost())
            || self
                .cost_lower_bound
                .strictly_greater_than(&self.cost_upper_bound)
            || self.queue.is_empty() // When the queue is empty before best == lower, then lower == upper and lower < best.
    }

    // Get the worst cost of the instances with feature value in the range.
    fn worst_cost_in_range(&self, feature: usize, range: Range<usize>) -> OT::CostType {
        if range.is_empty() {
            return OT::CostType::ZERO;
        }
        OT::worst_cost_in_range(
            &self.dataview,
            feature,
            self.dataview
                .instance_range_from_split_range(feature, range),
        )
    }

    fn compute_child_upper_bound(
        &mut self,
        context: &SolveContext<OT, SS>,
        child: &mut Node<'a, OT, SS>,
        sibling_lb: OT::CostType,
        interval_margin: OT::CostType,
    ) {
        let ub_new = match context.ub_strategy {
            UpperboundStrategy::SolutionsOnly => self.best.cost(),
            UpperboundStrategy::TightFromSibling => {
                self.cost_upper_bound - context.task.branching_cost() - sibling_lb
            }
            UpperboundStrategy::ForRemainingInterval => {
                // Best solution is never worse than this
                let best = self.best.cost();
                // Optimal solution is never worse than the upper bound, but we are interested in a better solution for pruning the remaining interval.
                let remaining_interval =
                    self.cost_upper_bound - context.task.branching_cost() - sibling_lb
                        + interval_margin;
                if best.less_or_not_much_greater_than(&remaining_interval) {
                    best
                } else {
                    remaining_interval
                }
            }
        };

        OT::update_upperbound(&mut child.cost_upper_bound, &ub_new);
    }

    pub fn lb_for(&self, item: &FeatureTest<'a, OT, SS>) -> OT::CostType {
        let lb_range = item.lb_range();

        let (closest_left, closest_left_lb) =
            self.pruner.closest_left_lb(item.feature, lb_range.start);

        let (closest_right, closest_right_lb) =
            self.pruner.closest_right_lb(item.feature, lb_range.end - 1);

        let mut lb = self.pruner.lb_for(item.feature, &lb_range);

        let neighbour_lb_left = closest_left_lb
            - self.worst_cost_in_range(item.feature, (closest_left + 1)..lb_range.end);
        let neighbour_lb_right = closest_right_lb
            - self.worst_cost_in_range(item.feature, lb_range.start..closest_right);
        OT::update_lowerbound(&mut lb, &neighbour_lb_left);
        OT::update_lowerbound(&mut lb, &neighbour_lb_right);
        lb
    }

    fn partition_point<P>(range: &Range<usize>, mut pred: P) -> usize
    where
        P: FnMut(usize) -> bool,
    {
        let mut lo = range.start;
        let mut hi = range.end;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if pred(mid) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn shrink_interval(&mut self, item: &mut FeatureTest<'a, OT, SS>) {
        // Find the closest left and right lower bound.
        let (closest_left, closest_left_lb) = self
            .pruner
            .closest_left_lb(item.feature, item.split_points.start);
        let (closest_right, closest_right_lb) = self
            .pruner
            .closest_right_lb(item.feature, item.split_points.end - 1);

        // Calculate the minimum divergence from those lower bounds for them to be interesting.
        let leeway_left = closest_left_lb - self.cost_upper_bound;
        let leeway_right = closest_right_lb - self.cost_upper_bound;

        // Shrink the interval based on the leeway.
        item.split_points.start = Self::partition_point(&item.split_points, |x| {
            leeway_left.strictly_greater_than(
                &self.worst_cost_in_range(item.feature, (closest_left + 1)..(x + 1)),
            )
        });
        item.split_points.end = Self::partition_point(&item.split_points, |x| {
            !leeway_right
                .strictly_greater_than(&self.worst_cost_in_range(item.feature, x..closest_right))
        });

        // Shrink the interval based on the interesting points
        item.split_points.start = item
            .split_points
            .start
            .max(self.interesting_solutions_range[item.feature].start);
        item.split_points.end = item
            .split_points
            .end
            .min(self.interesting_solutions_range[item.feature].end);
    }

    fn recalculate_item_bounds(
        &mut self,
        context: &SolveContext<OT, SS>,
        item: &mut FeatureTest<'a, OT, SS>,
    ) {
        let lb = self.lb_for(item) + context.task.branching_cost();
        OT::update_lowerbound(&mut item.cost_lower_bound, &lb);

        if let Some(ExpandedQueueItem::Children(children)) = &mut item.expanded {
            let child0_lb = children[0].cost_lower_bound;
            let child1_lb = children[1].cost_lower_bound;

            // Calculate the worst of the margins on the left and right sides.
            let margin_left =
                self.worst_cost_in_range(item.feature, item.split_points.start..item.split_point);
            let margin_right =
                self.worst_cost_in_range(item.feature, item.split_point + 1..item.split_points.end);

            let margin = if margin_left.greater_or_not_much_less_than(&margin_right) {
                margin_left
            } else {
                margin_right
            };

            self.compute_child_upper_bound(context, &mut children[0], child1_lb, margin);
            self.compute_child_upper_bound(context, &mut children[1], child0_lb, margin);
            // Child lower bounds could be computed in a similar way, but are useless.
            // Initial lower bounds are better for children, and other lower bounds for the parent are gained by backtracking the child.
        }
    }

    /// Reinsert an updated item in the queue after expanding a descendant.
    pub fn backtrack_item(
        &mut self,
        context: &SolveContext<OT, SS>,
        item: FeatureTest<'a, OT, SS>,
    ) {
        let expanded = item
            .expanded
            .as_ref()
            .expect("An item can only be backtracked if it has been expanded.");
        // No actual LB update here, add the solution to the pruner. The LB of the item will be updated by the pruner later.
        self.pruner.insert_left_subtree(
            item.feature,
            item.split_point,
            expanded.lower_bound_left(),
        );
        self.pruner.insert_right_subtree(
            item.feature,
            item.split_point,
            expanded.lower_bound_right(),
        );

        if expanded
            .upper_bound_left()
            .less_or_not_much_greater_than(&OT::CostType::ZERO)
        {
            self.interesting_solutions_range[item.feature].start = self.interesting_solutions_range
                [item.feature]
                .start
                .max(item.split_point);
        }
        if expanded
            .upper_bound_right()
            .less_or_not_much_greater_than(&OT::CostType::ZERO)
        {
            self.interesting_solutions_range[item.feature].end = self.interesting_solutions_range
                [item.feature]
                .end
                .min(item.split_point + 1);
        }

        // Update our upper bound based on the upper bound of this item.
        let ub = expanded.upper_bound(context);
        OT::update_upperbound(&mut self.cost_upper_bound, &ub);

        // Update our current best item.
        let best_cost = expanded.best_cost(context);
        if best_cost.strictly_less_than(&self.best.cost()) {
            self.best = match expanded {
                ExpandedQueueItem::Children(children) => Arc::new(Tree::Branch(BranchNode {
                    cost: best_cost,
                    split_feature: self
                        .dataview
                        .original_split_feature_from_split(item.feature, item.split_point),
                    split_threshold: self
                        .dataview
                        .threshold_from_split(item.feature, item.split_point),
                    left_child: children[0].best.clone(),
                    right_child: children[1].best.clone(),
                })),
                ExpandedQueueItem::Solution(tree) => tree.clone(),
            }
        }

        assert!(
            self.cost_upper_bound
                .less_or_not_much_greater_than(&self.best.cost())
        );
        if best_cost.less_or_not_much_greater_than(&context.task.branching_cost()) {
            // We cannot find a better solution from splitting: quickly clear the queue.
            self.queue.clear();
        }

        // Reinsert the item in the queue, and ensure the item in the front
        // of the queue has updated lower bounds for the next iteration.
        let mut maybe_item = Some(item);
        while let Some(mut item) = maybe_item.take() {
            // Shrink the range we need to prune first, as this impacts the upper bound computation
            self.shrink_interval(&mut item);
            // Prune if interval is empty after shrinking or expanded and the range we care about does not include the split_point.
            if item.split_points.is_empty()
                || item.is_expanded() && !item.split_points.contains(&item.split_point)
            {
                // NOTE: this is not an actual lower bound, but immediately after this the item is pruned.
                // This lower bound cannot be and is not used for similarity bounds.
                item.cost_lower_bound = self.best.cost();
            } else {
                self.recalculate_item_bounds(context, &mut item);
            }

            let update_needed = if item.is_complete()
                || item
                    .cost_lower_bound
                    .greater_or_not_much_less_than(&self.best.cost())
                || item
                    .cost_lower_bound
                    .strictly_greater_than(&self.cost_upper_bound)
            {
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

                // Note that the order needs to be antisymmetric, so this is guaranteed to return to front of queue if next <= &item
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
            // For bfs the front of queue will contain the lowest heuristic value.
            // Note that in this case we are always expanded, so we always want the lowest heuristic of the child, not of itself.
            self.lowest_descendant_heuristic = next.lowest_descendant_heuristic();
        } else {
            // Queue empty, we are done. No lower error solution than the upper bound could be found.
            assert!(
                self.cost_upper_bound
                    .less_or_not_much_greater_than(&self.best.cost())
            );
            OT::update_lowerbound(&mut self.cost_lower_bound, &self.cost_upper_bound);

            // Set heuristic to max, so partially unfinished queue items take the lower bound from its sibling, and not from a completed node.
            self.lowest_descendant_heuristic = f64::MAX;
        }
    }

    /// Select the path to the item to expand next. The deepest item is first in the path.
    pub fn select(
        &mut self,
        path_buffer: &mut VecDeque<(usize, FeatureTest<'a, OT, SS>)>,
        source: usize,
    ) {
        assert!(path_buffer.is_empty());
        assert!(!self.is_complete());

        let mut next = self
            .queue
            .pop()
            .expect("Select should only be called when the node is not yet complete.");

        if let Some(child_idx) = next.current_node() {
            let child = next.child_by_idx(child_idx).unwrap();
            child.select(path_buffer, child_idx);
        }

        path_buffer.push_back((source, next));
    }

    /// Expand an item from the queue one level by selecting a concrete split and instantiating its children.
    pub fn expand(&mut self, context: &SolveContext<OT, SS>, item: &mut FeatureTest<'a, OT, SS>) {
        assert!(!item.is_expanded());

        // TODO: could find better split point here?
        // get rs = right solution
        // get ls = left solution
        // for x = distance from a solution:
        // lower bound for left = max(ls.left, rs.left-worst*x)
        // lower bound for right = max(rs.right, ls.right-worst*x)
        // find minimum index of left_lb + right_lb. If in a flat area, pick the center of that area.
        // item.split_point = self.find_lowest_cost_split(item.feature, &item.split_points);

        // Because of interval shrinking the split point may be out of the range
        item.split_point = (item.split_points.start + item.split_points.end) / 2;
        assert!(
            item.split_points.start <= item.split_point && item.split_point < item.split_points.end
        );

        let (left_item, right_item) = item.split_off();

        if let Some(left_item) = left_item {
            self.queue.push(left_item);
        }
        if let Some(right_item) = right_item {
            self.queue.push(right_item);
        }

        let (left_view, right_view) = self.dataview.split(item.feature, item.split_point);

        let left = Self::new(context, left_view, item.discrepancies);
        let right = Self::new(context, right_view, item.discrepancies);

        let expanded = ExpandedQueueItem::Children([left, right]);
        item.expanded = Some(expanded);
    }
}
