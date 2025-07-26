use std::{
    cmp::Ordering, collections::VecDeque, fmt::Debug, marker::PhantomData, ops::Range, sync::Arc,
};

use crate::{
    model::{
        dataview::DataView,
        tree::{BranchNode, LeafNode, Tree},
    },
    search::{
        exhaustive::{solve_d1, solve_d2, solve_left_right},
        pruner::Pruner,
        queue::{BinaryHeapQueue, PQ},
        solver::{TerminalSolver, UpperboundStrategy},
        solver_impl::SolveContext,
        strategy::SearchStrategy,
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

pub struct QueueItem<'a, OT: OptimizationTask, SS: SearchStrategy> {
    /// The current lower bound on the error for this item in the queue. This is including possible branching costs
    pub cost_lower_bound: OT::CostType,

    /// The number of instances that reach this node.
    /// Used by some search heuristics.
    pub support: usize,

    /// The branching test for this node is one of `feature <= possible_split_points[s]` where s in split_points.
    pub feature: usize,
    /// The branching test for this node is one of `feature <= possible_split_points[s]` where s in split_points.
    pub split_points: Range<usize>,

    /// Item can only be expanded once the size of the `split_points` range is one.
    expanded: Option<ExpandedQueueItem<'a, OT, SS>>,

    /// A pseudorandom value for the node, if the search strategy requires it
    pub random_value: u64,

    /// The rank of the feature based on the greedy heuristic value, used by some search strategies to prioritize items.
    pub feature_rank: i32,

    _ss: PhantomData<SS>,
}

impl<OT: OptimizationTask, SS: SearchStrategy> Debug for QueueItem<'_, OT, SS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueueItem")
            .field("cost_lower_bound", &self.cost_lower_bound)
            .field("feature", &self.feature)
            .field("split_points", &self.split_points)
            .field("children", &self.expanded.is_some())
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
    fn new(
        context: &SolveContext<OT, SS>,
        feature: usize,
        split_points: Range<usize>,
        support: usize,
        feature_rank: i32,
    ) -> Self {
        Self {
            cost_lower_bound: context.task.branching_cost(),
            support,
            feature,
            split_points,
            expanded: None,
            random_value: if SS::generate_random_value() {
                rand::random()
            } else {
                0
            },
            feature_rank,
            _ss: PhantomData,
        }
    }

    /// Sets this item to split at an exact point. Returns the QueueItems left
    /// after splitting at this concrete splitting point. Only returns an item
    /// if there are remaining splits in that interval.
    fn split_at(&mut self, split: usize) -> (Option<Self>, Option<Self>) {
        let mut left = None;
        let mut right = None;
        if split - self.split_points.start > 0 {
            left = Some(Self {
                cost_lower_bound: self.cost_lower_bound,
                support: self.support,
                feature: self.feature,
                split_points: self.split_points.start..split,
                expanded: None,
                random_value: if SS::generate_random_value() {
                    rand::random()
                } else {
                    0
                },
                feature_rank: self.feature_rank,
                _ss: PhantomData,
            })
        }
        if self.split_points.end - 1 - split > 0 {
            right = Some(Self {
                cost_lower_bound: self.cost_lower_bound,
                support: self.support,
                feature: self.feature,
                split_points: (split + 1)..self.split_points.end,
                expanded: None,
                random_value: if SS::generate_random_value() {
                    rand::random()
                } else {
                    0
                },
                feature_rank: self.feature_rank,
                _ss: PhantomData,
            })
        }
        self.split_points = split..(split + 1);

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
    /// The remaining depth for the tree rooted at this node.
    pub remaining_depth_budget: u32,
    /// The view of the dataset containing each remaining instance.
    pub dataview: DataView<'a, OT>,
    /// The remaining candidate children for an optimal solution.
    pub queue: BinaryHeapQueue<QueueItem<'a, OT, SS>>,
    /// Best tree found so far.
    pub best: Arc<Tree<OT>>,
    /// Keeps track of found lower bounds for pruning similar items
    pub pruner: Pruner<OT>,
}

impl<'a, OT: OptimizationTask, SS: SearchStrategy> Node<'a, OT, SS> {
    pub fn new(context: &SolveContext<OT, SS>, dataview: DataView<'a, OT>, max_depth: u32) -> Self {
        if max_depth == 2 && context.terminal_solver == TerminalSolver::D2 {
            // If we are at depth 2, we can solve the node exhaustively.
            let tree = solve_d2(&dataview, context);
            let cost = tree.cost();
            return Node {
                cost_upper_bound: cost,
                cost_lower_bound: cost,
                lowest_descendant_heuristic: f64::MAX,
                remaining_depth_budget: max_depth,
                pruner: Pruner::new(dataview.num_features()),
                dataview,
                queue: BinaryHeapQueue::default(),
                best: tree,
            };
        } else if max_depth == 1 && context.terminal_solver == TerminalSolver::D1 {
            // If we are at depth 1, we can solve the node exhaustively.
            let tree = solve_d1(&dataview, context);
            let cost = tree.cost();
            return Node {
                cost_upper_bound: cost,
                cost_lower_bound: cost,
                lowest_descendant_heuristic: f64::MAX,
                remaining_depth_budget: max_depth,
                pruner: Pruner::new(dataview.num_features()),
                dataview,
                queue: BinaryHeapQueue::default(),
                best: tree,
            };
        }

        let ub = dataview.cost_summer.cost();

        let mut queue = BinaryHeapQueue::default();

        if max_depth > 0
            && context.task.branching_cost().strictly_less_than(&ub)
            && dataview.num_instances() > 1
        {
            for feature in 0..dataview.num_features() {
                let n_splitpoints = dataview.possible_split_values[feature].len();
                if n_splitpoints > 0 {
                    queue.push(QueueItem::new(
                        context,
                        feature,
                        0..n_splitpoints,
                        dataview.num_instances(),
                        dataview.feature_ranking[feature],
                    ));
                }
            }
        }

        let lb = if queue.is_empty() {
            ub
        } else {
            // We know that the cost is at least the branching cost (otherwise the queue is empty).
            let mut lb = context.task.branching_cost();

            if context.branch_relaxation.should_compute() {
                OT::update_lowerbound(
                    &mut lb,
                    &context.task.branch_relaxation(&dataview, max_depth),
                );
            }
            lb
        };

        Node {
            cost_upper_bound: ub,
            cost_lower_bound: lb,
            lowest_descendant_heuristic: queue
                .peek()
                .map_or(f64::MAX, |item| item.lowest_descendant_heuristic()),
            remaining_depth_budget: max_depth,
            pruner: Pruner::new(dataview.num_features()),
            best: Arc::new(Tree::Leaf(LeafNode {
                cost: ub,
                label: dataview.cost_summer.label(),
            })),
            dataview,
            queue,
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

    fn split(
        &self,
        context: &SolveContext<OT, SS>,
        feature: usize,
        split: usize,
    ) -> ExpandedQueueItem<'a, OT, SS> {
        let (left_view, right_view) = self.dataview.split(feature, split);

        let left = Self::new(context, left_view, self.remaining_depth_budget - 1);
        let right = Self::new(context, right_view, self.remaining_depth_budget - 1);

        ExpandedQueueItem::Children([left, right])
    }

    fn find_lowest_cost_split(&self, _feature: usize, range: &Range<usize>) -> usize {
        // get rs = right solution
        // get ls = left solution
        // for x = distance from a solution:
        // lower bound for left = max(ls.left, rs.left-worst*x)
        // lower bound for right = max(rs.right, ls.right-worst*x)
        // find minimum index of left_lb + right_lb. If in a flat area, pick the center of that area.
        assert!(range.start < range.end);
        (range.start + range.end) / 2
    }

    fn compute_child_upper_bound(
        &self,
        context: &SolveContext<OT, SS>,
        child: &mut Node<'a, OT, SS>,
        sibling_lb: OT::CostType,
    ) {
        let ub_new = match context.ub_strategy {
            UpperboundStrategy::SolutionsOnly => child.cost_upper_bound,
            UpperboundStrategy::TightFromSibling => {
                self.cost_upper_bound - context.task.branching_cost() - sibling_lb
            }
            UpperboundStrategy::ForRemainingInterval => {
                self.cost_upper_bound - context.task.branching_cost() - sibling_lb
            } // TODO + margin_of_interval,
        };

        OT::update_upperbound(&mut child.cost_upper_bound, &ub_new);
    }

    fn compute_child_lower_bound(
        &self,
        context: &SolveContext<OT, SS>,
        child: &mut Node<'a, OT, SS>,
        sibling_ub: OT::CostType,
    ) {
        let lb_new = self.cost_lower_bound - sibling_ub - context.task.branching_cost();

        OT::update_lowerbound(&mut child.cost_lower_bound, &lb_new);
    }

    fn recalculate_item_bounds(
        &mut self,
        context: &SolveContext<OT, SS>,
        item: &mut QueueItem<'a, OT, SS>,
    ) {
        if let Some(ExpandedQueueItem::Children(children)) = &mut item.expanded {
            let child0_lb = children[0].cost_lower_bound;
            let child0_ub = children[0].cost_upper_bound;
            let child1_lb = children[1].cost_lower_bound;
            let child1_ub = children[1].cost_upper_bound;
            self.compute_child_upper_bound(context, &mut children[0], child1_lb);
            self.compute_child_upper_bound(context, &mut children[1], child0_lb);
            self.compute_child_lower_bound(context, &mut children[0], child1_ub);
            self.compute_child_lower_bound(context, &mut children[1], child0_ub);
        }

        let lb =
            self.pruner.lb_for(item.feature, &item.split_points) + context.task.branching_cost();
        OT::update_lowerbound(&mut item.cost_lower_bound, &lb);
    }

    /// Reinsert an updated item in the queue after expanding a descendant.
    pub fn backtrack_item(&mut self, context: &SolveContext<OT, SS>, item: QueueItem<'a, OT, SS>) {
        // Logic below depends on only expanded nodes being backtracked.
        assert!(item.split_points.start == item.split_points.end - 1);

        if let Some(expanded) = &item.expanded {
            // No actual LB update here, add the solution to the pruner. The LB of the item will be updated by the pruner later.
            self.pruner.insert_left_subtree(
                item.feature,
                item.split_points.start,
                expanded.lower_bound_left(),
            );
            self.pruner.insert_right_subtree(
                item.feature,
                item.split_points.end - 1,
                expanded.lower_bound_right(),
            );

            // Update our upper bound if based on the upper bound of this item.
            let ub = expanded.upper_bound(context);
            OT::update_upperbound(&mut self.cost_upper_bound, &ub);

            // Update our current best item.
            let best_cost = expanded.best_cost(context);
            if best_cost.strictly_less_than(&self.best.cost()) {
                let expanded = item
                    .expanded
                    .as_ref()
                    .expect("An item can only be backtracked once it is expanded.");

                self.best = match expanded {
                    ExpandedQueueItem::Children(children) => Arc::new(Tree::Branch(BranchNode {
                        cost: best_cost,
                        split_feature: item.feature,
                        split_threshold: self
                            .dataview
                            .threshold_from_split(item.feature, item.split_points.start),
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
        }

        // Reinsert the item in the queue, and ensure the item in the front
        // of the queue has updated lower bounds for the next iteration.
        let mut maybe_item = Some(item);
        while let Some(mut item) = maybe_item.take() {
            self.recalculate_item_bounds(context, &mut item);

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
        path_buffer: &mut VecDeque<(usize, QueueItem<'a, OT, SS>)>,
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
    pub fn expand(&mut self, context: &SolveContext<OT, SS>, item: &mut QueueItem<'a, OT, SS>) {
        assert!(!item.is_expanded());
        assert!(self.remaining_depth_budget > 0);

        let x = self.find_lowest_cost_split(item.feature, &item.split_points);
        assert!(item.split_points.start <= x && x < item.split_points.end);

        let (left_item, right_item) = item.split_at(x);

        if let Some(left_item) = left_item {
            self.queue.push(left_item);
        }
        if let Some(right_item) = right_item {
            self.queue.push(right_item);
        }

        let expanded = if self.remaining_depth_budget == 2
            && context.terminal_solver == TerminalSolver::LeftRight
        {
            solve_left_right(&self.dataview, context, item.feature, x)
        } else {
            self.split(context, item.feature, x)
        };
        item.expanded = Some(expanded);
    }
}
