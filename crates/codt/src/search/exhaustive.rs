use std::sync::Arc;

use crate::{
    model::{
        dataview,
        tree::{BranchNode, LeafNode, Tree},
    },
    search::{node::ExpandedQueueItem, solver_impl::SolveContext, strategy::SearchStrategy},
    tasks::{Cost, CostSum, OptimizationTask},
};

/// Exhaustive search for a node with depth two, given a fixed feature split.
pub fn solve_left_right<'a, OT: OptimizationTask, SS: SearchStrategy>(
    dataview: &dataview::DataView<OT>,
    context: &SolveContext<OT, SS>,
    feature: usize,
    split_index: usize,
) -> ExpandedQueueItem<'a, OT, SS> {
    let split_value = dataview.possible_split_values[feature][split_index].feature_value;
    let mut total_right = dataview.cost_summer.clone();

    for instance in dataview.instances_iter(feature) {
        if dataview.dataset.feature_values[feature][instance] > split_value {
            break;
        } else {
            total_right -= &dataview.dataset.instances[instance];
        }
    }

    let mut total_left = dataview.cost_summer.clone();
    total_left -= &total_right;

    let mut left_tracker = D1ScoreTracker {
        cost: total_left.cost(),
        branching_cost: OT::CostType::ZERO,
        feature: None,
        threshold: None,
        left_leaf: None,
        right_leaf: None,
    };
    let mut right_tracker = D1ScoreTracker {
        cost: total_right.cost(),
        branching_cost: OT::CostType::ZERO,
        feature: None,
        threshold: None,
        left_leaf: None,
        right_leaf: None,
    };

    for feature_2 in 0..dataview.num_features() {
        // init totals of the left left node and right left node to zero.
        let mut total_left_left = total_left.clone();
        total_left_left -= &total_left;
        let mut total_right_left = total_left.clone();
        total_right_left -= &total_left;

        let mut prev_left_feature_value = None;
        let mut prev_right_feature_value = None;

        // Keep costsums out of loop, so that we can .clone_from in the loop and avoid any allocations.
        let mut total_left_right = total_left.clone();
        let mut total_right_right = total_left.clone();

        for instance in dataview.instances_iter(feature_2) {
            let feature1_value = dataview.dataset.feature_values[feature][instance];
            let feature2_value = dataview.dataset.feature_values[feature_2][instance];
            if feature1_value <= split_value {
                // Check if this is a point we can split at
                if let Some(prev_left_feature_value) = prev_left_feature_value {
                    if prev_left_feature_value != feature2_value {
                        total_left_right.clone_from(&total_left);
                        total_left_right -= &total_left_left;

                        let cost_ll = total_left_left.cost();
                        let cost_lr = total_left_right.cost();
                        let cost = cost_ll + cost_lr;
                        let branching_cost = context.task.branching_cost();
                        let total_cost = cost + branching_cost;

                        if total_cost.strictly_less_than(&left_tracker.total_cost()) {
                            let current_threshold =
                                dataview.dataset.internal_to_original_feature_value[feature_2]
                                    [prev_left_feature_value as usize];
                            let next_threshold =
                                dataview.dataset.internal_to_original_feature_value[feature_2]
                                    [feature2_value as usize];

                            left_tracker = D1ScoreTracker {
                                cost,
                                branching_cost,
                                feature: Some(feature_2),
                                threshold: Some((current_threshold + next_threshold) / 2.0),
                                left_leaf: Some(LeafNode {
                                    cost: cost_ll,
                                    label: total_left_left.label(),
                                }),
                                right_leaf: Some(LeafNode {
                                    cost: cost_lr,
                                    label: total_left_right.label(),
                                }),
                            }
                        }
                    }
                }
                total_left_left += &dataview.dataset.instances[instance];
                prev_left_feature_value = Some(feature2_value);
            } else {
                // Check if this is a point we can split at
                if let Some(prev_right_feature_value) = prev_right_feature_value {
                    if prev_right_feature_value != feature2_value {
                        total_right_right.clone_from(&total_right);
                        total_right_right -= &total_right_left;

                        let cost_rl = total_right_left.cost();
                        let cost_rr = total_right_right.cost();
                        let cost = cost_rl + cost_rr;
                        let branching_cost = context.task.branching_cost();
                        let total_cost = cost + branching_cost;

                        if total_cost.strictly_less_than(&right_tracker.total_cost()) {
                            let current_threshold =
                                dataview.dataset.internal_to_original_feature_value[feature_2]
                                    [prev_right_feature_value as usize];
                            let next_threshold =
                                dataview.dataset.internal_to_original_feature_value[feature_2]
                                    [feature2_value as usize];

                            right_tracker = D1ScoreTracker {
                                cost,
                                branching_cost,
                                feature: Some(feature_2),
                                threshold: Some((current_threshold + next_threshold) / 2.0),
                                left_leaf: Some(LeafNode {
                                    cost: cost_rl,
                                    label: total_right_left.label(),
                                }),
                                right_leaf: Some(LeafNode {
                                    cost: cost_rr,
                                    label: total_right_right.label(),
                                }),
                            }
                        }
                    }
                }
                total_right_left += &dataview.dataset.instances[instance];
                prev_right_feature_value = Some(feature2_value);
            }
        }
        if left_tracker.is_optimal(context) && right_tracker.is_optimal(context) {
            break;
        }
    }

    let cost =
        left_tracker.total_cost() + right_tracker.total_cost() + context.task.branching_cost();

    ExpandedQueueItem::Solution(Arc::new(Tree::Branch(BranchNode {
        cost,
        split_feature: feature,
        split_threshold: dataview.threshold_from_split(feature, split_index),
        left_child: left_tracker.get_tree(total_left.label()),
        right_child: right_tracker.get_tree(total_right.label()),
    })))
}

struct D1ScoreTracker<OT: OptimizationTask> {
    cost: OT::CostType,
    branching_cost: OT::CostType,
    feature: Option<usize>,
    threshold: Option<f64>,
    left_leaf: Option<LeafNode<OT>>,
    right_leaf: Option<LeafNode<OT>>,
}

impl<OT: OptimizationTask> D1ScoreTracker<OT> {
    fn is_optimal<SS: SearchStrategy>(&self, context: &SolveContext<'_, OT, SS>) -> bool {
        // If we do branch, then it is optimal if the cost after branching is zero.
        // If we do not branch (yet), this is optimal if branching cannot improve the cost.
        self.cost.is_zero()
            || (self.branching_cost.is_zero()
                && self
                    .cost
                    .less_or_not_much_greater_than(&context.task.branching_cost()))
    }

    fn total_cost(&self) -> OT::CostType {
        self.cost + self.branching_cost
    }

    fn get_tree(self, total_label: OT::LabelType) -> Arc<Tree<OT>> {
        let tree = if self.feature.is_none() {
            Tree::Leaf(LeafNode {
                cost: self.total_cost(),
                label: total_label,
            })
        } else {
            Tree::Branch(BranchNode {
                cost: self.total_cost(),
                split_feature: self.feature.unwrap(),
                split_threshold: self.threshold.unwrap(),
                left_child: Arc::new(Tree::Leaf(self.left_leaf.unwrap())),
                right_child: Arc::new(Tree::Leaf(self.right_leaf.unwrap())),
            })
        };

        Arc::new(tree)
    }
}
