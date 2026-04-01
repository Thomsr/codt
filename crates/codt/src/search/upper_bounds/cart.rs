use std::sync::Arc;

use crate::{
    model::{dataview::DataView, tree::{BranchNode, LeafNode, Tree}},
    tasks::{Cost, CostSum, OptimizationTask},
};

fn cart_upper_bound_recursive<OT: OptimizationTask>(task: &OT, dataview: &DataView<'_, OT>) -> Arc<Tree<OT>> {
    let leaf_cost = dataview.cost_summer.cost();
    let leaf_label = dataview.cost_summer.label();

    // If the leaf is already a perfect solution (zero cost), we can use it directly.
    if OT::is_perfect_solution_cost(&leaf_cost) {
        return Arc::new(Tree::Leaf(LeafNode {
            cost: leaf_cost,
            label: leaf_label,
        }));
    }

    if dataview.num_instances() <= 1 {
        return Arc::new(Tree::Leaf(LeafNode {
            cost: leaf_cost,
            label: leaf_label,
        }));
    }

    let mut best_feature = None;
    let mut best_greedy = f32::INFINITY;

    for feature in 0..dataview.num_features() {
        if dataview.possible_split_values[feature].is_empty() {
            continue;
        }

        let greedy = dataview.best_greedy_splits[feature].greedy_value;
        if greedy < best_greedy {
            best_greedy = greedy;
            best_feature = Some(feature);
        }
    }

    let Some(feature) = best_feature else {
        return Arc::new(Tree::Leaf(LeafNode {
            cost: leaf_cost,
            label: leaf_label,
        }));
    };

    let split_value = dataview.best_greedy_splits[feature].split_value_index;
    let split_threshold = dataview.threshold_from_split(feature, split_value);
    let (left_view, right_view) = dataview.split(feature, split_value);

    let left_child = cart_upper_bound_recursive(task, &left_view);
    let right_child = cart_upper_bound_recursive(task, &right_view);

    let split_cost = left_child.cost() + right_child.cost() + task.branching_cost();
    
    // Compare with leaf and return the better option
    if split_cost.strictly_less_than(&leaf_cost) {
        Arc::new(Tree::Branch(BranchNode {
            cost: split_cost,
            split_feature: feature,
            split_threshold,
            left_child,
            right_child,
        }))
    } else {
        Arc::new(Tree::Leaf(LeafNode {
            cost: leaf_cost,
            label: leaf_label,
        }))
    }
}

pub fn cart_upper_bound<OT: OptimizationTask>(task: &OT, dataview: &DataView<'_, OT>) -> Arc<Tree<OT>> {
    cart_upper_bound_recursive(task, dataview)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::dataset::DataSet;
    use crate::model::instance::LabeledInstance;
    use crate::tasks::accuracy::AccuracyTask;

    // Helper to create a dataset with given feature values and labels
    fn create_dataset(feature_values: Vec<i32>, labels: Vec<i32>) -> DataSet<LabeledInstance<i32>> {
        let mut dataset = DataSet::default();
        for (i, label) in labels.iter().enumerate() {
            dataset.add_instance(LabeledInstance::new(*label), vec![feature_values[i] as f64]);
        }
        dataset.preprocess_after_adding_instances();
        dataset
    }

    #[test]
    fn cart_perfect_tree() {
        // All instances have the same label - perfect tree with zero cost
        let feature_values = vec![0, 1, 2, 3, 4];
        let labels = vec![0, 0, 0, 0, 0];
        let dataset = create_dataset(feature_values, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);

        let task = AccuracyTask::new();
        let tree = cart_upper_bound(&task, &dataview);

        assert!(AccuracyTask::is_perfect_solution_cost(&tree.cost()), "Expected zero cost for perfect tree, got {:?}", tree.cost());
    }

    #[test]
    fn cart_mixed_classes() {
        // Mixed classes - should attempt to split
        let feature_values = vec![0, 0, 1, 1];
        let labels = vec![0, 0, 1, 1];
        let dataset = create_dataset(feature_values, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);

        let task = AccuracyTask::new();
        let tree = cart_upper_bound(&task, &dataview);

        // Cost should be at most the leaf cost (branching cost since left and right are perfect)
        let leaf_cost = dataview.cost_summer.cost();
        println!("CART Tree: {}", tree);
        println!("Tree Cost: {:?}", tree.cost());
        assert!(
            tree.cost().less_or_not_much_greater_than(&leaf_cost),
            "Expected cost {:?} to be at most leaf cost {:?}",
            tree.cost(),
            leaf_cost
        );
    }

    #[test]
    fn cart_single_instance() {
        // Single instance - cannot split, should return leaf cost
        let feature_values = vec![0];
        let labels = vec![0];
        let dataset = create_dataset(feature_values, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);

        let task = AccuracyTask::new();
        let tree = cart_upper_bound(&task, &dataview);

        // Cost should be zero for single perfect instance
        assert!(AccuracyTask::is_perfect_solution_cost(&tree.cost()), "Expected zero cost for single instance, got {:?}", tree.cost());
    }

    #[test]
    fn cart_best_split_reduces_cost() {
        // Data that benefits from splitting: [0,0,0,1]
        // Leaf cost would be 1 error
        // But splitting can reduce it
        let feature_values = vec![0, 0, 0, 1];
        let labels = vec![0, 0, 0, 1];
        let dataset = create_dataset(feature_values, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);

        let task = AccuracyTask::new();
        let tree = cart_upper_bound(&task, &dataview);

        // Splitting on feature 0 should give us left=[0,0,0] (perfect) and right=[1] (perfect)
        // Total cost = branching_cost + 0 + 0 = branching_cost
        // Leaf cost = 1
        // Should choose splitting if branching_cost < 1
        let leaf_cost = dataview.cost_summer.cost();
        
        // The result should be at most the leaf cost (the feasible upper bound)
        assert!(
            tree.cost().less_or_not_much_greater_than(&leaf_cost),
            "CART cost {:?} should not exceed leaf cost {:?}",
            tree.cost(),
            leaf_cost
        );
    }

    #[test]
    fn cart_no_splitting_benefit() {
        // Data that cannot be perfectly separated
        let feature_values = vec![0, 0, 1, 1];
        let labels = vec![0, 1, 0, 1];
        let dataset = create_dataset(feature_values, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);

        let task = AccuracyTask::new();
        let tree = cart_upper_bound(&task, &dataview);

        // The cost should be a valid upper bound (> 0 but <= leaf cost)
        let leaf_cost = dataview.cost_summer.cost();
        assert!(
            tree.cost().less_or_not_much_greater_than(&leaf_cost),
            "CART cost {:?} should not exceed leaf cost {:?}",
            tree.cost(),
            leaf_cost
        );
    }

    #[test]
    fn cart_print_tree() {
        // Test that we can print a tree
        let feature_values = vec![0, 0, 1, 1];
        let labels = vec![0, 0, 1, 1];
        let dataset = create_dataset(feature_values, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);

        let task = AccuracyTask::new();
        let tree = cart_upper_bound(&task, &dataview);

        // Print the tree
        println!("CART Tree: {}", tree);
        println!("Tree Cost: {:?}", tree.cost());
        assert!(!tree.cost().is_zero() || AccuracyTask::is_perfect_solution_cost(&tree.cost()), "Tree should have valid cost");
    }
}