use std::sync::Arc;

use crate::{
    model::{
        dataview::DataView,
        tree::{BranchNode, LeafNode, Tree},
    },
    tasks::{CostSum, OptimizationTask},
};

fn cart_upper_bound_recursive<OT: OptimizationTask>(
    task: &OT,
    dataview: &DataView<'_, OT>,
) -> Arc<Tree<OT>> {
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

    let mut best_split = None;
    let mut best_greedy = f32::INFINITY;

    for (feature, split_values) in dataview.possible_split_values.iter().enumerate() {
        for (split_value, split) in split_values.iter().enumerate() {
            if split.greedy_value < best_greedy {
                best_greedy = split.greedy_value;
                best_split = Some((feature, split_value));
            }
        }
    }

    let Some((feature, split_value)) = best_split else {
        return Arc::new(Tree::Leaf(LeafNode {
            cost: leaf_cost,
            label: leaf_label,
        }));
    };

    let split_threshold = dataview.threshold_from_split(feature, split_value);
    let (left_view, right_view) = dataview.split(feature, split_value);

    let left_child = cart_upper_bound_recursive(task, &left_view);
    let right_child = cart_upper_bound_recursive(task, &right_view);

    let split_cost = left_child.cost() + right_child.cost() + task.branching_cost();

    Arc::new(Tree::Branch(BranchNode {
        cost: split_cost,
        split_feature: feature,
        split_threshold,
        left_child,
        right_child,
    }))
}

pub fn cart_upper_bound<OT: OptimizationTask>(
    task: &OT,
    dataview: &DataView<'_, OT>,
) -> Arc<Tree<OT>> {
    cart_upper_bound_recursive(task, dataview)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    mod dataset_by_difficulty {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/dataset-by-difficulty.rs"
        ));
    }

    use super::*;
    use crate::model::dataset::DataSet;
    use crate::model::instance::LabeledInstance;
    use crate::tasks::Cost;
    use crate::tasks::accuracy::AccuracyTask;
    use dataset_by_difficulty::DATASETS_BY_DIFFICULTY;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../")
    }

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
    fn cart_small_perfect_tree() {
        // All instances have the same label - perfect tree with zero cost
        let feature_values = vec![0, 1, 2, 3, 4];
        let labels = vec![0, 0, 0, 0, 0];
        let dataset = create_dataset(feature_values, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);

        let task = AccuracyTask::new();
        let tree = cart_upper_bound(&task, &dataview);

        println!("CART Tree: {}", tree);

        assert!(
            AccuracyTask::is_perfect_solution_cost(&tree.cost()),
            "Expected zero cost for perfect tree, got {:?}",
            tree.cost()
        );
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

        println!("CART Tree: {}", tree);

        assert!(
            AccuracyTask::is_perfect_solution_cost(&tree.cost()),
            "Expected zero cost for perfect tree, got {:?}",
            tree.cost()
        );
    }

    #[test]
    fn cart_top10_easy_sampled_datasets() {
        let datasets = DATASETS_BY_DIFFICULTY;

        for dataset_name in datasets {
            let dataset =
                DataSet::from_csv(&repo_root().join("data/normal/sampled").join(dataset_name));
            let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);

            let task = AccuracyTask::new();
            let tree = cart_upper_bound(&task, &dataview);

            println!(
                "CART Tree for {} instances {}: {}",
                dataset_name,
                dataset.num_instances(),
                tree.cost()
            );

            assert!(
                tree.cost()
                    .less_or_not_much_greater_than(&dataview.cost_summer.cost()),
                "Expected CART upper bound to be no worse than leaf for {}, got {:?} vs {:?}",
                dataset_name,
                tree.cost(),
                dataview.cost_summer.cost()
            );

            assert!(
                AccuracyTask::is_perfect_solution_cost(&tree.cost()),
                "Expected CART to find a perfect tree for {}, got cost {:?}",
                dataset_name,
                tree.cost()
            )
        }
    }
}
