use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};
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
    use_subset: Option<bool>,
    rng: &mut StdRng,
) -> Arc<Tree<OT>> {
    let use_subset = use_subset.unwrap_or(true);

    let leaf_cost = dataview.cost_summer.cost();
    let leaf_label = dataview.cost_summer.label();

    // If the leaf is already a perfect solution (zero cost), we can use it directly.
    if OT::is_perfect_solution_cost(&leaf_cost) || dataview.num_instances() <= 1 {
        return Arc::new(Tree::Leaf(LeafNode {
            cost: leaf_cost,
            label: leaf_label,
        }));
    }

    let mut feature_indices: Vec<usize> = (0..dataview.possible_split_values.len()).collect();
    feature_indices.shuffle(rng);

    if use_subset {
        let n_features = dataview.possible_split_values.len() as f64;
        let subset_size = n_features.sqrt().ceil() as usize;
        feature_indices.truncate(subset_size);
    }

    let candidate = feature_indices
        .iter()
        .filter_map(|&feature_idx| {
            dataview.possible_split_values[feature_idx]
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.greedy_value.total_cmp(&b.greedy_value))
                .map(|(split_value_index, split)| {
                    (feature_idx, split_value_index, split.greedy_value)
                })
        })
        .min_by(|(_, _, a), (_, _, b)| a.total_cmp(b));

    let Some((best_feature, best_split_value_index, _)) = candidate else {
        return Arc::new(Tree::Leaf(LeafNode {
            cost: leaf_cost,
            label: leaf_label,
        }));
    };

    let (left_view, right_view) = dataview.split(best_feature, best_split_value_index);

    let left_child = cart_upper_bound_recursive(task, &left_view, Some(use_subset), rng);
    let right_child = cart_upper_bound_recursive(task, &right_view, Some(use_subset), rng);

    let split_cost = left_child.cost() + right_child.cost() + task.branching_cost();

    Arc::new(Tree::Branch(BranchNode {
        cost: split_cost,
        split_feature: dataview
            .original_split_feature_from_split(best_feature, best_split_value_index),
        split_threshold: dataview.threshold_from_split(best_feature, best_split_value_index),
        left_child,
        right_child,
    }))
}

/// Computes an upper bound on the optimal tree cost using a CART-like greedy approach.
/// Uses all features by default, matching standard CART behavior.
pub fn cart_upper_bound<OT: OptimizationTask>(
    task: &OT,
    dataview: &DataView<'_, OT>,
) -> Arc<Tree<OT>> {
    let mut rng = StdRng::seed_from_u64(42);
    cart_upper_bound_recursive(task, dataview, Some(false), &mut rng)
}

/// Computes an upper bound on the optimal tree cost using a CART-like greedy approach.
/// `use_subset` is true by default.
pub fn cart_upper_bound_with_subset<OT: OptimizationTask>(
    task: &OT,
    dataview: &DataView<'_, OT>,
    use_subset: Option<bool>,
) -> Arc<Tree<OT>> {
    cart_upper_bound_with_subset_seed(task, dataview, use_subset, 42)
}

/// Computes an upper bound on the optimal tree cost using a CART-like greedy approach.
/// Uses an explicit RNG seed, useful for repeated stochastic runs.
pub fn cart_upper_bound_with_subset_seed<OT: OptimizationTask>(
    task: &OT,
    dataview: &DataView<'_, OT>,
    use_subset: Option<bool>,
    seed: u64,
) -> Arc<Tree<OT>> {
    let mut rng = StdRng::seed_from_u64(seed);
    cart_upper_bound_recursive(task, dataview, use_subset, &mut rng)
}

#[cfg(test)]
mod tests {
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
    use crate::test_support::{read_from_file, repo_root};
    use dataset_by_difficulty::DATASETS_BY_DIFFICULTY;

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
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset, false);

        let task = AccuracyTask::new();
        let tree = cart_upper_bound_with_subset(&task, &dataview, Some(false));

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
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset, false);

        let task = AccuracyTask::new();
        let tree = cart_upper_bound_with_subset(&task, &dataview, Some(false));

        println!("CART Tree: {}", tree);

        assert!(
            AccuracyTask::is_perfect_solution_cost(&tree.cost()),
            "Expected zero cost for perfect tree, got {:?}",
            tree.cost()
        );
    }

    #[test]
    fn cart_on_datasets() {
        let datasets = DATASETS_BY_DIFFICULTY;

        for datasets in datasets {
            // for dataset_name in datasets
            let mut dataset = DataSet::default();
            let sampled_name = datasets
                .strip_suffix(".csv")
                .expect("Expected dataset names to end with .csv");
            let file = repo_root()
                .join("data/sampled")
                .join(format!("{sampled_name}.txt"));
            read_from_file(&mut dataset, &file).unwrap();
            let dataview = DataView::<AccuracyTask>::from_dataset(&dataset, false);

            let task = AccuracyTask::new();
            let tree = cart_upper_bound_with_subset(&task, &dataview, Some(false));

            println!(
                "CART Tree for {} instances {}: {}",
                datasets,
                dataset.num_instances(),
                tree.cost()
            );

            assert!(
                tree.cost()
                    .less_or_not_much_greater_than(&dataview.cost_summer.cost()),
                "Expected CART upper bound to be no worse than leaf for {}, got {:?} vs {:?}",
                datasets,
                tree.cost(),
                dataview.cost_summer.cost()
            );

            assert!(
                AccuracyTask::is_perfect_solution_cost(&tree.cost()),
                "Expected CART to find a perfect tree for {}, got cost {:?}",
                datasets,
                tree.cost()
            );
        }
    }
}
