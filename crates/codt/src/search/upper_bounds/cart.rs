use std::sync::Arc;

use crate::{
    model::{dataview::DataView, tree::{BranchNode, LeafNode, Tree}},
    tasks::{CostSum, OptimizationTask},
};

fn safe_threshold_from_split<OT: OptimizationTask>(
    dataview: &DataView<'_, OT>,
    split_feature: usize,
    split_value: usize,
) -> f64 {
    let split_values = &dataview.possible_split_values[split_feature];
    if split_values.is_empty() {
        return 0.0;
    }

    let split_idx = split_value.min(split_values.len() - 1);
    let current_internal = split_values[split_idx].feature_value.max(0) as usize;
    let next_internal = split_values
        .get(split_idx + 1)
        .map(|s| s.feature_value.max(0) as usize)
        .unwrap_or(current_internal);

    let mapping = &dataview.dataset.internal_to_original_feature_value[split_feature];
    if mapping.is_empty() {
        return split_values[split_idx].feature_value as f64;
    }

    let last = mapping.len() - 1;
    let current = mapping[current_internal.min(last)];
    let next = mapping[next_internal.min(last)];
    (current + next) / 2.0
}

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

    let split_value = dataview.best_greedy_splits[feature]
        .split_value_index
        .min(dataview.possible_split_values[feature].len() - 1);
    let split_threshold = safe_threshold_from_split(dataview, feature, split_value);
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

pub fn cart_upper_bound<OT: OptimizationTask>(task: &OT, dataview: &DataView<'_, OT>) -> Arc<Tree<OT>> {
    cart_upper_bound_recursive(task, dataview)
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    mod dataset_by_difficulty {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/dataset-by-difficulty.rs"));
    }

    use super::*;
    use crate::model::dataset::DataSet;
    use crate::model::instance::LabeledInstance;
    use crate::tasks::accuracy::AccuracyTask;
    use crate::tasks::Cost;
    use dataset_by_difficulty::DATASETS_BY_DIFFICULTY;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../")
    }

    fn load_sampled_csv_dataset(name: &str) -> DataSet<LabeledInstance<i32>> {
        let path = repo_root().join("data/normal/sampled").join(name);
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read dataset {}: {}", path.display(), e));

        let mut dataset = DataSet::<LabeledInstance<i32>>::default();
        for (line_idx, line) in content.lines().enumerate() {
            if line_idx == 0 || line.trim().is_empty() {
                continue;
            }

            let cols: Vec<&str> = line.split(',').collect();
            assert!(
                cols.len() >= 2,
                "Expected at least one feature and one label in {}",
                path.display()
            );

            let features = cols[..cols.len() - 1]
                .iter()
                .map(|v| v.parse::<f64>().expect("Feature should be a float"));
            let label = cols[cols.len() - 1]
                .parse::<i32>()
                .expect("Label should be an integer class");

            dataset.add_instance(LabeledInstance::new(label), features);
        }

        dataset.preprocess_after_adding_instances();
        dataset
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

        println!("CART Tree: {}", tree);

        assert!(AccuracyTask::is_perfect_solution_cost(&tree.cost()), "Expected zero cost for perfect tree, got {:?}", tree.cost());
    }

    #[test]
    fn cart_top10_easy_sampled_datasets() {
        let datasets = DATASETS_BY_DIFFICULTY;

        for dataset_name in datasets {
            let dataset = load_sampled_csv_dataset(dataset_name);
            let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);

            let task = AccuracyTask::new();
            let tree = cart_upper_bound(&task, &dataview);

            assert!(
                tree.cost().less_or_not_much_greater_than(&dataview.cost_summer.cost()),
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