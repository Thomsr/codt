use crate::{
    model::{dataview::DataView, difference_table::DifferenceTable},
    tasks::OptimizationTask,
};

pub fn improvement_lower_bound<OT: OptimizationTask>(dataview: &DataView<'_, OT>) -> OT::CostType {
    let diff_table = DifferenceTable::new(dataview);

    if diff_table.diffs.is_empty() {
        return OT::to_cost_type(0);
    }

    let n_pairs = diff_table.pairs.len();
    let improvement_lower_bound = diff_table.min_size_based_cover(n_pairs);

    OT::to_cost_type(improvement_lower_bound as i64)
}

#[cfg(test)]
mod tests {
    use crate::{
        model::{dataset::DataSet, dataview::DataView, instance::LabeledInstance},
        search::lower_bounds::{improvement::improvement_lower_bound, pair::pair_lower_bound},
        tasks::{Cost, accuracy::AccuracyTask},
    };

    fn create_dataset(features: Vec<Vec<i32>>, labels: Vec<i32>) -> DataSet<LabeledInstance<i32>> {
        let mut dataset = DataSet::default();

        for label in labels {
            dataset.instances.push(LabeledInstance::new(label));
        }

        for feature_col in features {
            dataset.feature_values.push(feature_col);
        }

        dataset
    }

    #[test]
    fn no_conflicts() {
        // All labels identical, no conflicting pairs, LB = 0
        let features = vec![vec![0, 1, 2, 3]];
        let labels = vec![1, 1, 1, 1];

        let dataset = create_dataset(features, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset, false);

        let lb = improvement_lower_bound(&dataview);
        assert_eq!(lb.secondary, 0);
    }

    #[test]
    fn single_feature_sufficient() {
        // One feature perfectly separates classes, LB = 1
        let features = vec![vec![0, 1, 2, 3]];
        let labels = vec![0, 0, 1, 1];

        let dataset = create_dataset(features, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset, false);

        let lb = improvement_lower_bound(&dataview);
        assert_eq!(lb.secondary, 1);
    }

    #[test]
    fn improvement_looser_than_pair() {
        let features = vec![vec![0, 1, 2, 3], vec![0, 1, 2, 3]];
        let labels = vec![0, 1, 0, 1];

        let dataset = create_dataset(features, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset, false);
        let improvement_lower_bound = improvement_lower_bound(&dataview);
        let pair_lower_bound = pair_lower_bound(&dataview);

        assert!(
            improvement_lower_bound.less_or_not_much_greater_than(&pair_lower_bound),
            "Improvement LB should be less than Pair LB"
        );
    }
}
