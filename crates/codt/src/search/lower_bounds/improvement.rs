use crate::{model::difference_table::DifferenceTableView, tasks::OptimizationTask};

pub fn improvement_lower_bound_with_feature_counts<OT: OptimizationTask>(
    diff_table: &DifferenceTableView<'_>,
) -> (OT::CostType, Vec<usize>) {
    if diff_table.is_empty() {
        return (OT::to_cost_type(0), Vec::new());
    }

    let (improvement_lower_bound, feature_counts) =
        diff_table.size_based_cover_feature_counts(diff_table.n_rows());

    (
        OT::to_cost_type(improvement_lower_bound as i64),
        feature_counts,
    )
}

#[cfg(test)]
mod tests {
    use crate::{
        model::difference_table::DifferenceTable,
        model::{dataset::DataSet, dataview::DataView, instance::LabeledInstance},
        search::lower_bounds::{
            improvement::improvement_lower_bound_with_feature_counts, pair::pair_lower_bound,
        },
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
        let difference_table = DifferenceTable::new(&dataview);
        let view = difference_table.view();

        let lb = improvement_lower_bound_with_feature_counts::<AccuracyTask>(&view).0;
        assert_eq!(lb.secondary, 0);
    }

    #[test]
    fn single_feature_sufficient() {
        // One feature perfectly separates classes, LB = 1
        let features = vec![vec![0, 1, 2, 3]];
        let labels = vec![0, 0, 1, 1];

        let dataset = create_dataset(features, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset, false);
        let difference_table = DifferenceTable::new(&dataview);
        let view = difference_table.view();

        let lb = improvement_lower_bound_with_feature_counts::<AccuracyTask>(&view).0;
        assert_eq!(lb.secondary, 1);
    }

    #[test]
    fn improvement_looser_than_pair() {
        let features = vec![vec![0, 1, 2, 3], vec![0, 1, 2, 3]];
        let labels = vec![0, 1, 0, 1];

        let dataset = create_dataset(features, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset, false);
        let difference_table = DifferenceTable::new(&dataview);
        let view = difference_table.view();
        let improvement_lower_bound =
            improvement_lower_bound_with_feature_counts::<AccuracyTask>(&view).0;
        let pair_lower_bound = pair_lower_bound::<AccuracyTask>(&view);

        assert!(
            improvement_lower_bound.less_or_not_much_greater_than(&pair_lower_bound),
            "Improvement LB should be less than Pair LB"
        );
    }
}
