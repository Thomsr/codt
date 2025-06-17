use crate::tasks::{CostSum, OptimizationTask};

use super::dataset::DataSet;

use std::fmt::Debug;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FeatureValue {
    /// The id of the instance that has this feature value
    pub instance_id: usize,

    /// Float feature values are substituted by subsequent integer values such that
    /// `x.feature_value < y.feature_value` iff `x.original_feature_value < y.original_feature_value`
    /// (non-unique values get the same int value).
    pub feature_value: i32,
}

pub struct DataView<'a, OT: OptimizationTask> {
    /// This struct is a view over this dataset.
    pub dataset: &'a DataSet<OT::InstanceType>,
    /// The feature values for instances that remain in this view. Indexed first
    /// by feature_id, then sorted by feature value.
    feature_values_sorted: Vec<Vec<FeatureValue>>,
    /// All of the feature values that are still possible to split on per feature.
    /// A reduced set of all unique values of `feature_values_sorted`.
    pub possible_split_values: Vec<Vec<i32>>,
    pub cost_summer: OT::CostSumType,
}

impl<OT: OptimizationTask> Debug for DataView<'_, OT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("DataView");
        for feature in &self.feature_values_sorted {
            d.field("feature_values_sorted", &feature.len());
        }
        d.finish()
    }
}

impl<'a, OT: OptimizationTask> DataView<'a, OT> {
    fn add_possible_split_value(
        possible_split_values: &mut Vec<i32>,
        last_feature_value: &mut i32,
        value: &FeatureValue,
        costsum: &mut OT::CostSumType,
        last_left_cost: &mut OT::CostType,
        keep_until: &mut usize,
        dataset: &'a DataSet<OT::InstanceType>,
    ) {
        if value.feature_value != *last_feature_value {
            // Only allow max of one split (biggest) where left side is OT::MIN_COST
            // For feature values:         00011112224444566
            // And cumulative left cost:   00000000123456666
            // And cumulative right cost:  66666666543210000
            // We add possible splits:     0--1---2--4---56-
            //
            // When adding split 2, we see that the old_cost is 0, so
            // split 1 is a bigger zero split than split 0./
            //
            // Similarly, split 4 is a bigger right zero split than 5 and 6.
            // However, we do not know the cumulative right cost yet,
            // so we decide to keep it if the left cost stays the same. (Max left cost = min right cost)

            let old_cost: OT::CostType = costsum.cost();

            // Keep track of the first split with the last left cost, so later we can
            // remove all splits after that if the left cost doesn't change anymore.
            if old_cost != *last_left_cost {
                *keep_until = possible_split_values.len();
                *last_left_cost = old_cost;
            }

            // Ensure we have at most one split with zero cost on the left side.
            if old_cost == OT::ZERO_COST && possible_split_values.len() > 1 {
                possible_split_values[0] = possible_split_values[1];
                possible_split_values[1] = value.feature_value;
            } else {
                possible_split_values.push(value.feature_value);
            }
            *last_feature_value = value.feature_value;
        }
        *costsum += &dataset.instances[value.instance_id];
    }

    fn post_process_possible_splits(
        costsum: &mut OT::CostSumType,
        possible_split_values: &mut Vec<i32>,
        last_left_cost: &mut OT::CostType,
        keep_until: &mut usize,
    ) {
        let full_cost = costsum.cost();

        if full_cost == OT::ZERO_COST {
            *keep_until = 0;
        } else if full_cost != *last_left_cost {
            // For each feature, consider only useful splitting points. The last
            // feature value is not a useful splitting point because all instances
            // would go to the left. So only keep len - 1.
            *keep_until = possible_split_values.len() - 1;
        }

        possible_split_values.truncate(*keep_until);
    }

    /// Initialize a dataview from a dataset. The new dataview contains all instances of the dataset
    pub fn from_dataset(dataset: &'a DataSet<OT::InstanceType>) -> Self {
        // Copy over all the feature values from the dataset, and sort them by the feature values.
        let mut feature_values_sorted = Vec::new();
        let mut possible_split_values = Vec::new();

        let mut total_cost = None;

        for feature in &dataset.feature_values {
            let mut feature_values_sorted_i = Vec::new();
            let mut possible_split_values_i = Vec::new();

            for (instance_id, &feature_value) in feature.iter().enumerate() {
                feature_values_sorted_i.push(FeatureValue {
                    instance_id,
                    feature_value,
                })
            }

            feature_values_sorted_i.sort_by_key(|fv| fv.feature_value);

            let mut costsum = OT::init_costsum(dataset);
            let mut previous = -1;
            let mut last_left_cost = OT::ZERO_COST;
            let mut keep_until = 1;

            for fv in &feature_values_sorted_i {
                Self::add_possible_split_value(
                    &mut possible_split_values_i,
                    &mut previous,
                    fv,
                    &mut costsum,
                    &mut last_left_cost,
                    &mut keep_until,
                    dataset,
                );
            }

            Self::post_process_possible_splits(
                &mut costsum,
                &mut possible_split_values_i,
                &mut last_left_cost,
                &mut keep_until,
            );

            feature_values_sorted.push(feature_values_sorted_i);
            possible_split_values.push(possible_split_values_i);
            total_cost = Some(costsum); // Any loop iteration works
        }

        Self {
            dataset,
            feature_values_sorted,
            possible_split_values,
            cost_summer: total_cost.expect("No features in dataset"),
        }
    }

    /// Helper function for left and right side.
    #[inline]
    fn add_feature_value(feature_values: &mut Vec<FeatureValue>, value: FeatureValue) {
        // Assure the compiler that we do not need reallocation when pushing.
        unsafe {
            std::hint::assert_unchecked(feature_values.len() < feature_values.capacity());
        }
        feature_values.push(value);
    }

    /// Split this dataview into two, the first containing only those instances where
    /// `split_feature <= threshold`, and the second where `split_feature > threshold`.
    pub fn split(&self, split_feature: usize, split_value: usize) -> (Self, Self) {
        let threshold = self.possible_split_values[split_feature][split_value];
        let mut feature_values_left = Vec::with_capacity(self.feature_values_sorted.len());
        let mut feature_values_right = Vec::with_capacity(self.feature_values_sorted.len());
        let mut possible_split_values_left = Vec::with_capacity(self.possible_split_values.len());
        let mut possible_split_values_right = Vec::with_capacity(self.possible_split_values.len());

        let mut left_costsum = self.cost_summer.clone();
        left_costsum.clear();
        let mut right_costsum = left_costsum.clone();

        for (feature_idx, feature) in self.feature_values_sorted.iter().enumerate() {
            // Overestimate all of these capacities to the current length so that no reallocations are needed.
            let mut feature_values_left_i = Vec::with_capacity(feature.len());
            let mut feature_values_right_i = Vec::with_capacity(feature.len());
            let mut possible_split_values_left_i =
                Vec::with_capacity(self.possible_split_values[feature_idx].len());
            let mut possible_split_values_right_i =
                Vec::with_capacity(self.possible_split_values[feature_idx].len());
            let mut last_feature_value_left = -1;
            let mut last_feature_value_right = -1;
            let mut last_left_left_cost = OT::ZERO_COST;
            let mut last_right_left_cost = OT::ZERO_COST;
            let mut keep_until_left = 1;
            let mut keep_until_right = 1;
            left_costsum.clear();
            right_costsum.clear();

            for &value in feature {
                if self.dataset.feature_values[split_feature][value.instance_id] <= threshold {
                    Self::add_feature_value(&mut feature_values_left_i, value);
                    Self::add_possible_split_value(
                        &mut possible_split_values_left_i,
                        &mut last_feature_value_left,
                        &value,
                        &mut left_costsum,
                        &mut last_left_left_cost,
                        &mut keep_until_left,
                        self.dataset,
                    );
                } else {
                    Self::add_feature_value(&mut feature_values_right_i, value);
                    Self::add_possible_split_value(
                        &mut possible_split_values_right_i,
                        &mut last_feature_value_right,
                        &value,
                        &mut right_costsum,
                        &mut last_right_left_cost,
                        &mut keep_until_right,
                        self.dataset,
                    );
                }
            }

            Self::post_process_possible_splits(
                &mut left_costsum,
                &mut possible_split_values_left_i,
                &mut last_left_left_cost,
                &mut keep_until_left,
            );

            Self::post_process_possible_splits(
                &mut right_costsum,
                &mut possible_split_values_right_i,
                &mut last_right_left_cost,
                &mut keep_until_right,
            );

            assert!(!feature_values_left_i.is_empty());
            assert!(!feature_values_right_i.is_empty());

            feature_values_left.push(feature_values_left_i);
            feature_values_right.push(feature_values_right_i);
            possible_split_values_left.push(possible_split_values_left_i);
            possible_split_values_right.push(possible_split_values_right_i);
        }

        (
            Self {
                dataset: self.dataset,
                feature_values_sorted: feature_values_left,
                possible_split_values: possible_split_values_left,
                cost_summer: left_costsum,
            },
            Self {
                dataset: self.dataset,
                feature_values_sorted: feature_values_right,
                possible_split_values: possible_split_values_right,
                cost_summer: right_costsum,
            },
        )
    }

    /// Iterate all instances, sorted by its feature value of a specific feature.
    pub fn instances_iter(&self, feature: usize) -> impl Iterator<Item = usize> {
        self.feature_values_sorted[feature]
            .iter()
            .map(|i| i.instance_id)
    }

    pub fn num_instances(&self) -> usize {
        self.feature_values_sorted[0].len()
    }

    pub fn num_features(&self) -> usize {
        self.feature_values_sorted.len()
    }

    pub fn threshold_from_split(&self, split_feature: usize, split_value: usize) -> f64 {
        let current_split_value = self.possible_split_values[split_feature][split_value];
        let next_split_value = match self.possible_split_values[split_feature].get(split_value + 1) {
            Some(&next_split_value) => next_split_value,
            None => self.feature_values_sorted[split_feature]
                .last()
                .expect("There is at least one threshold remaining, otherwise there would be no useful values to split on, and this method would never be called.")
                .feature_value
        };

        let current_threshold = self.dataset.internal_to_original_feature_value[split_feature]
            [current_split_value as usize];
        let next_threshold = self.dataset.internal_to_original_feature_value[split_feature]
            [next_split_value as usize];
        (current_threshold + next_threshold) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::instance::LabeledInstance;
    use crate::tasks::accuracy::AccuracyTask;

    /// Helper function to test `add_possible_split_value` and `post_process_possible_splits`.
    fn test_possible_splits(feature_values: Vec<i32>, labels: Vec<i32>, expected_splits: Vec<i32>) {
        // Create a dataset with the given labels and feature values.
        let mut dataset = DataSet::default();
        for label in labels {
            dataset.instances.push(LabeledInstance::new(label));
        }
        dataset.feature_values.push(feature_values);

        let view = DataView::<AccuracyTask>::from_dataset(&dataset);

        // Assert that the resulting splits match the expected splits.
        assert_eq!(view.possible_split_values[0], expected_splits);
    }

    #[test]
    fn possible_splits_smoke_test() {
        // For feature values:         00011112224444566
        // And cumulative left cost:   00000000123456666
        // And cumulative right cost:  66666666543210000
        // We add possible splits:     0--1---2--4---56-
        // - 0, 1 is a bigger zero split.
        // + 1
        // + 2
        // - 3, feature value not in this range.
        // + 4
        // - 5, 4 is a bigger right zero split.
        // - 6, 4 is a bigger right zero split.
        let feature_values = vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4, 5, 6, 6];
        let labels = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0];
        let expected_splits = vec![1, 2, 4];
        test_possible_splits(feature_values, labels, expected_splits);
    }

    #[test]
    fn possible_splits_no_last() {
        let feature_values = vec![0, 1, 2];
        let labels = vec![0, 1, 2];
        let expected_splits = vec![0, 1];
        test_possible_splits(feature_values, labels, expected_splits);
    }

    #[test]
    fn possible_splits_one() {
        let feature_values = vec![0, 1];
        let labels = vec![0, 1];
        let expected_splits = vec![0];
        test_possible_splits(feature_values, labels, expected_splits);
    }

    #[test]
    fn possible_splits_none() {
        let feature_values = vec![0, 1];
        let labels = vec![0, 0];
        let expected_splits = vec![];
        test_possible_splits(feature_values, labels, expected_splits);
    }

    #[test]
    fn possible_splits_largest_left() {
        let feature_values = vec![0, 1, 2, 3, 4, 5];
        let labels = vec![0, 0, 0, 0, 0, 1];
        let expected_splits = vec![4];
        test_possible_splits(feature_values, labels, expected_splits);
    }
}
