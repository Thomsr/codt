use crate::tasks::OptimizationTask;

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
    pub cost_summer: OT::CostSummer,
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

            // Only allow max of one split (biggest) where left side is OT::MIN_COST
            // TODO: also right side is OT::MIN_COST
            let mut biggest_left_min_cost_split = None;

            for fv in &feature_values_sorted_i {
                costsum += &dataset.instances[fv.instance_id];

                let cost: OT::CostType = (&costsum).into();

                if cost == OT::MIN_COST {
                    if biggest_left_min_cost_split.is_none() {
                        // Reserve a slot at the start for this.
                        possible_split_values_i.push(0);
                    }
                    biggest_left_min_cost_split = Some(fv.feature_value);
                    continue;
                }

                if fv.feature_value != previous {
                    possible_split_values_i.push(fv.feature_value);
                    previous = fv.feature_value;
                }
            }

            if let Some(biggest_left_min_cost_split) = biggest_left_min_cost_split {
                possible_split_values_i[0] = biggest_left_min_cost_split;
            }

            // For each feature, consider only useful splitting points. The last feature value is
            // not a useful splitting point because all instances would go to the left.
            possible_split_values_i.pop();

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

    /// Split this dataview into two, the first containing only those instances where `split_feature <= threshold`, and the second where `split_feature > threshold`.
    pub fn split(&self, split_feature: usize, split_value: usize) -> (Self, Self) {
        let threshold = self.possible_split_values[split_feature][split_value];
        let mut feature_values_left = Vec::with_capacity(self.feature_values_sorted.len());
        let mut feature_values_right = Vec::with_capacity(self.feature_values_sorted.len());
        let mut possible_split_values_left = Vec::with_capacity(self.possible_split_values.len());
        let mut possible_split_values_right = Vec::with_capacity(self.possible_split_values.len());

        let mut left_costsum = self.cost_summer.clone();

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

            for &value in feature {
                if self.dataset.feature_values[split_feature][value.instance_id] <= threshold {
                    // Assure the compiler that we do not need reallocation when pushing.
                    unsafe {
                        std::hint::assert_unchecked(
                            feature_values_left_i.len() < feature_values_left_i.capacity(),
                        );
                        std::hint::assert_unchecked(
                            possible_split_values_left_i.len()
                                < possible_split_values_left_i.capacity(),
                        )
                    }
                    feature_values_left_i.push(value);
                    if value.feature_value != last_feature_value_left {
                        possible_split_values_left_i.push(last_feature_value_left);
                    }
                    last_feature_value_left = value.feature_value;
                } else {
                    // Assure the compiler that we do not need reallocation when pushing.
                    unsafe {
                        std::hint::assert_unchecked(
                            feature_values_right_i.len() < feature_values_right_i.capacity(),
                        );
                        std::hint::assert_unchecked(
                            possible_split_values_right_i.len()
                                < possible_split_values_right_i.capacity(),
                        )
                    }
                    feature_values_right_i.push(value);
                    if value.feature_value != last_feature_value_right {
                        possible_split_values_right_i.push(last_feature_value_right);
                    }
                    last_feature_value_right = value.feature_value;

                    // Only once, subtract all values to the right from the left costsum.
                    if feature_idx == 0 {
                        left_costsum -= &self.dataset.instances[value.instance_id]
                    }
                }
            }

            // For each feature, consider only useful splitting points. The last feature value is
            // not a useful splitting point because all instances would go to the left.
            possible_split_values_left_i.pop();
            possible_split_values_right_i.pop();

            feature_values_left.push(feature_values_left_i);
            feature_values_right.push(feature_values_right_i);
            possible_split_values_left.push(possible_split_values_left_i);
            possible_split_values_right.push(possible_split_values_right_i);
        }

        // Subtract left from the total to get the right costsum
        let mut right_costsum = self.cost_summer.clone();
        right_costsum -= &left_costsum;

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

    pub fn instances_iter(&self) -> impl Iterator<Item = usize> {
        self.feature_values_sorted[0].iter().map(|i| i.instance_id)
    }

    pub fn num_instances(&self) -> usize {
        self.feature_values_sorted[0].len()
    }

    pub fn num_features(&self) -> usize {
        self.feature_values_sorted.len()
    }
}
