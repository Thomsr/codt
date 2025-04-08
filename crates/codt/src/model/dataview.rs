use super::{dataset::DataSet, instance::Instance};

use std::{cmp::Ordering, fmt::Debug, ops::Range};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FeatureValue {
    /// The id of the instance that has this feature value
    pub instance_id: usize,

    /// Float feature values are substituted by subsequent integer values such that
    /// `x.feature_value < y.feature_value` iff `x.original_feature_value < y.original_feature_value`
    /// (non-unique values get the same int value).
    pub feature_value: i32,
}

pub struct DataView<'a, I: Instance> {
    /// This struct is a view over this dataset.
    pub dataset: &'a DataSet<I>,
    /// The feature values for instances that remain in this view. Indexed first by feature_id, then sorted by feature value.
    feature_values_sorted: Vec<Vec<FeatureValue>>,
}

impl<I: Instance> Debug for DataView<'_, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("DataView");
        for feature in &self.feature_values_sorted {
            d.field("feature_values_sorted", &feature.len());
        }
        d.finish()
    }
}

impl<'a, I: Instance> DataView<'a, I> {
    pub fn from_dataset(dataset: &'a DataSet<I>) -> Self {
        // Copy over all the feature values from the dataset, and sort them by the feature values.
        let mut feature_values_sorted = Vec::new();

        for feature in &dataset.feature_values {
            let mut feature_values_sorted_i = Vec::new();

            for (instance_id, &feature_value) in feature.iter().enumerate() {
                feature_values_sorted_i.push(FeatureValue {
                    instance_id,
                    feature_value,
                })
            }

            feature_values_sorted_i.sort_by_key(|fv| fv.feature_value);

            feature_values_sorted.push(feature_values_sorted_i)
        }

        Self {
            dataset,
            feature_values_sorted,
        }
    }

    /// Split this dataview into two, the first containing only those instances where `split_feature <= threshold`, and the second where `split_feature > threshold`.
    pub fn split(&self, split_feature: usize, threshold: i32) -> (Self, Self) {
        let mut feature_values_left = Vec::with_capacity(self.feature_values_sorted.len());
        let mut feature_values_right = Vec::with_capacity(self.feature_values_sorted.len());

        for feature in &self.feature_values_sorted {
            let mut feature_values_left_i = Vec::with_capacity(feature.len());
            let mut feature_values_right_i = Vec::with_capacity(feature.len());

            for &value in feature {
                if self.dataset.feature_values[split_feature][value.instance_id] <= threshold {
                    // Assure the compiler that we do not need reallocation when pushing.
                    unsafe {
                        std::hint::assert_unchecked(
                            feature_values_left_i.len() < feature_values_left_i.capacity(),
                        )
                    }
                    feature_values_left_i.push(value);
                } else {
                    // Assure the compiler that we do not need reallocation when pushing.
                    unsafe {
                        std::hint::assert_unchecked(
                            feature_values_right_i.len() < feature_values_right_i.capacity(),
                        )
                    }
                    feature_values_right_i.push(value);
                }
            }

            feature_values_left.push(feature_values_left_i);
            feature_values_right.push(feature_values_right_i);
        }

        (
            Self {
                dataset: self.dataset,
                feature_values_sorted: feature_values_left,
            },
            Self {
                dataset: self.dataset,
                feature_values_sorted: feature_values_right,
            },
        )
    }

    pub fn instances_iter(&self) -> impl Iterator<Item = usize> {
        self.feature_values_sorted[0].iter().map(|i| i.instance_id)
    }

    pub fn remaining_feature_ranges(&self) -> impl Iterator<Item = (usize, Range<i32>)> {
        self.feature_values_sorted
            .iter()
            .enumerate()
            .filter_map(|(feature, values)| match (values.first(), values.last()) {
                (Some(x), Some(y)) => {
                    // For each feature, consider all useful splitting points. Note: last feature value is
                    // not a useful splitting point because all instances would go to the left. The range
                    // excludes the endpoint.
                    assert!(x.feature_value <= y.feature_value);
                    if x.feature_value != y.feature_value {
                        Some((feature, x.feature_value..y.feature_value))
                    } else {
                        None
                    }
                }
                // No branching decisions for this feature.
                _ => None,
            })
    }

    pub fn feature_range_remains(&self, feature: usize, range: &Range<i32>) -> bool {
        self.feature_values_sorted[feature]
            .binary_search_by(|a| {
                if a.feature_value < range.start {
                    Ordering::Less
                } else if a.feature_value >= range.end {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            })
            .is_ok()
    }

    pub fn num_instances(&self) -> usize {
        self.feature_values_sorted[0].len()
    }

    pub fn num_features(&self) -> usize {
        self.feature_values_sorted.len()
    }
}
