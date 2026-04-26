use crate::{
    model::reduction::{ReductionStats, reduce_dataset},
    tasks::{CostSum, OptimizationTask},
};

use super::dataset::DataSet;

use std::{fmt::Debug, ops::Range};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FeatureValue {
    /// The id of the instance that has this feature value
    pub instance_id: usize,

    /// Float feature values are substituted by subsequent integer values such that
    /// `x.feature_value < y.feature_value` iff `x.original_feature_value < y.original_feature_value`
    /// (non-unique values get the same int value).
    pub feature_value: i32,
}

#[derive(Clone, Copy)]
pub struct SplitValue {
    /// The feature value to split on.
    pub feature_value: i32,
    /// The score greedily assignd to this split value. E.g. gini or SSE.
    pub greedy_value: f32,
}

#[derive(Clone, Copy)]
pub struct BestGreedySplit {
    /// The index of the split value in the `possible_split_values` vector.
    pub split_value_index: usize,
    /// The greedy value of this split.
    pub greedy_value: f32,
}

pub struct DataView<'a, OT: OptimizationTask> {
    /// This struct is a view over this dataset.
    pub dataset: &'a DataSet<OT::InstanceType>,
    /// The feature values for instances that remain in this view. Indexed first
    /// by feature_id, then sorted by feature value.
    feature_values_sorted: Vec<Vec<FeatureValue>>,
    pub feature_values: Vec<Vec<i32>>, // [feature][instance_pos]
    pub instance_ids: Vec<usize>,      // instance_pos -> dataset instance id
    instance_pos_by_id: Vec<usize>,
    value_sources: Vec<Vec<(usize, i32)>>, // [feature][feature_value] -> (original_feature, original_cut_value)
    pub possible_split_values: Vec<Vec<SplitValue>>,
    pub cost_summer: OT::CostSumType,
    /// The best greedy splits per feature for this dataview. Indexed by feature.
    pub best_greedy_splits: Vec<BestGreedySplit>,
    /// The ranking of each feature based on their best greedy split. Ties resolved arbitrarily. Indexed by the feature id.
    pub feature_ranking: Vec<i32>,
    pub unique_labels: Vec<OT::LabelType>,
    pub extra_data: OT::ExtraDataviewData,
    pub reduction_stats: ReductionStats,
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
    fn collect_unique_labels(
        dataset: &'a DataSet<OT::InstanceType>,
        feature_values_sorted: &[Vec<FeatureValue>],
    ) -> Vec<OT::LabelType>
    where
        OT::LabelType: PartialEq,
    {
        let Some(first_feature_values) = feature_values_sorted.first() else {
            return Vec::new();
        };

        let mut unique_labels = Vec::new();
        for feature_value in first_feature_values {
            let label = OT::label_of_instance(&dataset.instances[feature_value.instance_id]);
            if !unique_labels.contains(&label) {
                unique_labels.push(label);
            }
        }

        unique_labels
    }

    fn add_possible_split_value(
        possible_split_values: &mut Vec<SplitValue>,
        previous_feature_value: &mut i32,
        value: &FeatureValue,
        left_costsum: &mut OT::CostSumType,
        right_costsum: &mut OT::CostSumType,
        best_greedy_split: &mut BestGreedySplit,
        dataset: &'a DataSet<OT::InstanceType>,
    ) {
        if value.feature_value != *previous_feature_value {
            // We reached a new feature value, so the previous split value (if any) can now be evaluated:
            // left currently has all instances <= previous_feature_value.
            if let Some(prev_split_value) = possible_split_values.last_mut() {
                // If the previous split value was not added for some reason, then we do not need to update it.
                // This can happen if the previous split value was not useful, e.g. a previous split had a zero cost on the right side.
                if prev_split_value.feature_value == *previous_feature_value {
                    let prev_greedy_value = OT::greedy_value(left_costsum, right_costsum);
                    prev_split_value.greedy_value = prev_greedy_value;

                    if best_greedy_split.greedy_value > prev_greedy_value {
                        best_greedy_split.split_value_index = possible_split_values.len() - 1;
                        best_greedy_split.greedy_value = prev_greedy_value;
                    }
                }
            }

            *previous_feature_value = value.feature_value;

            possible_split_values.push(SplitValue {
                feature_value: value.feature_value,
                greedy_value: 0.0, // This will be set later.
            });
        }

        *left_costsum += &dataset.instances[value.instance_id];
        *right_costsum -= &dataset.instances[value.instance_id];
    }

    /// Post-process the possible split values to remove the last one if it is the final feature value.
    fn post_process_possible_splits(
        possible_split_values: &mut Vec<SplitValue>,
        final_feature_value: i32,
    ) {
        // If it is the final feature value, the right side is empty. So it is not useful.
        if let Some(last) = possible_split_values.last() {
            if last.feature_value == final_feature_value {
                possible_split_values.pop();
            }
        }
    }

    fn feature_rank_from_best_greedy_splits(best_greedy_splits: &[BestGreedySplit]) -> Vec<i32> {
        // Create a ranking of the features based on their best greedy split.
        let mut feature_ranking: Vec<usize> = (0..best_greedy_splits.len()).collect();
        feature_ranking.sort_by(|&a, &b| {
            best_greedy_splits[a]
                .greedy_value
                .total_cmp(&best_greedy_splits[b].greedy_value)
        });
        (0..best_greedy_splits.len() as i32)
            .map(|i| {
                // The feature ranking is the index of the feature in the sorted list.
                // So we need to find the index of the feature in the original list.
                feature_ranking
                    .iter()
                    .position(|&x| x == i as usize)
                    .unwrap() as i32
            })
            .collect()
    }

    pub fn from_dataset(dataset: &'a DataSet<OT::InstanceType>, use_reduction: bool) -> Self
    where
        OT::LabelType: PartialEq,
    {
        let reduced = reduce_dataset::<OT>(dataset, use_reduction);

        let mut feature_values_sorted = Vec::new();
        let mut possible_split_values = Vec::new();
        let mut best_greedy_splits = Vec::new();

        let mut left_costsum = OT::init_costsum(dataset);
        let mut right_costsum = left_costsum.clone();

        for &dataset_instance_id in &reduced.instance_ids {
            left_costsum += &dataset.instances[dataset_instance_id];
        }

        let mut value_sources = Vec::with_capacity(reduced.feature_values.len());

        for (feature_idx, feature) in reduced.feature_values.iter().enumerate() {
            let mut feature_values_sorted_i = Vec::new();
            let mut possible_split_values_i = Vec::new();

            for (pos, &feature_value) in feature.iter().enumerate() {
                feature_values_sorted_i.push(FeatureValue {
                    instance_id: reduced.instance_ids[pos],
                    feature_value,
                })
            }

            feature_values_sorted_i.sort_by_key(|fv| fv.feature_value);

            right_costsum += &left_costsum;
            left_costsum.clear();

            let mut previous = -1;
            let mut best_greedy_split = BestGreedySplit {
                split_value_index: 0,
                greedy_value: f32::INFINITY,
            };

            for fv in &feature_values_sorted_i {
                Self::add_possible_split_value(
                    &mut possible_split_values_i,
                    &mut previous,
                    fv,
                    &mut left_costsum,
                    &mut right_costsum,
                    &mut best_greedy_split,
                    dataset,
                );
            }

            if previous != -1 {
                Self::post_process_possible_splits(&mut possible_split_values_i, previous);
            }

            let max_value = feature.iter().copied().max().unwrap_or(0).max(0) as usize;
            let mut sources = vec![(feature_idx, 0); max_value + 1];
            let mut explicit = vec![false; max_value + 1];
            for (i, src) in reduced.cut_sources[feature_idx].iter().copied().enumerate() {
                if i < sources.len() {
                    sources[i] = src;
                    explicit[i] = true;
                }
            }
            if let Some(last_known) = reduced.cut_sources[feature_idx].last().copied() {
                for (idx, src) in sources.iter_mut().enumerate() {
                    if !explicit[idx] {
                        *src = last_known;
                    }
                }
            }
            value_sources.push(sources);

            feature_values_sorted.push(feature_values_sorted_i);
            possible_split_values.push(possible_split_values_i);
            best_greedy_splits.push(best_greedy_split);
        }

        let instance_ids = reduced.instance_ids;
        let mut instance_pos_by_id = vec![usize::MAX; dataset.instances.len()];
        for (pos, &id) in instance_ids.iter().enumerate() {
            instance_pos_by_id[id] = pos;
        }

        let feature_ranking = Self::feature_rank_from_best_greedy_splits(&best_greedy_splits);
        let unique_labels = Self::collect_unique_labels(dataset, &feature_values_sorted);
        let extra_data = OT::init_extra_dataview_data(dataset, &feature_values_sorted);

        Self {
            dataset,
            feature_values_sorted,
            feature_values: reduced.feature_values,
            instance_ids,
            instance_pos_by_id,
            value_sources,
            possible_split_values,
            cost_summer: left_costsum,
            best_greedy_splits,
            feature_ranking,
            unique_labels,
            extra_data,
            reduction_stats: reduced.stats,
        }
    }

    #[inline]
    fn add_feature_value(feature_values: &mut Vec<FeatureValue>, value: FeatureValue) {
        unsafe {
            std::hint::assert_unchecked(feature_values.len() < feature_values.capacity());
        }
        feature_values.push(value);
    }

    #[inline]
    pub fn feature_value(&self, feature: usize, dataset_instance_id: usize) -> i32 {
        let pos = self.instance_pos_by_id[dataset_instance_id];
        self.feature_values[feature][pos]
    }

    fn source_for_split_value(&self, feature: usize, split_feature_value: i32) -> (usize, i32) {
        let value = split_feature_value.max(0) as usize;
        if value < self.value_sources[feature].len() {
            self.value_sources[feature][value]
        } else {
            (feature, split_feature_value)
        }
    }

    pub fn split(&self, split_feature: usize, split_value: usize) -> (Self, Self)
    where
        OT::LabelType: PartialEq,
    {
        let threshold = self.possible_split_values[split_feature][split_value].feature_value;
        let mut feature_values_left = Vec::with_capacity(self.feature_values_sorted.len());
        let mut feature_values_right = Vec::with_capacity(self.feature_values_sorted.len());
        let mut possible_split_values_left = Vec::with_capacity(self.possible_split_values.len());
        let mut possible_split_values_right = Vec::with_capacity(self.possible_split_values.len());
        let mut best_greedy_splits_left = Vec::with_capacity(self.best_greedy_splits.len());
        let mut best_greedy_splits_right = Vec::with_capacity(self.best_greedy_splits.len());

        let mut costsum_ll = self.cost_summer.clone();
        costsum_ll.clear();
        let mut costsum_lr = costsum_ll.clone();
        let mut costsum_rr = costsum_lr.clone();

        for value in &self.feature_values_sorted[split_feature] {
            if self.feature_value(split_feature, value.instance_id) <= threshold {
                costsum_ll += &self.dataset.instances[value.instance_id];
            } else {
                break;
            }
        }
        let mut costsum_rl = self.cost_summer.clone();
        costsum_rl -= &costsum_ll;

        for (feature_idx, feature) in self.feature_values_sorted.iter().enumerate() {
            let mut feature_values_left_i = Vec::with_capacity(feature.len());
            let mut feature_values_right_i = Vec::with_capacity(feature.len());
            let mut possible_split_values_left_i =
                Vec::with_capacity(self.possible_split_values[feature_idx].len());
            let mut possible_split_values_right_i =
                Vec::with_capacity(self.possible_split_values[feature_idx].len());

            let mut last_feature_value_left = -1;
            let mut last_feature_value_right = -1;
            let mut best_greedy_split_left = BestGreedySplit {
                split_value_index: 0,
                greedy_value: f32::INFINITY,
            };
            let mut best_greedy_split_right = BestGreedySplit {
                split_value_index: 0,
                greedy_value: f32::INFINITY,
            };

            costsum_lr += &costsum_ll;
            costsum_ll.clear();
            costsum_rr += &costsum_rl;
            costsum_rl.clear();

            for &value in feature {
                if self.feature_value(split_feature, value.instance_id) <= threshold {
                    Self::add_feature_value(&mut feature_values_left_i, value);
                    Self::add_possible_split_value(
                        &mut possible_split_values_left_i,
                        &mut last_feature_value_left,
                        &value,
                        &mut costsum_ll,
                        &mut costsum_lr,
                        &mut best_greedy_split_left,
                        self.dataset,
                    );
                } else {
                    Self::add_feature_value(&mut feature_values_right_i, value);
                    Self::add_possible_split_value(
                        &mut possible_split_values_right_i,
                        &mut last_feature_value_right,
                        &value,
                        &mut costsum_rl,
                        &mut costsum_rr,
                        &mut best_greedy_split_right,
                        self.dataset,
                    );
                }
            }

            Self::post_process_possible_splits(
                &mut possible_split_values_left_i,
                last_feature_value_left,
            );
            Self::post_process_possible_splits(
                &mut possible_split_values_right_i,
                last_feature_value_right,
            );

            assert!(!feature_values_left_i.is_empty());
            assert!(!feature_values_right_i.is_empty());

            feature_values_left.push(feature_values_left_i);
            feature_values_right.push(feature_values_right_i);
            possible_split_values_left.push(possible_split_values_left_i);
            possible_split_values_right.push(possible_split_values_right_i);
            best_greedy_splits_left.push(best_greedy_split_left);
            best_greedy_splits_right.push(best_greedy_split_right);
        }

        let feature_ranking_left =
            Self::feature_rank_from_best_greedy_splits(&best_greedy_splits_left);
        let feature_ranking_right =
            Self::feature_rank_from_best_greedy_splits(&best_greedy_splits_right);
        let unique_labels_left = Self::collect_unique_labels(self.dataset, &feature_values_left);
        let unique_labels_right = Self::collect_unique_labels(self.dataset, &feature_values_right);

        let extra_data_left = OT::init_extra_dataview_data(self.dataset, &feature_values_left);
        let extra_data_right = OT::init_extra_dataview_data(self.dataset, &feature_values_right);

        let left_instance_ids: Vec<usize> = feature_values_left[0]
            .iter()
            .map(|fv| fv.instance_id)
            .collect();
        let right_instance_ids: Vec<usize> = feature_values_right[0]
            .iter()
            .map(|fv| fv.instance_id)
            .collect();

        let mut left_pos = vec![usize::MAX; self.dataset.instances.len()];
        for (pos, &id) in left_instance_ids.iter().enumerate() {
            left_pos[id] = pos;
        }
        let mut right_pos = vec![usize::MAX; self.dataset.instances.len()];
        for (pos, &id) in right_instance_ids.iter().enumerate() {
            right_pos[id] = pos;
        }

        let mut left_feature_values = vec![vec![0; left_instance_ids.len()]; self.num_features()];
        let mut right_feature_values = vec![vec![0; right_instance_ids.len()]; self.num_features()];

        for f in 0..self.num_features() {
            for fv in &feature_values_left[f] {
                left_feature_values[f][left_pos[fv.instance_id]] = fv.feature_value;
            }
            for fv in &feature_values_right[f] {
                right_feature_values[f][right_pos[fv.instance_id]] = fv.feature_value;
            }
        }

        (
            Self {
                dataset: self.dataset,
                feature_values_sorted: feature_values_left,
                feature_values: left_feature_values,
                instance_ids: left_instance_ids,
                instance_pos_by_id: left_pos,
                value_sources: self.value_sources.clone(),
                possible_split_values: possible_split_values_left,
                cost_summer: costsum_ll,
                best_greedy_splits: best_greedy_splits_left,
                feature_ranking: feature_ranking_left,
                unique_labels: unique_labels_left,
                extra_data: extra_data_left,
                reduction_stats: self.reduction_stats.clone(),
            },
            Self {
                dataset: self.dataset,
                feature_values_sorted: feature_values_right,
                feature_values: right_feature_values,
                instance_ids: right_instance_ids,
                instance_pos_by_id: right_pos,
                value_sources: self.value_sources.clone(),
                possible_split_values: possible_split_values_right,
                cost_summer: costsum_rl,
                best_greedy_splits: best_greedy_splits_right,
                feature_ranking: feature_ranking_right,
                unique_labels: unique_labels_right,
                extra_data: extra_data_right,
                reduction_stats: self.reduction_stats.clone(),
            },
        )
    }

    pub fn instances_iter(&self, feature: usize) -> impl Iterator<Item = &FeatureValue> {
        self.feature_values_sorted[feature].iter()
    }

    pub fn num_instances(&self) -> usize {
        self.feature_values_sorted[0].len()
    }

    pub fn num_features(&self) -> usize {
        self.feature_values_sorted.len()
    }

    pub fn num_unique_labels(&self) -> usize {
        self.unique_labels.len()
    }

    #[inline]
    pub fn is_pure(&self) -> bool {
        self.num_unique_labels() <= 1
    }

    fn threshold_from_original_cut(&self, original_feature: usize, cut_value: i32) -> f64 {
        let cut_idx = cut_value.max(0) as usize;
        let vals = &self.dataset.internal_to_original_feature_value[original_feature];
        let current = vals[cut_idx];
        let next = vals.get(cut_idx + 1).copied().unwrap_or(current);
        (current + next) / 2.0
    }

    pub fn threshold_from_split(&self, split_feature: usize, split_value: usize) -> f64 {
        let split_feature_value =
            self.possible_split_values[split_feature][split_value].feature_value;
        let (orig_feature, orig_cut_value) =
            self.source_for_split_value(split_feature, split_feature_value);
        self.threshold_from_original_cut(orig_feature, orig_cut_value)
    }

    pub fn original_split_feature_from_split(
        &self,
        split_feature: usize,
        split_value: usize,
    ) -> usize {
        let split_feature_value =
            self.possible_split_values[split_feature][split_value].feature_value;
        self.source_for_split_value(split_feature, split_feature_value)
            .0
    }

    pub fn instance_range_from_split_range(
        &self,
        split_feature: usize,
        split_values: Range<usize>,
    ) -> Range<usize> {
        if split_values.is_empty() {
            return 0..0;
        }
        let start_fv = self.possible_split_values[split_feature][split_values.start].feature_value;
        let end_fv = self.possible_split_values[split_feature][split_values.end - 1].feature_value;

        let start = self.feature_values_sorted[split_feature]
            .partition_point(|fv| fv.feature_value < start_fv);
        let end = self.feature_values_sorted[split_feature]
            .partition_point(|fv| fv.feature_value <= end_fv);
        start..end
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::instance::LabeledInstance;
    use crate::tasks::accuracy::AccuracyTask;

    fn create_dataset(feature_values: Vec<i32>, labels: Vec<i32>) -> DataSet<LabeledInstance<i32>> {
        let mut dataset = DataSet::default();
        for label in labels {
            dataset.instances.push(LabeledInstance::new(label));
        }
        dataset.feature_values.push(feature_values);
        dataset
            .internal_to_original_feature_value
            .push(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        dataset
    }

    fn test_possible_splits(feature_values: Vec<i32>, labels: Vec<i32>, expected_splits: Vec<i32>) {
        let dataset = create_dataset(feature_values, labels);
        let view = DataView::<AccuracyTask>::from_dataset(&dataset, false);

        assert_eq!(
            view.possible_split_values[0]
                .iter()
                .map(|s| s.feature_value)
                .collect::<Vec<i32>>(),
            expected_splits
        );
    }

    #[test]
    fn possible_splits_smoke_test() {
        let feature_values = vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4, 5, 6, 6];
        let labels = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0];
        let expected_splits = vec![0, 1, 2, 4, 5];
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
    fn instance_range_from_split_range_test() {
        let feature_values = vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4, 5, 6, 6];
        let labels = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0];
        let dataset = create_dataset(feature_values, labels);
        let view = DataView::<AccuracyTask>::from_dataset(&dataset, false);
        assert_eq!(view.possible_split_values[0][0].feature_value, 0);
        assert_eq!(view.possible_split_values[0][2].feature_value, 2);
        assert_eq!(view.instance_range_from_split_range(0, 0..2), 0..7);
    }
}
