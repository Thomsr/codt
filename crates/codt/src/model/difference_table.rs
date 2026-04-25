use crate::{model::dataview::DataView, tasks::OptimizationTask};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitColumn {
    pub feature: usize,
    pub split_value: usize,
    pub threshold_value: i32,
}

pub struct DifferenceTable {
    pub pairs: Vec<(usize, usize)>,
    pub diffs: Vec<Vec<bool>>,
    pub split_columns: Vec<SplitColumn>,
    pub n_columns: usize,
}

impl DifferenceTable {
    pub fn new<OT: OptimizationTask>(dataview: &DataView<'_, OT>) -> Self {
        let dataset = dataview.dataset;

        let labels = &dataview.unique_labels;
        if labels.len() <= 1 {
            // All instances have the same label, no features needed
            let split_columns = Self::collect_split_columns(dataview);
            return Self {
                pairs: Vec::new(),
                diffs: Vec::new(),
                n_columns: split_columns.len(),
                split_columns,
            };
        }

        let mut label_to_idx = HashMap::new();
        for (idx, &label) in labels.iter().enumerate() {
            label_to_idx.insert(label, idx);
        }

        let mut group_instance_ids = vec![Vec::new(); labels.len()];
        for feature_value in dataview.instances_iter(0) {
            let instance_idx = feature_value.instance_id;
            let label = OT::label_of_instance(&dataset.instances[instance_idx]);
            let group_idx = *label_to_idx
                .get(&label)
                .expect("Instance label missing from dataview.unique_labels");
            group_instance_ids[group_idx].push(instance_idx);
        }

        let mut pairs = Vec::new();
        let mut diffs = Vec::new();
        let split_columns = Self::collect_split_columns(dataview);

        for left_group in 0..group_instance_ids.len() {
            for right_group in (left_group + 1)..group_instance_ids.len() {
                for &left_instance in &group_instance_ids[left_group] {
                    for &right_instance in &group_instance_ids[right_group] {
                        let row = Self::compute_diff_for_pair(
                            dataview,
                            left_instance,
                            right_instance,
                            &split_columns,
                        );
                        pairs.push((left_instance, right_instance));
                        diffs.push(row);
                    }
                }
            }
        }

        Self {
            pairs,
            diffs,
            n_columns: split_columns.len(),
            split_columns,
        }
    }

    pub fn min_size_based_cover(&self, target: usize) -> usize {
        let mut counts = vec![0usize; self.n_columns];

        for row in &self.diffs {
            for (col, &val) in row.iter().enumerate() {
                counts[col] += val as usize;
            }
        }

        counts.sort_unstable_by(|a, b| b.cmp(a));

        let mut covered = 0usize;
        for (i, &c) in counts.iter().enumerate() {
            covered += c;
            if covered >= target {
                return i + 1;
            }
        }

        counts.len()
    }

    pub fn print(&self) {
        println!("\n=== Difference Table ===");

        for (i, ((p, n), diffs)) in self.pairs.iter().zip(&self.diffs).enumerate() {
            let diff_str = diffs
                .iter()
                .enumerate()
                .map(|(c, &d)| {
                    let split = self.split_columns[c];
                    format!(
                        "f{}<=v{}(={}):{}",
                        split.feature,
                        split.split_value,
                        split.threshold_value,
                        if d { "1" } else { "0" }
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");

            println!("Pair {} (pos={}, neg={}) -> [{}]", i, p, n, diff_str);
        }

        println!("========================\n");
    }

    fn collect_split_columns<OT: OptimizationTask>(
        dataview: &DataView<'_, OT>,
    ) -> Vec<SplitColumn> {
        dataview
            .possible_split_values
            .iter()
            .enumerate()
            .flat_map(|(feature, split_values)| {
                split_values
                    .iter()
                    .enumerate()
                    .map(move |(split_value, split)| SplitColumn {
                        feature,
                        split_value,
                        threshold_value: split.feature_value,
                    })
            })
            .collect()
    }

    fn compute_diff_for_pair<OT: OptimizationTask>(
        dataview: &DataView<'_, OT>,
        p: usize,
        n: usize,
        split_columns: &[SplitColumn],
    ) -> Vec<bool> {
        split_columns
            .iter()
            .map(|split| {
                let p_left = dataview.feature_value(split.feature, p) <= split.threshold_value;
                let n_left = dataview.feature_value(split.feature, n) <= split.threshold_value;
                p_left != n_left
            })
            .collect::<Vec<bool>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::{dataset::DataSet, dataview::DataView, instance::LabeledInstance},
        tasks::accuracy::AccuracyTask,
    };

    #[test]
    fn diff_table_xor() {
        let mut dataset = DataSet::default();

        let labels = vec![0, 1];
        for l in labels {
            dataset.instances.push(LabeledInstance::new(l));
        }

        dataset.feature_values.push(vec![0, 1]);
        dataset.feature_values.push(vec![0, 1]);
        dataset.feature_values.push(vec![1, 0]);

        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);
        let diff_table = DifferenceTable::new(&dataview);

        assert_eq!(diff_table.diffs.len(), 1);
        for i in 0..3 {
            assert_eq!(diff_table.diffs[0][i], true);
        }
    }

    #[test]
    fn diff_table_four_instances() {
        let mut dataset = DataSet::default();

        let labels = vec![0, 0, 1, 1];
        for l in labels {
            dataset.instances.push(LabeledInstance::new(l));
        }

        dataset.feature_values.push(vec![0, 1, 0, 1]);
        dataset.feature_values.push(vec![0, 1, 1, 1]);
        dataset.feature_values.push(vec![1, 0, 1, 0]);

        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);
        let diff_table = DifferenceTable::new(&dataview);

        let expected_diffs = vec![
            vec![false, true, false],
            vec![true, true, true],
            vec![true, false, true],
            vec![false, false, false],
        ];

        assert_eq!(diff_table.diffs, expected_diffs);
    }

    #[test]
    fn three_labels() {
        let mut dataset = DataSet::default();

        let labels = vec![0, 0, 1, 2];
        for l in labels {
            dataset.instances.push(LabeledInstance::new(l));
        }

        dataset.feature_values.push(vec![0, 1, 0, 1]);
        dataset.feature_values.push(vec![0, 1, 1, 0]);
        dataset.feature_values.push(vec![1, 0, 1, 1]);

        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);
        let diff_table = DifferenceTable::new(&dataview);
        diff_table.print();

        let expected_diffs = vec![
            vec![false, true, false],
            vec![true, false, true],
            vec![true, false, false],
            vec![false, true, true],
            vec![true, true, false],
        ];
        assert_eq!(diff_table.diffs, expected_diffs);
    }

    #[test]
    fn split_columns_include_all_feature_threshold_candidates() {
        let mut dataset = DataSet::default();

        let labels = vec![0, 1, 0, 1];
        for l in labels {
            dataset.instances.push(LabeledInstance::new(l));
        }

        // Feature 0 yields two candidate split thresholds: 0 and 1.
        dataset.feature_values.push(vec![0, 1, 2, 0]);
        // Feature 1 yields one candidate split threshold: 0.
        dataset.feature_values.push(vec![0, 0, 1, 1]);

        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);
        let diff_table = DifferenceTable::new(&dataview);

        assert_eq!(diff_table.n_columns, 3);
        assert_eq!(
            diff_table
                .split_columns
                .iter()
                .map(|c| (c.feature, c.threshold_value))
                .collect::<Vec<_>>(),
            vec![(0, 0), (0, 1), (1, 0)]
        );
        assert!(diff_table.diffs.contains(&vec![true, true, false]));
    }
}
