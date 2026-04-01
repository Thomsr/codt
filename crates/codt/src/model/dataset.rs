use crate::model::instance::LabeledInstance;

use super::instance::Instance;

pub struct DataSet<I: Instance> {
    /// Collection of instances. The index into this array determines its instance_id.
    pub instances: Vec<I>,
    /// The original float feature values. Indexed first by feature_id, then by instance_id. Only used to read the data.
    pub original_feature_values: Vec<Vec<f64>>,
    /// The internally used feature values. Indexed first by feature_id, then by instance_id.
    pub feature_values: Vec<Vec<i32>>,
    /// The original float feature values. Indexed first by feature_id, then by feature_value. Used to reconstruct the final tree.
    pub internal_to_original_feature_value: Vec<Vec<f64>>,
}

/// Explicit implementation of Default, since I is not constrained by Default
impl<I: Instance> Default for DataSet<I> {
    fn default() -> Self {
        Self {
            instances: Vec::new(),
            original_feature_values: Vec::new(),
            feature_values: Vec::new(),
            internal_to_original_feature_value: Vec::new(),
        }
    }
}

impl<I: Instance> DataSet<I> {
    /// Add an instance to the data set.
    pub fn add_instance<T>(&mut self, instance: I, feature_values: T)
    where
        T: IntoIterator<Item = f64>,
    {
        self.instances.push(instance);

        for (i, feature_value) in feature_values.into_iter().enumerate() {
            if i >= self.original_feature_values.len() {
                self.original_feature_values.push(Vec::new());
            }
            self.original_feature_values[i].push(feature_value);
        }
    }

    /// After adding all the instances, this needs to be run to substitute feature values with ints and set some auxiliary values.
    pub fn preprocess_after_adding_instances(&mut self) {
        let mut ids: Vec<usize> = (0..self.instances.len()).collect();

        for i in 0..self.original_feature_values.len() {
            ids.sort_by(|&a, &b| {
                self.original_feature_values[i][a]
                    .partial_cmp(&self.original_feature_values[i][b])
                    .expect("Uncomparable floating point value in feature values. E.g. NaN.")
            });

            let mut feature_values = vec![0; self.original_feature_values[i].len()];
            let mut internal_to_original_feature_value = Vec::new();

            feature_values[ids[0]] = 0;
            internal_to_original_feature_value.push(self.original_feature_values[i][ids[0]]);
            for idx in 1..ids.len() {
                let last_original = self.original_feature_values[i][ids[idx - 1]];
                let this_original = self.original_feature_values[i][ids[idx]];
                let last_feature_value = feature_values[ids[idx - 1]];

                // Use f32::EPSILON to avoid floating point precision issues with lesser precision input.
                feature_values[ids[idx]] = if last_original + f32::EPSILON as f64 >= this_original {
                    last_feature_value
                } else {
                    let new_id = last_feature_value + 1;
                    internal_to_original_feature_value.push(this_original);
                    new_id
                }
            }

            self.feature_values.push(feature_values);
            self.internal_to_original_feature_value
                .push(internal_to_original_feature_value);
        }
    }

    pub fn num_instances(&self) -> usize {
        self.feature_values[0].len()
    }

    pub fn from_csv(path: &std::path::Path) -> Self
    where
        I: From<LabeledInstance<i32>>,
    {
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read dataset {}: {}", path.display(), e));

        let mut dataset = DataSet::<I>::default();
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

            dataset.add_instance(I::from(LabeledInstance::new(label)), features);
        }

        dataset.preprocess_after_adding_instances();
        dataset
    }
}
