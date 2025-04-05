use super::instance::Instance;

pub struct DataSet<I: Instance> {
    /// Collection of instances. The index into this array determines its instance_id.
    pub instances: Vec<I>,
    /// The original float feature values. Indexed first by feature_id, then by instance_id. Only used in the final tree.
    pub original_feature_values: Vec<Vec<f64>>,
    /// The internally used feature values. Indexed first by feature_id, then by instance_id.
    pub feature_values: Vec<Vec<i32>>,
}

/// Explicit implementation of Default, since I is not constrained by Default
impl<I: Instance> Default for DataSet<I> {
    fn default() -> Self {
        Self {
            instances: Vec::new(),
            original_feature_values: Vec::new(),
            feature_values: Vec::new(),
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

            feature_values[ids[0]] = 0;
            for idx in 1..ids.len() {
                let last_original = self.original_feature_values[i][ids[idx - 1]];
                let this_original = self.original_feature_values[i][ids[idx]];
                let last_feature_value = feature_values[ids[idx - 1]];

                feature_values[ids[idx]] = if last_original == this_original {
                    last_feature_value
                } else {
                    last_feature_value + 1
                }
            }

            self.feature_values.push(feature_values);
        }
    }
}
