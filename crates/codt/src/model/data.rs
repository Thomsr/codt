use super::instance::Instance;

pub struct DataSet<I: Instance> {
    /// Collection of instances. The index into this array determines its instance_id.
    instances: Vec<I>,
    /// The original float feature values. Indexed first by feature_id, then by instance_id. Only used in the final tree.
    pub original_feature_values: Vec<Vec<f32>>,
    /// The internally used feature values. Indexed first by feature_id, then by instance_id.
    /// Preprocessed so they are subsubsequent integer values (non-unique values get the same int value)
    pub feature_values: Vec<Vec<i32>>,
}

impl<I: Instance> DataSet<I> {
    pub fn instances_mut(&mut self) -> &mut Vec<I> {
        &mut self.instances
    }

    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
            original_feature_values: Vec::new(),
            feature_values: Vec::new(),
        }
    }

    /// Add an instance to the data set.
    pub fn add_instance(&mut self, instance: I, feature_values: &[f32]) {
        let instance_id = self.instances.len();
        self.instances.push(instance);

        // All feature vectors should be the same size
        let feature_vector_len = self.feature_values.len();
        if feature_vector_len == 0 {
            self.feature_values
                .resize_with(feature_vector_len, Vec::new);
            self.original_feature_values
                .resize_with(feature_vector_len, Vec::new);
        }

        assert!(feature_vector_len == feature_values.len());

        for (i, &feature_value) in feature_values.iter().enumerate() {
            self.original_feature_values[i][instance_id] = feature_value;
        }
    }

    /// Sets the next feature value for the last instance added.
    pub fn add_feature_value_for_last_instance(&mut self, feature_value: f32) {
        let instance_id = self.instances.len() - 1;
        self.original_feature_values[instance_id].push(feature_value);
    }

    /// After adding all the instances, this needs to be run to substitute feature values with ints and set some auxiliary values.
    pub fn preprocess_after_adding_instances(&mut self) {
        let mut ids: Vec<usize> = (0..self.instances.len()).collect();

        for i in 0..self.original_feature_values.len() {
            ids.sort_by(|&a, &b| {
                self.original_feature_values[i][a]
                    .partial_cmp(&self.original_feature_values[i][b])
                    .unwrap()
            });

            self.feature_values[i][ids[0]] = 0;
            for idx in 1..ids.len() {
                let last_original = self.original_feature_values[i][ids[idx - 1]];
                let this_original = self.original_feature_values[i][ids[idx]];
                let last_feature_value = self.feature_values[i][ids[idx - 1]];

                self.feature_values[i][ids[idx]] = if last_original == this_original {
                    last_feature_value
                } else {
                    last_feature_value + 1
                }
            }
        }
    }
}
