use crate::model::{dataview::DataView, instance::LabeledInstance};

use super::OptimizationTask;

#[derive(Default)]
pub struct AccuracyTask {
    dataset_size: usize,
    num_labels: i32,
}

impl OptimizationTask for AccuracyTask {
    type InstanceType = LabeledInstance<i32>;
    type CostType = i32;
    const MIN_COST: Self::CostType = 0;

    fn prepare_for_data(&mut self, dataview: &mut DataView<Self::InstanceType>) {
        self.dataset_size = dataview.num_instances();
        self.num_labels = 0;
        for instance in &dataview.dataset.instances {
            self.num_labels = self.num_labels.max(instance.label + 1);
        }
    }

    fn print_cost(&mut self, cost: &Self::CostType) -> String {
        format!(
            "Misclassifications: {}. Accuracy: {}%",
            cost,
            (1.0 - *cost as f64 / self.dataset_size as f64) * 100.0
        )
    }

    fn leaf_cost(&self, dataview: &DataView<Self::InstanceType>) -> i32 {
        let mut instance_count_per_class = vec![0; self.num_labels as usize];

        for instance_id in dataview.instances_iter() {
            instance_count_per_class[dataview.dataset.instances[instance_id].label as usize] += 1;
        }

        let largest_class_size = instance_count_per_class
            .iter()
            .fold(0, |acc, e| acc.max(*e));

        dataview.num_instances() as i32 - largest_class_size
    }
}
