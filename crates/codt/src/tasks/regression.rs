use crate::model::{dataview::DataView, instance::RegressionInstance};

use super::OptimizationTask;

#[derive(Default)]
pub struct RegressionTask {
    dataset_size: usize,
}

impl OptimizationTask for RegressionTask {
    type InstanceType = RegressionInstance;
    type CostType = f64;
    const MIN_COST: Self::CostType = 0.0;

    fn prepare_for_data(&mut self, dataview: &mut DataView<Self::InstanceType>) {
        self.dataset_size = dataview.num_instances();
    }

    fn print_cost(&mut self, cost: &Self::CostType) -> String {
        format!("SSE: {}. MSE: {}.", cost, *cost / self.dataset_size as f64)
    }

    fn leaf_cost(&self, dataview: &DataView<Self::InstanceType>) -> f64 {
        let mut y = 0.0;
        let mut y2 = 0.0;

        for instance_id in dataview.instances_iter() {
            let label = dataview.dataset.instances[instance_id].label;
            y += label;
            y2 += label * label;
        }

        // The sum of squared errors from the mean can be computed from (sum of (y^2)) - ((sum of y)^2 / N)
        y2 - (y * y) / (dataview.num_instances() as f64)
    }
}
