use std::ops::{AddAssign, SubAssign};

use crate::model::{dataset::DataSet, dataview::DataView, instance::LabeledInstance};

use super::OptimizationTask;

#[derive(Default)]
pub struct RegressionTask {
    dataset_size: usize,
}

#[derive(Clone)]
pub struct RegressionCostSummer {
    y: f64,
    y2: f64,
    n: i32,
}

impl AddAssign<&RegressionCostSummer> for RegressionCostSummer {
    fn add_assign(&mut self, rhs: &Self) {
        self.y += rhs.y;
        self.y2 += rhs.y2;
        self.n += rhs.n;
    }
}

impl AddAssign<&LabeledInstance<f64>> for RegressionCostSummer {
    fn add_assign(&mut self, rhs: &LabeledInstance<f64>) {
        self.y += rhs.label;
        self.y2 += rhs.label * rhs.label;
        self.n += 1;
    }
}

impl SubAssign<&RegressionCostSummer> for RegressionCostSummer {
    fn sub_assign(&mut self, rhs: &Self) {
        self.y -= rhs.y;
        self.y2 -= rhs.y2;
        self.n -= rhs.n;
    }
}

impl SubAssign<&LabeledInstance<f64>> for RegressionCostSummer {
    fn sub_assign(&mut self, rhs: &LabeledInstance<f64>) {
        self.y -= rhs.label;
        self.y2 -= rhs.label * rhs.label;
        self.n -= 1;
    }
}

impl From<&RegressionCostSummer> for f64 {
    fn from(value: &RegressionCostSummer) -> Self {
        // The sum of squared errors from the mean can be computed from (sum of (y^2)) - ((sum of y)^2 / N)
        value.y2 - (value.y * value.y) / (value.n as f64)
    }
}

impl OptimizationTask for RegressionTask {
    type InstanceType = LabeledInstance<f64>;
    type CostType = f64;
    type CostSummer = RegressionCostSummer;
    const MIN_COST: Self::CostType = 0.0;

    fn prepare_for_data(&mut self, dataview: &mut DataView<Self>) {
        self.dataset_size = dataview.num_instances();
    }

    fn print_cost(&mut self, cost: &Self::CostType) -> String {
        format!("SSE: {}. MSE: {}.", cost, *cost / self.dataset_size as f64)
    }

    fn init_costsum(_dataset: &DataSet<Self::InstanceType>) -> Self::CostSummer {
        Self::CostSummer {
            y: 0.0,
            y2: 0.0,
            n: 0,
        }
    }
}
