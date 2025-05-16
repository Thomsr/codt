use std::ops::{AddAssign, SubAssign};

use crate::model::{dataset::DataSet, dataview::DataView, instance::LabeledInstance};

use super::{CostSum, OptimizationTask};

#[derive(Clone)]
pub struct SquaredErrorTask {
    dataset_size: usize,
    branching_cost: f64,
    complexity_cost: f64,
}

impl SquaredErrorTask {
    pub fn new(complexity_cost: f64) -> Self {
        Self {
            dataset_size: 0,
            branching_cost: 0.0,
            complexity_cost,
        }
    }
}

#[derive(Clone)]
pub struct SquaredErrorCostSum {
    y: f64,
    y2: f64,
    n: i32,
}

impl AddAssign<&SquaredErrorCostSum> for SquaredErrorCostSum {
    fn add_assign(&mut self, rhs: &Self) {
        self.y += rhs.y;
        self.y2 += rhs.y2;
        self.n += rhs.n;
    }
}

impl AddAssign<&LabeledInstance<f64>> for SquaredErrorCostSum {
    fn add_assign(&mut self, rhs: &LabeledInstance<f64>) {
        self.y += rhs.label;
        self.y2 += rhs.label * rhs.label;
        self.n += 1;
    }
}

impl SubAssign<&SquaredErrorCostSum> for SquaredErrorCostSum {
    fn sub_assign(&mut self, rhs: &Self) {
        self.y -= rhs.y;
        self.y2 -= rhs.y2;
        self.n -= rhs.n;
    }
}

impl SubAssign<&LabeledInstance<f64>> for SquaredErrorCostSum {
    fn sub_assign(&mut self, rhs: &LabeledInstance<f64>) {
        self.y -= rhs.label;
        self.y2 -= rhs.label * rhs.label;
        self.n -= 1;
    }
}

impl CostSum<f64, LabeledInstance<f64>, f64> for SquaredErrorCostSum {
    fn label(&self) -> f64 {
        // The mean gives the optimal SSE in a leaf.
        self.y / (self.n as f64)
    }

    fn cost(&self) -> f64 {
        // The sum of squared errors from the mean can be computed from (sum of (y^2)) - ((sum of y)^2 / N)
        self.y2 - (self.y * self.y) / (self.n as f64)
    }
}

impl OptimizationTask for SquaredErrorTask {
    type LabelType = f64;
    type InstanceType = LabeledInstance<f64>;
    type CostType = f64;
    type CostSumType = SquaredErrorCostSum;
    const ZERO_COST: Self::CostType = 0.0;

    fn prepare_for_data(&mut self, dataview: &mut DataView<Self>) {
        self.dataset_size = dataview.num_instances();
        self.branching_cost = dataview.cost_summer.cost() * self.complexity_cost;
    }

    fn print_cost(&mut self, cost: &Self::CostType) -> String {
        format!(
            "SSE: {}. MSE: {}. (Only accurate when complexity cost is zero)",
            cost,
            *cost / self.dataset_size as f64
        )
    }

    fn init_costsum(_dataset: &DataSet<Self::InstanceType>) -> Self::CostSumType {
        Self::CostSumType {
            y: 0.0,
            y2: 0.0,
            n: 0,
        }
    }

    fn branching_cost(&self) -> Self::CostType {
        self.branching_cost
    }
}
