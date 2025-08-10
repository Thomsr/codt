use std::ops::{AddAssign, SubAssign};

use ckmeans::ckmeans_dynamic_stop;
use segment_tree::SegmentPoint;

use crate::{
    model::{
        dataset::DataSet,
        dataview::{DataView, FeatureValue},
        instance::LabeledInstance,
    },
    tasks::{CostSum, FloatCost, OptimizationTask},
};

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

impl CostSum<f64, LabeledInstance<f64>, FloatCost> for SquaredErrorCostSum {
    fn label(&self) -> f64 {
        debug_assert!(
            self.n > 0,
            "Getting a label from 0 instances is undefined. This should be excluded by disallowing empty splits."
        );
        // The mean gives the optimal SSE in a leaf.
        self.y / (self.n as f64)
    }

    fn cost(&self) -> FloatCost {
        debug_assert!(
            self.n > 0,
            "Getting a cost from 0 instances is undefined. This should be excluded by disallowing empty splits."
        );
        // The sum of squared errors from the mean can be computed from (sum of (y^2)) - ((sum of y)^2 / N)
        (self.y2 - (self.y * self.y) / (self.n as f64)).into()
    }

    fn clear(&mut self) {
        self.y = 0.0;
        self.y2 = 0.0;
        self.n = 0;
    }
}

pub struct ExtraDataviewData {
    /// The maximum of the distance to the minimum or maximum label squared. For each datapoint. With range queries of the sum.
    dist2_tree: Vec<SegmentPoint<f64, segment_tree::ops::Add>>,
}

impl OptimizationTask for SquaredErrorTask {
    type LabelType = f64;
    type InstanceType = LabeledInstance<f64>;
    type CostType = FloatCost;
    type CostSumType = SquaredErrorCostSum;
    type ExtraDataviewData = ExtraDataviewData;

    fn prepare_for_data(&mut self, dataview: &mut DataView<Self>) {
        self.dataset_size = dataview.num_instances();
        self.branching_cost = dataview.cost_summer.cost().0 * self.complexity_cost;
    }

    fn print_cost(&mut self, cost: &Self::CostType) -> String {
        format!(
            "SSE: {}. MSE: {}. (Only accurate when complexity cost is zero)",
            cost,
            cost.0 / self.dataset_size as f64
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
        self.branching_cost.into()
    }

    fn branch_relaxation(&self, dataview: &DataView<Self>, max_depth: u32) -> Self::CostType
    where
        Self: Sized,
    {
        if max_depth > 7 {
            // Since clusters are u8, limit to max depth 7
            FloatCost(0.0)
        } else {
            // The maximum number of clusters is the remaining leaf count or the number of instances remaining.
            let clusters: u8 = dataview.num_instances().min(1 << max_depth) as u8;
            let labels: Vec<f64> = dataview
                .instances_iter(0)
                .map(|i| dataview.dataset.instances[i.instance_id].label)
                .collect();
            let kmeans_result = ckmeans_dynamic_stop(&labels, clusters, self.branching_cost);
            let kmeans_matrix = kmeans_result.unwrap();
            let nclusters = kmeans_matrix.len();
            let branch_cost = self.branching_cost * (nclusters - 1) as f64;
            let sse = kmeans_matrix
                .last()
                .expect("At least one cluster should be found")
                .last()
                .expect("At least one instance in dataview, so this should exist");
            (sse + branch_cost).into()
        }
    }

    fn greedy_value(left_costsum: &Self::CostSumType, right_costsum: &Self::CostSumType) -> f32 {
        left_costsum.cost().0 as f32 + right_costsum.cost().0 as f32
    }

    fn worst_cost_in_range(
        dataview: &DataView<Self>,
        feature: usize,
        range: std::ops::Range<usize>,
    ) -> Self::CostType
    where
        Self: Sized,
    {
        FloatCost(dataview.extra_data.dist2_tree[feature].query(range.start, range.end))
    }

    fn init_extra_dataview_data(
        dataset: &DataSet<Self::InstanceType>,
        feature_values: &[Vec<FeatureValue>],
    ) -> Self::ExtraDataviewData {
        let mut label_min = dataset.instances[feature_values[0][0].instance_id].label;
        let mut label_max = label_min;

        for value in &feature_values[0] {
            label_max = label_max.max(dataset.instances[value.instance_id].label);
            label_min = label_min.min(dataset.instances[value.instance_id].label);
        }

        let mut dist2_tree = Vec::with_capacity(feature_values.len());

        let n = feature_values[0].len();
        for values in feature_values {
            let mut dist2_values = vec![0.0_f64; n * 2];
            for i in 0..n {
                let y = dataset.instances[values[i].instance_id].label;
                let dist = (y - label_min).max(label_max - y);
                dist2_values[n + i] = dist * dist;
            }
            dist2_tree.push(SegmentPoint::build_noalloc(
                dist2_values,
                segment_tree::ops::Add,
            ));
        }

        ExtraDataviewData { dist2_tree }
    }
}
