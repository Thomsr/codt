use std::ops::{AddAssign, SubAssign};

use crate::{
    model::{dataset::DataSet, dataview::DataView, instance::LabeledInstance},
    tasks::{CostSum, FloatCost, OptimizationTask},
};

#[derive(Clone)]
pub struct AccuracyTask {
    dataset_size: usize,
    num_labels: i32,
    branching_cost: f64,
    complexity_cost: f64,
}

impl AccuracyTask {
    pub fn new(complexity_cost: f64) -> Self {
        Self {
            dataset_size: 0,
            num_labels: 0,
            branching_cost: 0.0,
            complexity_cost,
        }
    }
}

pub struct AccuracyCostSum {
    instance_count_per_class: Vec<i32>,
}

impl Clone for AccuracyCostSum {
    fn clone(&self) -> Self {
        Self {
            instance_count_per_class: self.instance_count_per_class.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        assert_eq!(
            self.instance_count_per_class.len(),
            source.instance_count_per_class.len()
        );
        for i in 0..self.instance_count_per_class.len() {
            self.instance_count_per_class[i] = source.instance_count_per_class[i]
        }
    }
}

impl SubAssign<&AccuracyCostSum> for AccuracyCostSum {
    fn sub_assign(&mut self, rhs: &Self) {
        assert_eq!(
            self.instance_count_per_class.len(),
            rhs.instance_count_per_class.len()
        );
        for (count, other) in self
            .instance_count_per_class
            .iter_mut()
            .zip(rhs.instance_count_per_class.iter())
        {
            *count -= other;
            assert!(*count >= 0);
        }
    }
}

impl SubAssign<&LabeledInstance<i32>> for AccuracyCostSum {
    fn sub_assign(&mut self, rhs: &LabeledInstance<i32>) {
        self.instance_count_per_class[rhs.label as usize] -= 1;
        assert!(self.instance_count_per_class[rhs.label as usize] >= 0);
    }
}

impl AddAssign<&AccuracyCostSum> for AccuracyCostSum {
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(
            self.instance_count_per_class.len(),
            rhs.instance_count_per_class.len()
        );
        for (count, other) in self
            .instance_count_per_class
            .iter_mut()
            .zip(rhs.instance_count_per_class.iter())
        {
            *count += other;
        }
    }
}

impl AddAssign<&LabeledInstance<i32>> for AccuracyCostSum {
    fn add_assign(&mut self, rhs: &LabeledInstance<i32>) {
        self.instance_count_per_class[rhs.label as usize] += 1;
    }
}

impl CostSum<i32, LabeledInstance<i32>, FloatCost> for AccuracyCostSum {
    fn label(&self) -> i32 {
        self.instance_count_per_class
            .iter()
            .enumerate()
            .max_by_key(|(_, val)| *val)
            .map(|(idx, _)| idx)
            .expect("Expected at least one class") as i32
    }

    fn cost(&self) -> FloatCost {
        let (total, largest_class_size) = self
            .instance_count_per_class
            .iter()
            .fold((0, 0), |(acc_total, acc_max), e| {
                (acc_total + *e, acc_max.max(*e))
            });

        FloatCost((total - largest_class_size) as f64)
    }

    fn clear(&mut self) {
        for count in &mut self.instance_count_per_class {
            *count = 0;
        }
    }
}

impl OptimizationTask for AccuracyTask {
    type LabelType = i32;
    type InstanceType = LabeledInstance<i32>;
    type CostType = FloatCost;
    type CostSumType = AccuracyCostSum;

    fn prepare_for_data(&mut self, dataview: &mut DataView<Self>) {
        self.dataset_size = dataview.num_instances();
        self.num_labels = 0;
        for instance in &dataview.dataset.instances {
            self.num_labels = self.num_labels.max(instance.label + 1);
        }
        self.branching_cost = self.complexity_cost * dataview.num_instances() as f64
    }

    fn print_cost(&mut self, cost: &Self::CostType) -> String {
        format!(
            "Misclassifications: {}. Accuracy: {}%. (Only accurate when complexity cost is zero)",
            cost,
            (1.0 - cost.0 / self.dataset_size as f64) * 100.0
        )
    }

    fn init_costsum(dataset: &DataSet<Self::InstanceType>) -> Self::CostSumType {
        let mut num_labels = 0;
        for instance in &dataset.instances {
            num_labels = num_labels.max(instance.label + 1);
        }

        Self::CostSumType {
            instance_count_per_class: vec![0; num_labels as usize],
        }
    }

    fn branching_cost(&self) -> Self::CostType {
        self.branching_cost.into()
    }

    fn greedy_value(left_costsum: &Self::CostSumType, right_costsum: &Self::CostSumType) -> f32 {
        let left_total: i32 = left_costsum.instance_count_per_class.iter().sum();
        let right_total: i32 = right_costsum.instance_count_per_class.iter().sum();

        let left_gini = if left_total > 0 {
            left_costsum
                .instance_count_per_class
                .iter()
                .fold(1.0, |gini, &count| {
                    let probability = count as f32 / left_total as f32;
                    gini - probability * probability
                })
        } else {
            0.0
        };

        let right_gini = if right_total > 0 {
            right_costsum
                .instance_count_per_class
                .iter()
                .fold(1.0, |gini, &count| {
                    let probability = count as f32 / right_total as f32;
                    gini - probability * probability
                })
        } else {
            0.0
        };

        (left_gini * left_total as f32 + right_gini * right_total as f32)
            / (left_total + right_total) as f32
    }

    fn branch_relaxation(&self, dataview: &DataView<Self>, max_depth: u32) -> Self::CostType
    where
        Self: Sized,
    {
        if max_depth > 7 {
            FloatCost(0.0)
        } else {
            let mut counts = dataview.cost_summer.instance_count_per_class.clone();
            counts.sort_unstable();

            // The maximum number of clusters is the remaining leaf count or the number of instances remaining.
            let max_clusters = dataview.num_instances().min(1 << max_depth);

            // We try all cluster counts, and see how many misclassifications there are.
            let mut total_misclassifications = 0;
            let mut min_cost = f64::MAX;

            for i in 0..counts.len() {
                let clusters = counts.len() - i;
                if clusters < max_clusters {
                    min_cost = min_cost.min(
                        (clusters - 1) as f64 * self.branching_cost
                            + total_misclassifications as f64,
                    );
                }

                // Note that the last (largest) count is never included, as that is the label.
                total_misclassifications += counts[i];
            }

            FloatCost(min_cost)
        }
    }
}
