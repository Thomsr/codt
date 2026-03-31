use std::ops::{AddAssign, Range, SubAssign};

use crate::{
    model::{
        dataset::DataSet,
        dataview::{DataView, FeatureValue},
        instance::LabeledInstance,
    },
    tasks::{CostSum, LexicographicCost, OptimizationTask},
};

#[derive(Clone)]
pub struct AccuracyTask {
    dataset_size: usize,
}

impl AccuracyTask {
    pub fn new() -> Self {
        Self {
            dataset_size: 0,
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

impl CostSum<i32, LabeledInstance<i32>, LexicographicCost> for AccuracyCostSum {
    fn label(&self) -> i32 {
        self.instance_count_per_class
            .iter()
            .enumerate()
            .max_by_key(|(_, val)| *val)
            .map(|(idx, _)| idx)
            .expect("Expected at least one class") as i32
    }

    fn cost(&self) -> LexicographicCost {
        let (total, largest_class_size) = self
            .instance_count_per_class
            .iter()
            .fold((0, 0), |(acc_total, acc_max), e| {
                (acc_total + *e, acc_max.max(*e))
            });

        LexicographicCost::new((total - largest_class_size) as i64, 0)
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
    type CostType = LexicographicCost;
    type CostSumType = AccuracyCostSum;
    type ExtraDataviewData = ();

    fn prepare_for_data(&mut self, dataview: &mut DataView<Self>) {
        self.dataset_size = dataview.num_instances();
    }

    fn print_cost(&mut self, cost: &Self::CostType) -> String {
        let accuracy = (1.0 - cost.primary as f64 / self.dataset_size as f64) * 100.0;
        format!(
            "Misclassifications: {}. Branch nodes: {}. Accuracy: {}%.",
            cost.primary, cost.secondary, accuracy
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
        LexicographicCost::new(0, 1)
    }

    fn is_perfect_solution_cost(cost: &Self::CostType) -> bool
    where
        Self: Sized,
    {
        cost.primary == 0
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

    fn worst_cost_in_range(
        _dataview: &DataView<Self>,
        _feature: usize,
        range: Range<usize>,
    ) -> Self::CostType
    where
        Self: Sized,
    {
        LexicographicCost::new(range.len() as i64, 0)
    }

    fn init_extra_dataview_data(
        _dataset: &DataSet<Self::InstanceType>,
        _feature_values: &[Vec<FeatureValue>],
    ) -> Self::ExtraDataviewData {
    }
}