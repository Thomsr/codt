use std::ops::{AddAssign, SubAssign};

use crate::model::{dataset::DataSet, dataview::DataView, instance::LabeledInstance};

use super::OptimizationTask;

#[derive(Default)]
pub struct AccuracyTask {
    dataset_size: usize,
    num_labels: i32,
}

#[derive(Clone)]
pub struct AccuracyCostSummer {
    instance_count_per_class: Vec<i32>,
}

impl SubAssign<&AccuracyCostSummer> for AccuracyCostSummer {
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
        }
    }
}

impl SubAssign<&LabeledInstance<i32>> for AccuracyCostSummer {
    fn sub_assign(&mut self, rhs: &LabeledInstance<i32>) {
        self.instance_count_per_class[rhs.label as usize] -= 1;
    }
}

impl AddAssign<&AccuracyCostSummer> for AccuracyCostSummer {
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

impl AddAssign<&LabeledInstance<i32>> for AccuracyCostSummer {
    fn add_assign(&mut self, rhs: &LabeledInstance<i32>) {
        self.instance_count_per_class[rhs.label as usize] += 1;
    }
}

impl From<&AccuracyCostSummer> for i32 {
    fn from(value: &AccuracyCostSummer) -> Self {
        let (total, largest_class_size) = value
            .instance_count_per_class
            .iter()
            .fold((0, 0), |(acc_total, acc_max), e| {
                (acc_total + *e, acc_max.max(*e))
            });

        total - largest_class_size
    }
}

impl OptimizationTask for AccuracyTask {
    type InstanceType = LabeledInstance<i32>;
    type CostType = i32;
    type CostSummer = AccuracyCostSummer;
    const MIN_COST: Self::CostType = 0;

    fn prepare_for_data(&mut self, dataview: &mut DataView<Self>) {
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

    fn init_costsum(dataset: &DataSet<Self::InstanceType>) -> Self::CostSummer {
        let mut num_labels = 0;
        for instance in &dataset.instances {
            num_labels = num_labels.max(instance.label + 1);
        }

        Self::CostSummer {
            instance_count_per_class: vec![0; num_labels as usize],
        }
    }
}
