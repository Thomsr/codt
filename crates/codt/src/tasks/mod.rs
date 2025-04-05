pub mod classification;
pub mod regression;

use crate::model::dataset::DataSet;
use crate::model::{dataview::DataView, instance::Instance};

use std::fmt::Debug;
use std::ops::Add;

pub trait OptimizationTask {
    type InstanceType: Instance;
    type CostType: PartialOrd + Add<Output = Self::CostType> + Debug;

    fn preprocess_dataset(dataset: &mut DataSet<Self::InstanceType>) {
        let _ = dataset;
    }
    fn prepare_for_data(&mut self, dataview: &mut DataView<Self::InstanceType>) {
        let _ = dataview;
    }
    fn leaf_cost(&self, dataview: &DataView<Self::InstanceType>) -> Self::CostType;
    fn print_cost(&mut self, cost: &Self::CostType) -> String;
}
