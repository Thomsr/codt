use crate::model::dataset::DataSet;
use crate::model::{dataview::DataView, instance::Instance};

use std::fmt::Debug;
use std::ops::{Add, AddAssign, SubAssign};

pub mod accuracy;
pub mod regression;

pub trait OptimizationTask {
    type InstanceType: Instance;
    type CostType: Clone
        + Copy
        + PartialOrd
        + Add<Output = Self::CostType>
        + Debug
        + for<'a> From<&'a Self::CostSummer>;
    /// A type from which the cost is easily derivable, and that can be summed and subtracted easily.
    type CostSummer: for<'a> AddAssign<&'a Self::CostSummer>
        + for<'a> AddAssign<&'a Self::InstanceType>
        + for<'a> SubAssign<&'a Self::CostSummer>
        + for<'a> SubAssign<&'a Self::InstanceType>
        + Clone;
    /// The minimum possible cost, to e.g. initialize lower bounds. Usually zero.
    const MIN_COST: Self::CostType;

    fn preprocess_dataset(dataset: &mut DataSet<Self::InstanceType>) {
        let _ = dataset;
    }

    /// Initialize a costsum, this should only be done once at the start.
    fn init_costsum(dataset: &DataSet<Self::InstanceType>) -> Self::CostSummer;

    fn prepare_for_data(&mut self, dataview: &mut DataView<Self>)
    where
        Self: Sized,
    {
        let _ = dataview;
    }
    fn print_cost(&mut self, cost: &Self::CostType) -> String;

    fn update_lowerbound(lb: &mut Self::CostType, candidate: &Self::CostType) {
        if candidate > lb {
            *lb = *candidate;
        }
    }

    fn update_upperbound(ub: &mut Self::CostType, candidate: &Self::CostType) {
        if candidate < ub {
            *ub = *candidate;
        }
    }
}
