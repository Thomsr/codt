use crate::model::dataset::DataSet;
use crate::model::{dataview::DataView, instance::Instance};

use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Sub, SubAssign};

pub mod accuracy;
pub mod squared_error;

pub trait CostSum<LabelType, InstanceType, CostType>:
    for<'a> AddAssign<&'a Self>
    + for<'a> AddAssign<&'a InstanceType>
    + for<'a> SubAssign<&'a Self>
    + for<'a> SubAssign<&'a InstanceType>
    + Clone
{
    fn label(&self) -> LabelType;
    fn cost(&self) -> CostType;
}

pub trait OptimizationTask {
    type LabelType: Clone + Copy + Display;
    type InstanceType: Instance;
    /// The TryInto<f64> is only required for global best first search. May be implemented using `unimplemented` macro if not required.
    type CostType: Clone
        + Copy
        + PartialOrd
        + Add<Output = Self::CostType>
        + Sub<Output = Self::CostType>
        + TryInto<f64, Error: Debug>
        + Debug;
    /// A type from which the cost is easily derivable. When a CostSum for disjoint datasets
    /// are summed, it results in the CostSum of their union.
    type CostSumType: CostSum<Self::LabelType, Self::InstanceType, Self::CostType>;
    /// The minimum possible cost, to e.g. initialize lower bounds. Requires that ZERO_COST + ZERO_COST = ZERO_COST. For example 0 or 0.0
    const ZERO_COST: Self::CostType;

    fn preprocess_dataset(dataset: &mut DataSet<Self::InstanceType>) {
        let _ = dataset;
    }

    /// Initialize a costsum, this should only be done once at the start.
    fn init_costsum(dataset: &DataSet<Self::InstanceType>) -> Self::CostSumType;

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

    fn branching_cost(&self) -> Self::CostType;
}
