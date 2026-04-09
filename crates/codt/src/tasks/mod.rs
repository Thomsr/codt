use crate::model::dataset::DataSet;
use crate::model::dataview::FeatureValue;
use crate::model::{dataview::DataView, instance::Instance};

use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Range, Sub, SubAssign};

pub mod accuracy;

pub trait CostSum<LabelType, InstanceType, CostType>:
    for<'a> AddAssign<&'a Self>
    + for<'a> AddAssign<&'a InstanceType>
    + for<'a> SubAssign<&'a Self>
    + for<'a> SubAssign<&'a InstanceType>
    + Clone
{
    fn label(&self) -> LabelType;
    fn cost(&self) -> CostType;
    fn clear(&mut self);
}

/// The TryInto<f64> is only required for global best first search. May be implemented using `unimplemented` macro if not required.
pub trait Cost:
    Clone
    + Copy
    + Debug
    + Display
    + Add<Output = Self>
    + Sub<Output = Self>
    + TryInto<f64, Error: Debug>
{
    /// The minimum possible cost, to e.g. initialize lower bounds. Requires that ZERO_COST + ZERO_COST = ZERO_COST. For example 0 or 0.0
    const ZERO: Self;

    /// Returns true if the cost is zero. Use this for comparison to prevent floating point errors.
    fn is_zero(&self) -> bool;

    fn strictly_greater_than(&self, other: &Self) -> bool;
    fn strictly_less_than(&self, other: &Self) -> bool;

    fn greater_or_not_much_less_than(&self, other: &Self) -> bool;
    fn less_or_not_much_greater_than(&self, other: &Self) -> bool;

    /// Should only be used for sorting, e.g. in a priority queue.
    fn to_order(&self) -> impl Ord;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LexicographicCost {
    pub primary: i64,
    pub secondary: i64,
}

impl LexicographicCost {
    pub fn new(primary: i64, secondary: i64) -> Self {
        Self { primary, secondary }
    }
}

impl From<f64> for LexicographicCost {
    fn from(value: f64) -> Self {
        Self {
            primary: value.round() as i64,
            secondary: 0,
        }
    }
}

impl From<i64> for LexicographicCost {
    fn from(value: i64) -> Self {
        Self {
            primary: value,
            secondary: 0,
        }
    }
}

impl From<usize> for LexicographicCost {
    fn from(value: usize) -> Self {
        Self {
            primary: value as i64,
            secondary: 0,
        }
    }
}

impl Add for LexicographicCost {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            primary: self.primary + other.primary,
            secondary: self.secondary + other.secondary,
        }
    }
}

impl Sub for LexicographicCost {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            primary: self.primary - other.primary,
            secondary: self.secondary - other.secondary,
        }
    }
}

impl Display for LexicographicCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "(primary: {}, secondary: {})",
            self.primary, self.secondary
        )
    }
}

impl TryInto<f64> for LexicographicCost {
    type Error = &'static str;

    fn try_into(self) -> Result<f64, Self::Error> {
        Ok(self.primary as f64)
    }
}

impl Cost for LexicographicCost {
    const ZERO: Self = Self {
        primary: 0,
        secondary: 0,
    };

    fn is_zero(&self) -> bool {
        self.primary == 0 && self.secondary == 0
    }

    fn strictly_greater_than(&self, other: &Self) -> bool {
        self.primary > other.primary
            || (self.primary == other.primary && self.secondary > other.secondary)
    }

    fn strictly_less_than(&self, other: &Self) -> bool {
        self.primary < other.primary
            || (self.primary == other.primary && self.secondary < other.secondary)
    }

    fn greater_or_not_much_less_than(&self, other: &Self) -> bool {
        !self.strictly_less_than(other)
    }

    fn less_or_not_much_greater_than(&self, other: &Self) -> bool {
        !self.strictly_greater_than(other)
    }

    fn to_order(&self) -> impl Ord {
        (self.primary, self.secondary)
    }
}

pub trait OptimizationTask {
    /// The label type, e.g. a class label for classification tasks, or a regression target for regression tasks.
    type LabelType: Clone + Copy + Display + PartialEq + Eq + std::hash::Hash;
    /// The instance type. For classification and regression, each instance only has a label, see `LabeledInstance`.
    type InstanceType: Instance;
    type CostType: Cost;
    /// A type from which the cost is easily derivable. When a CostSum for disjoint datasets
    /// are summed, it results in the CostSum of their union.
    type CostSumType: CostSum<Self::LabelType, Self::InstanceType, Self::CostType>;
    type ExtraDataviewData;

    fn preprocess_dataset(dataset: &mut DataSet<Self::InstanceType>) {
        let _ = dataset;
    }

    /// Initialize a costsum, this should only be done once at the start.
    fn init_costsum(dataset: &DataSet<Self::InstanceType>) -> Self::CostSumType;

    fn label_of_instance(instance: &Self::InstanceType) -> Self::LabelType;

    fn prepare_for_data(&mut self, dataview: &mut DataView<Self>)
    where
        Self: Sized,
    {
        let _ = dataview;
    }
    fn print_cost(&mut self, cost: &Self::CostType) -> String;

    fn is_perfect_solution_cost(cost: &Self::CostType) -> bool
    where
        Self: Sized,
    {
        cost.is_zero()
    }

    fn to_cost_type(size: i64) -> Self::CostType;

    fn update_lowerbound(lb: &mut Self::CostType, candidate: &Self::CostType) {
        if candidate.strictly_greater_than(lb) {
            *lb = *candidate;
        }
    }

    fn update_upperbound(ub: &mut Self::CostType, candidate: &Self::CostType) {
        if candidate.strictly_less_than(ub) {
            *ub = *candidate;
        }
    }

    fn branching_cost(&self) -> Self::CostType;

    fn greedy_value(left_costsum: &Self::CostSumType, right_costsum: &Self::CostSumType) -> f32;

    fn init_extra_dataview_data(
        dataset: &DataSet<Self::InstanceType>,
        feature_values: &[Vec<FeatureValue>],
    ) -> Self::ExtraDataviewData;

    fn worst_cost_in_range(
        dataview: &DataView<Self>,
        feature: usize,
        range: Range<usize>,
    ) -> Self::CostType
    where
        Self: Sized;
}
