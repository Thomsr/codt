use std::cmp;

use crate::tasks::OptimizationTask;

#[inline]
pub fn class_count_lower_bound<OT: OptimizationTask>(num_classes: usize) -> OT::CostType {
    OT::to_cost_type(cmp::max(0, num_classes.saturating_sub(1)) as i64)
}
