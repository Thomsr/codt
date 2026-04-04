use crate::tasks::OptimizationTask;

pub fn class_count_lower_bound<OT: OptimizationTask>(nr_classes: usize) -> OT::CostType {
    OT::lower_bound_for_num_labels(nr_classes)
}
