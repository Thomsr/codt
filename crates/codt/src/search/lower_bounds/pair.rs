use std::{ops::Sub, time::Instant};

use crate::{
    model::{dataview::DataView, difference_table::DifferenceTable},
    tasks::OptimizationTask,
};
use gurobi::*;

pub fn pair_lower_bound<OT: OptimizationTask>(dataview: &DataView<'_, OT>) -> OT::CostType {
    let diff_table_start_time = Instant::now();
    let diff_table = DifferenceTable::new(dataview);
    let diff_table_duration = diff_table_start_time.elapsed();

    log::debug!(
        "Constructed difference table with {} pairs and {} split columns in {:.2?}",
        diff_table.pairs.len(),
        diff_table.n_columns,
        diff_table_duration
    );

    if diff_table.diffs.is_empty() {
        return OT::to_cost_type(0);
    }

    let mut env = Env::new("").unwrap();
    env.set(param::OutputFlag, 0).unwrap();
    env.set(param::LogToConsole, 0).unwrap();
    let mut model = Model::new("pair_lb_relaxation", &env).unwrap();

    let n_columns = diff_table.n_columns;

    let x_vars: Vec<Var> = (0..n_columns)
        .map(|column| {
            model
                .add_var(
                    &format!("x_{}", column),
                    VarType::Continuous,
                    0.0,
                    0.0,
                    1.0,
                    &[],
                    &[],
                )
                .unwrap()
        })
        .collect();

    let mut obj = LinExpr::new();
    for var in &x_vars {
        obj = obj.add_term(1.0, var.clone());
    }

    for (p_idx, diffs) in diff_table.diffs.iter().enumerate() {
        let mut expr = LinExpr::new();

        for (f, &d) in diffs.iter().enumerate() {
            if d {
                expr = expr.add_term(1.0, x_vars[f].clone());
            }
        }

        model
            .add_constr(
                &format!("cover_pair_{}", p_idx),
                expr,
                ConstrSense::Greater,
                1.0,
            )
            .unwrap();
    }

    model.update().unwrap();

    model.set_objective(obj, Minimize).unwrap();

    let optimization_start_time = Instant::now();
    model.optimize().unwrap();
    let optimization_duration = optimization_start_time.elapsed();

    log::debug!(
        "Solved pair LB relaxation in {:.2?}, objective value = {}",
        optimization_duration,
        model.get(attr::ObjVal).unwrap()
    );

    let obj_val = model.get(attr::ObjVal).unwrap();

    // Small delta to ensure we round up correctly when the LP solution is fractional.
    let delta = 1e-5;
    let lb = obj_val.sub(delta).ceil() as i64;

    OT::to_cost_type(lb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::dataset::DataSet;
    use crate::model::dataview::DataView;
    use crate::model::instance::LabeledInstance;
    use crate::tasks::accuracy::AccuracyTask;

    /// Helper to build a dataset with multiple features
    fn create_dataset(features: Vec<Vec<i32>>, labels: Vec<i32>) -> DataSet<LabeledInstance<i32>> {
        let mut dataset = DataSet::default();

        for label in labels {
            dataset.instances.push(LabeledInstance::new(label));
        }

        for feature_col in features {
            dataset.feature_values.push(feature_col);
        }

        dataset
    }

    #[test]
    fn pair_lb_no_conflicts() {
        // All labels identical → no conflicting pairs → LB = 0

        let features = vec![vec![0, 1, 2, 3]];
        let labels = vec![1, 1, 1, 1];

        let dataset = create_dataset(features, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);

        let lb = pair_lower_bound(&dataview);

        assert_eq!(lb.secondary, 0);
    }

    #[test]
    fn pair_lb_single_feature_sufficient() {
        // One feature perfectly separates classes → LB = 1

        // Feature: 0 1 2 3
        // Labels:  0 0 1 1

        let features = vec![vec![0, 1, 2, 3]];
        let labels = vec![0, 0, 1, 1];

        let dataset = create_dataset(features, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);

        let lb = pair_lower_bound(&dataview);

        assert_eq!(lb.secondary, 1);
    }

    #[test]
    fn pair_lb_three_thresholds_required() {
        // Under threshold-based split columns, this dataset needs three distinct
        // thresholds to separate all conflicting pairs.

        // Feature 1: 0 1 2 3
        // Feature 2: 0 0 1 1
        // Labels:    0 1 0 1

        let features = vec![vec![0, 1, 2, 3], vec![0, 0, 1, 1]];
        let labels = vec![0, 1, 0, 1];

        let dataset = create_dataset(features, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);

        let lb = pair_lower_bound(&dataview);

        assert_eq!(lb.secondary, 3);
    }

    #[test]
    fn pair_lb_fractional_relaxation_ceiling() {
        // Classic case where LP relaxation is fractional but rounds up

        // Feature 1: separates some pairs
        // Feature 2: separates the rest
        //
        // LP may assign ~0.5 to each → objective ≈ 1.0 → ceil = 1

        let features = vec![vec![0, 0, 1, 1], vec![0, 1, 0, 1]];
        let labels = vec![0, 1, 2, 3]; // all different → all pairs conflict

        let dataset = create_dataset(features, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);

        let lb = pair_lower_bound(&dataview);

        // At least 1 feature is needed (though integrally it's also ≥1)
        assert!(lb.secondary >= 1);
    }

    #[test]
    fn pair_lb_two_features_fractional_symmetry() {
        // Symmetric 4-class dataset where all class pairs must be separated.
        // With this construction, both features are required.

        let features = vec![vec![0, 1, 0, 1], vec![0, 0, 1, 1]];
        let labels = vec![0, 1, 2, 3];

        let dataset = create_dataset(features, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);

        let lb = pair_lower_bound(&dataview);

        assert_eq!(lb.secondary, 2);
    }
}
