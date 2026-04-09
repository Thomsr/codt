use std::ops::Sub;

use crate::{
    model::{dataview::DataView, difference_table::DifferenceTable},
    tasks::OptimizationTask,
};
use gurobi::*;

pub fn pair_lower_bound<OT: OptimizationTask>(dataview: &DataView<'_, OT>) -> OT::CostType {
    let diff_table = DifferenceTable::new(dataview);

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

    model.optimize().unwrap();

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
    fn no_conflicts() {
        let features = vec![vec![0, 1, 2, 3]];
        let labels = vec![1, 1, 1, 1];

        let dataset = create_dataset(features, labels);
        let dataview = DataView::<AccuracyTask>::from_dataset(&dataset);

        let lb = pair_lower_bound(&dataview);

        assert_eq!(lb.secondary, 0);
    }

    #[test]
    fn single_feature_sufficient() {
        // One feature perfectly separates classes, LB = 1

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
    fn three_thresholds_required() {
        // Under threshold-based split columns, this dataset needs three splits to separate all pairs,
        // even though only two features are needed with general splits.
        // This tests that the pair LB correctly captures the need for multiple splits when using
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
}
