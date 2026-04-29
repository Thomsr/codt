use codt::{
    model::{dataset::DataSet, dataview::DataView, instance::LabeledInstance},
    search::solver::{SearchStrategyEnum, SolveStatus, SolverOptions, solver_with_strategy},
    tasks::accuracy::AccuracyTask,
};

fn build_dataset() -> DataSet<LabeledInstance<i32>> {
    let mut ds = DataSet::default();
    ds.add_instance(LabeledInstance::new(0), [0.0, 0.0, 1.0]);
    ds.add_instance(LabeledInstance::new(0), [0.0, 0.0, 1.0]);
    ds.add_instance(LabeledInstance::new(1), [1.0, 1.0, 0.0]);
    ds.add_instance(LabeledInstance::new(1), [2.0, 2.0, 0.0]);
    ds.preprocess_after_adding_instances();
    ds
}

#[test]
fn reduction_toggle_changes_view_size() {
    let dataset = build_dataset();

    let reduced = DataView::<AccuracyTask>::from_dataset(&dataset, true);
    let unreduced = DataView::<AccuracyTask>::from_dataset(&dataset, false);

    assert!(reduced.num_instances() <= unreduced.num_instances());
    assert!(reduced.num_features() <= unreduced.num_features());

    let mut solver_reduced = solver_with_strategy(
        AccuracyTask::new(),
        reduced,
        SearchStrategyEnum::BfsBalanceSmallLb,
    );
    let mut solver_unreduced = solver_with_strategy(
        AccuracyTask::new(),
        unreduced,
        SearchStrategyEnum::BfsBalanceSmallLb,
    );

    let result_reduced = solver_reduced.solve(SolverOptions::default());
    let result_unreduced = solver_unreduced.solve(SolverOptions::default());

    assert_eq!(result_reduced.status, SolveStatus::PerfectTreeFound);
    assert_eq!(result_unreduced.status, SolveStatus::PerfectTreeFound);
}

#[test]
fn merged_feature_split_maps_to_original_feature() {
    let dataset = build_dataset();
    let reduced = DataView::<AccuracyTask>::from_dataset(&dataset, true);

    if reduced.num_features() == 0 || reduced.possible_split_values[0].is_empty() {
        return;
    }

    let original_feature = reduced.original_split_feature_from_split(0, 0);
    assert!(original_feature < dataset.feature_values.len());

    let threshold = reduced.threshold_from_split(0, 0);
    assert!(threshold.is_finite());
}
