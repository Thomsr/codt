use codt::{
    model::{dataset::DataSet, dataview::DataView, instance::LabeledInstance},
    search::{
        solver::{SolveResult, Solver},
        strategy::{andor::AndOrSearchStrategy, dfs::DfsSearchStrategy},
    },
    tasks::{OptimizationTask, accuracy::AccuracyTask},
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, ndarray::Axis};
use pyo3::prelude::*;

#[pyclass(eq, eq_int)]
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum SearchStrategyEnum {
    Dfs,
    AndOr,
}

#[pyclass]
pub struct OptimalDecisionTreeClassifier {
    max_depth: u32,
    strategy: SearchStrategyEnum,
    result: Option<SolveResult<AccuracyTask>>,
}

#[pymethods]
#[allow(non_snake_case)]
impl OptimalDecisionTreeClassifier {
    #[new]
    #[pyo3(signature = (max_depth=2, strategy=SearchStrategyEnum::Dfs))]
    fn new(max_depth: u32, strategy: SearchStrategyEnum) -> OptimalDecisionTreeClassifier {
        OptimalDecisionTreeClassifier {
            max_depth,
            strategy,
            result: None,
        }
    }

    #[pyo3(signature = (X, y))]
    fn fit<'py>(&mut self, X: PyReadonlyArray2<'py, f64>, y: PyReadonlyArray1<'py, i64>) {
        let x_arr = X.as_array();
        let y_arr = y.as_array();
        let mut dataset = DataSet::<LabeledInstance<i32>>::default();
        for i in 0..y_arr.dim() {
            let features = x_arr.index_axis(Axis(0), i);
            dataset.add_instance(
                LabeledInstance::new(y_arr[i] as i32),
                features.iter().copied(),
            );
        }
        dataset.preprocess_after_adding_instances();

        AccuracyTask::preprocess_dataset(&mut dataset);
        let task = AccuracyTask::default();
        let full_view = DataView::from_dataset(&dataset);

        let result = match self.strategy {
            SearchStrategyEnum::Dfs => {
                let mut solver: Solver<'_, AccuracyTask, DfsSearchStrategy> =
                    Solver::new(task, full_view);
                solver.solve(self.max_depth)
            }
            SearchStrategyEnum::AndOr => {
                let mut solver: Solver<'_, AccuracyTask, AndOrSearchStrategy> =
                    Solver::new(task, full_view);
                solver.solve(self.max_depth)
            }
        };

        self.result = Some(result);
    }

    #[pyo3(signature = (X))]
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        X: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray1<i64>> {
        let tree = &self
            .result
            .as_ref()
            .expect(".fit(X,y) before .predict(X) should be checked in the python wrapper")
            .tree;
        X.as_array()
            .map_axis(Axis(1), |x| {
                tree.predict(x.iter().copied().collect()) as i64
            })
            .into_pyarray(py)
    }
}
