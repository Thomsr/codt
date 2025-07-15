use std::{convert::Infallible, marker::PhantomData, time::Duration};

use codt::{
    model::{dataset::DataSet, dataview::DataView, instance::LabeledInstance, tree::Tree},
    search::solver::{
        BranchRelaxation, SearchStrategyEnum, SolveResult, SolverOptions, TerminalSolver,
        UpperboundStrategy, solver_with_strategy,
    },
    tasks::{OptimizationTask, accuracy::AccuracyTask, squared_error::SquaredErrorTask},
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, ndarray::Axis};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};

fn tree_to_py<'a, 'py, OT: OptimizationTask, X>(
    tree: &'a Tree<OT>,
    py: Python<'py>,
) -> Result<Bound<'py, PyAny>, PyErr>
where
    OT::LabelType: IntoPyObject<'py, Target = X, Output = Bound<'py, X>, Error = Infallible>,
{
    Ok(match tree {
        Tree::Branch(b) => PyList::new(
            py,
            &[
                b.split_feature.into_pyobject(py)?.into_any(),
                b.split_threshold.into_pyobject(py)?.into_any(),
                tree_to_py(b.left_child.as_ref(), py)?,
                tree_to_py(b.right_child.as_ref(), py)?,
            ],
        )?
        .into_any(),
        Tree::Leaf(l) => l.label.into_pyobject(py)?.into_any(),
    })
}

macro_rules! impl_optimal_decision_tree_pyclass {
    ($pyclass:ident, $task:ty, $label:ty, $array:ty) => {
        #[pyclass]
        pub struct $pyclass {
            max_depth: u32,
            upperbound: UpperboundStrategy,
            terminal_solver: TerminalSolver,
            branch_relaxation: BranchRelaxation,
            timeout: Option<Duration>,
            memory_limit: Option<u64>,
            intermediates: bool,
            strategy: SearchStrategyEnum,
            task: $task,
            result: Option<SolveResult<$task>>,
            _phantom: PhantomData<$label>,
        }

        #[pymethods]
        #[allow(non_snake_case,clippy::too_many_arguments)]
        impl $pyclass {
            #[new]
            #[pyo3(signature = (
                max_depth=2,
                strategy="bfs-gosdt",
                complexity_cost=0.0,
                timeout=None,
                upperbound="for-remaining-interval",
                terminal_solver="left-right",
                intermediates=false,
                branch_relaxation="lowerbound",
                memory_limit=None
            ))]
            fn new(
                max_depth: u32,
                strategy: &str,
                complexity_cost: f64,
                timeout: Option<u64>,
                upperbound: &str,
                terminal_solver: &str,
                intermediates: bool,
                branch_relaxation: &str,
                memory_limit: Option<u64>,
            ) -> Result<Self, PyErr> {
                let strategy = strategy.parse().map_err(|_| {
                    PyValueError::new_err("Not a valid search strategy")
                })?;

                let upperbound = upperbound.parse().map_err(|_| {
                    PyValueError::new_err("Not a valid upper bounding strategy")
                })?;

                let terminal_solver = terminal_solver.parse().map_err(|_| {
                    PyValueError::new_err("Not a valid terminal solver")
                })?;

                let branch_relaxation = branch_relaxation.parse().map_err(|_| {
                    PyValueError::new_err("Not a valid branch relaxation strategy")
                })?;

                Ok(Self {
                    max_depth,
                    upperbound,
                    terminal_solver,
                    strategy,
                    intermediates,
                    timeout: timeout.map(Duration::from_secs),
                    branch_relaxation,
                    memory_limit,
                    task: <$task>::new(complexity_cost),
                    result: None,
                    _phantom: PhantomData,
                })
            }

            #[pyo3(signature = (X, y))]
            fn fit<'py>(&mut self, X: PyReadonlyArray2<'py, f64>, y: PyReadonlyArray1<'py, $label>) {
                let x_arr = X.as_array();
                let y_arr = y.as_array();
                let mut dataset = DataSet::<LabeledInstance<<$task as OptimizationTask>::LabelType>>::default();
                for i in 0..y_arr.dim() {
                    let features = x_arr.index_axis(Axis(0), i);
                    dataset.add_instance(
                        LabeledInstance::new(y_arr[i].try_into().unwrap()),
                        features.iter().copied(),
                    );
                }
                dataset.preprocess_after_adding_instances();

                <$task>::preprocess_dataset(&mut dataset);
                let full_view = DataView::from_dataset(&dataset);

                let options = SolverOptions {
                    max_depth: self.max_depth,
                    ub_strategy: self.upperbound,
                    terminal_solver: self.terminal_solver,
                    branch_relaxation: self.branch_relaxation,
                    timeout: self.timeout,
                    track_intermediates: self.intermediates,
                    memory_limit: self.memory_limit,
                };

                let mut solver = solver_with_strategy(
                    self.task.clone(),
                    full_view,
                    self.strategy,
                );

                self.result = Some(solver.solve(options));
            }

            #[pyo3(signature = (X))]
            fn predict<'py>(
                &self,
                py: Python<'py>,
                X: PyReadonlyArray2<'py, f64>,
            ) -> Bound<'py, $array> {
                let tree = &self
                    .result
                    .as_ref()
                    .expect(".fit(X,y) before .predict(X) should be checked in the python wrapper")
                    .tree;
                X.as_array()
                    .map_axis(Axis(1), |x| {
                        tree.predict(x.iter().copied().collect()).into()
                    })
                    .into_pyarray(py)
            }

            #[pyo3(signature = ())]
            fn tree<'py>(&mut self, py: Python<'py>) -> Result<Bound<'py, PyAny>, PyErr> {
                let tree = &self
                    .result
                    .as_ref()
                    .expect(".fit(X,y) should be called before this function and should be checked in the python wrapper")
                    .tree;

                tree_to_py(tree, py)
            }

            #[pyo3(signature = ())]
            fn intermediate_lbs<'py>(&mut self, py: Python<'py>) -> Result<Bound<'py, PyAny>, PyErr> {
                let intermediates = &self
                    .result
                    .as_ref()
                    .expect(".fit(X,y) should be called before this function and should be checked in the python wrapper")
                    .intermediate_lbs
                    .iter()
                    .map(|(lb, exp, time)| {
                        let lb_float: f64 = (*lb).into();
                        (
                            lb_float,
                            exp,
                            time,
                        )
                    })
                    .collect::<Vec<_>>();
                intermediates.into_pyobject(py)
            }

            #[pyo3(signature = ())]
            fn intermediate_ubs<'py>(&mut self, py: Python<'py>) -> Result<Bound<'py, PyAny>, PyErr> {
                let intermediates = &self
                    .result
                    .as_ref()
                    .expect(".fit(X,y) should be called before this function and should be checked in the python wrapper")
                    .intermediate_ubs
                    .iter()
                    .map(|(ub, exp, time)| {
                        let ub_float: f64 = (*ub).into();
                        (
                            ub_float,
                            exp,
                            time,
                        )
                    })
                    .collect::<Vec<_>>();
                intermediates.into_pyobject(py)
            }

            #[pyo3(signature = ())]
            fn expansions(&mut self) -> i64 {
                self
                    .result
                    .as_ref()
                    .expect(".fit(X,y) should be called before this function and should be checked in the python wrapper")
                    .graph_expansions as i64
            }
        }
    };
}

impl_optimal_decision_tree_pyclass!(
    OptimalDecisionTreeClassifier,
    AccuracyTask,
    i64,
    PyArray1<i64>
);

impl_optimal_decision_tree_pyclass!(
    OptimalDecisionTreeRegressor,
    SquaredErrorTask,
    f64,
    PyArray1<f64>
);
