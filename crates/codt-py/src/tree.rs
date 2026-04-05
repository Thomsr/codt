use std::{
    collections::HashSet,
    convert::{Infallible, TryInto},
    marker::PhantomData,
    time::Duration,
};

use codt::{
    model::{dataset::DataSet, dataview::DataView, instance::LabeledInstance, tree::Tree},
    search::solver::{
        LowerBoundStrategy, SearchStrategyEnum, SolveResult, SolverOptions, UpperboundStrategy,
        solver_with_strategy,
    },
    tasks::{OptimizationTask, accuracy::AccuracyTask},
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, ndarray::Axis};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};

const ERROR_FIT_NOT_CALLED: &str =
    ".fit(X,y) should be called before this function and should be checked in the python wrapper";

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

fn tree_size<OT: OptimizationTask>(tree: &Tree<OT>) -> usize {
    match tree {
        Tree::Branch(b) => 1 + tree_size(b.left_child.as_ref()) + tree_size(b.right_child.as_ref()),
        Tree::Leaf(_) => 1,
    }
}

macro_rules! impl_optimal_decision_tree_pyclass {
    ($pyclass:ident, $task:ty, $label:ty, $array:ty) => {
        #[pyclass]
        pub struct $pyclass {
            lowerbound: HashSet<LowerBoundStrategy>,
            upperbound: UpperboundStrategy,
            timeout: Option<Duration>,
            memory_limit: Option<u64>,
            intermediates: bool,
            strategy: SearchStrategyEnum,
            task: $task,
            result: Option<SolveResult<$task>>,
            _phantom: PhantomData<$label>,
        }

        #[pymethods]
        #[allow(non_snake_case, clippy::too_many_arguments)]
        impl $pyclass {
            #[new]
            #[pyo3(signature = (
                                                                strategy="bfs-balance-small-lb",
                                                                timeout=None,
                                                                lowerbound="class-count",
                                                                upperbound="for-remaining-interval",
                                                                intermediates=false,
                                                                memory_limit=None
                                                            ))]
            fn new(
                strategy: &str,
                timeout: Option<u64>,
                lowerbound: &str,
                upperbound: &str,
                intermediates: bool,
                memory_limit: Option<u64>,
            ) -> Result<Self, PyErr> {
                let strategy = strategy
                    .parse()
                    .map_err(|_| PyValueError::new_err("Not a valid search strategy"))?;

                let upperbound = upperbound
                    .parse()
                    .map_err(|_| PyValueError::new_err("Not a valid upper bounding strategy"))?;

                let lowerbound = lowerbound
                    .parse()
                    .map_err(|_| PyValueError::new_err("Not a valid lower bounding strategy"))?;

                Ok(Self {
                    lowerbound: HashSet::from([lowerbound]),
                    upperbound,
                    strategy,
                    intermediates,
                    timeout: timeout.map(Duration::from_secs),
                    memory_limit,
                    task: <$task>::new(),
                    result: None,
                    _phantom: PhantomData,
                })
            }

            #[pyo3(signature = (X, y))]
            fn fit<'py>(
                &mut self,
                X: PyReadonlyArray2<'py, f64>,
                y: PyReadonlyArray1<'py, $label>,
            ) {
                let x_arr = X.as_array();
                let y_arr = y.as_array();
                let mut dataset =
                    DataSet::<LabeledInstance<<$task as OptimizationTask>::LabelType>>::default();
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
                    lb_strategy: self.lowerbound.clone(),
                    ub_strategy: self.upperbound,
                    timeout: self.timeout,
                    track_intermediates: self.intermediates,
                    memory_limit: self.memory_limit,
                };

                let mut solver = solver_with_strategy(self.task.clone(), full_view, self.strategy);

                self.result = Some(solver.solve(options));
            }

            #[pyo3(signature = (X))]
            fn predict<'py>(
                &self,
                py: Python<'py>,
                X: PyReadonlyArray2<'py, f64>,
            ) -> Result<Bound<'py, $array>, PyErr> {
                let result = self.result.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(ERROR_FIT_NOT_CALLED)
                })?;
                let tree = result.tree.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(ERROR_FIT_NOT_CALLED)
                })?;
                let tree = tree.as_ref();

                Ok(X
                    .as_array()
                    .map_axis(Axis(1), |x| {
                        tree.predict(x.iter().copied().collect()).into()
                    })
                    .into_pyarray(py))
            }

            #[pyo3(signature = ())]
            fn tree<'py>(&mut self, py: Python<'py>) -> Result<Bound<'py, PyAny>, PyErr> {
                let result = self.result.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(ERROR_FIT_NOT_CALLED)
                })?;
                let tree = result.tree.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(ERROR_FIT_NOT_CALLED)
                })?;
                let tree = tree.as_ref();
                tree_to_py(tree, py)
            }

            #[pyo3(signature = ())]
            fn status(&self) -> &'static str {
                match self.result.as_ref().expect(ERROR_FIT_NOT_CALLED).status {
                    codt::search::solver::SolveStatus::PerfectTreeFound => "perfect-tree-found",
                    codt::search::solver::SolveStatus::NoPerfectTree => "no-perfect-tree",
                }
            }

            #[pyo3(signature = ())]
            fn is_perfect(&self) -> bool {
                matches!(
                    self.result.as_ref().expect(ERROR_FIT_NOT_CALLED).status,
                    codt::search::solver::SolveStatus::PerfectTreeFound
                )
            }

            #[pyo3(signature = ())]
            fn branch_count(&self) -> i64 {
                self.result
                    .as_ref()
                    .expect(ERROR_FIT_NOT_CALLED)
                    .tree
                    .as_ref()
                    .expect(ERROR_FIT_NOT_CALLED)
                    .branch_count() as i64
            }

            #[pyo3(signature = ())]
            fn tree_size(&self) -> i64 {
                tree_size(
                    self.result
                        .as_ref()
                        .expect(ERROR_FIT_NOT_CALLED)
                        .tree
                        .as_ref()
                        .expect(ERROR_FIT_NOT_CALLED)
                        .as_ref(),
                ) as i64
            }

            #[pyo3(signature = ())]
            fn intermediate_lbs<'py>(
                &mut self,
                py: Python<'py>,
            ) -> Result<Bound<'py, PyAny>, PyErr> {
                let intermediates = &self
                    .result
                    .as_ref()
                    .expect(ERROR_FIT_NOT_CALLED)
                    .intermediate_lbs
                    .iter()
                    .map(|(lb, exp, time)| {
                        let lb_float: f64 = (*lb).try_into().expect("Cost should convert to f64");
                        (lb_float, exp, time)
                    })
                    .collect::<Vec<_>>();
                intermediates.into_pyobject(py)
            }

            #[pyo3(signature = ())]
            fn intermediate_ubs<'py>(
                &mut self,
                py: Python<'py>,
            ) -> Result<Bound<'py, PyAny>, PyErr> {
                let intermediates = &self
                    .result
                    .as_ref()
                    .expect(ERROR_FIT_NOT_CALLED)
                    .intermediate_ubs
                    .iter()
                    .map(|(ub, exp, time)| {
                        let ub_float: f64 = (*ub).try_into().expect("Cost should convert to f64");
                        (ub_float, exp, time)
                    })
                    .collect::<Vec<_>>();
                intermediates.into_pyobject(py)
            }

            #[pyo3(signature = ())]
            fn expansions(&mut self) -> i64 {
                self.result
                    .as_ref()
                    .expect(ERROR_FIT_NOT_CALLED)
                    .graph_expansions as i64
            }

            #[pyo3(signature = ())]
            fn memory_usage_bytes(&mut self) -> i64 {
                self.result
                    .as_ref()
                    .expect(ERROR_FIT_NOT_CALLED)
                    .memory_usage_bytes
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
