use std::{convert::Infallible, marker::PhantomData, time::Duration};

use codt::{
    model::{dataset::DataSet, dataview::DataView, instance::LabeledInstance, tree::Tree},
    search::{
        solver::{SolveResult, Solver, SolverOptions, TerminalSolver, UpperboundStrategy},
        strategy::{
            andor::AndOrSearchStrategy,
            bfs::{BfsSearchStrategy, CuriosityHeuristic, GOSDTHeuristic, LBHeuristic},
            dfs::DfsSearchStrategy,
            dfsprio::DfsPrioSearchStrategy,
        },
    },
    tasks::{OptimizationTask, accuracy::AccuracyTask, squared_error::SquaredErrorTask},
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, ndarray::Axis};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};
use strum_macros::EnumString;

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

#[pyclass(eq, eq_int)]
#[derive(PartialEq, Eq, Clone, Copy, EnumString)]
#[strum(serialize_all = "snake_case")]
pub enum SearchStrategyEnum {
    Dfs,
    AndOr,
    BfsLb,
    BfsCuriosity,
    BfsGosdt,
    DfsPrio,
}

#[pyclass(eq, eq_int)]
#[derive(PartialEq, Eq, Clone, Copy, EnumString)]
#[strum(serialize_all = "snake_case")]
pub enum UpperboundStrategyEnum {
    SolutionsOnly,
    TightFromSibling,
    ForRemainingInterval,
}

#[pyclass(eq, eq_int)]
#[derive(PartialEq, Eq, Clone, Copy, EnumString)]
#[strum(serialize_all = "snake_case")]
pub enum TerminalSolverEnum {
    Leaf,
    LeftRight,
    D2,
}

#[pyfunction(signature = (strategy=""))]
pub fn search_strategy_from_string(strategy: &str) -> Result<SearchStrategyEnum, PyErr> {
    strategy
        .parse()
        .map_err(|_| PyValueError::new_err("Not a valid search strategy"))
}

#[pyfunction(signature = (strategy=""))]
pub fn ub_from_string(strategy: &str) -> Result<UpperboundStrategyEnum, PyErr> {
    strategy
        .parse()
        .map_err(|_| PyValueError::new_err("Not a valid upper bounding strategy"))
}

#[pyfunction(signature = (solver=""))]
pub fn terminal_solver_from_string(solver: &str) -> Result<TerminalSolverEnum, PyErr> {
    solver
        .parse()
        .map_err(|_| PyValueError::new_err("Not a valid terminal solver"))
}

macro_rules! impl_optimal_decision_tree_pyclass {
    ($pyclass:ident, $task:ty, $label:ty, $array:ty) => {
        #[pyclass]
        pub struct $pyclass {
            max_depth: u32,
            upperbound: UpperboundStrategyEnum,
            terminal_solver: TerminalSolverEnum,
            node_lowerbound: bool,
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
                strategy=SearchStrategyEnum::Dfs,
                complexity_cost=0.0,
                timeout=None,
                upperbound=UpperboundStrategyEnum::ForRemainingInterval,
                terminal_solver=TerminalSolverEnum::LeftRight,
                intermediates=false,
                node_lowerbound=true,
                memory_limit=None
            ))]
            fn new(
                max_depth: u32,
                strategy: SearchStrategyEnum,
                complexity_cost: f64,
                timeout: Option<u64>,
                upperbound: UpperboundStrategyEnum,
                terminal_solver: TerminalSolverEnum,
                intermediates: bool,
                node_lowerbound: bool,
                memory_limit: Option<u64>,
            ) -> Self {
                Self {
                    max_depth,
                    upperbound,
                    terminal_solver,
                    strategy,
                    intermediates,
                    timeout: timeout.map(Duration::from_secs),
                    node_lowerbound,
                    memory_limit,
                    task: <$task>::new(complexity_cost),
                    result: None,
                    _phantom: PhantomData,
                }
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
                    ub_strategy: match self.upperbound {
                        UpperboundStrategyEnum::SolutionsOnly => UpperboundStrategy::SolutionsOnly,
                        UpperboundStrategyEnum::TightFromSibling => UpperboundStrategy::TightFromSibling,
                        UpperboundStrategyEnum::ForRemainingInterval => UpperboundStrategy::ForRemainingInterval,
                    },
                    terminal_solver: match self.terminal_solver {
                        TerminalSolverEnum::Leaf => TerminalSolver::Leaf,
                        TerminalSolverEnum::LeftRight => TerminalSolver::LeftRight,
                        TerminalSolverEnum::D2 => TerminalSolver::D2,
                    },
                    node_lowerbound: self.node_lowerbound,
                    timeout: self.timeout,
                    track_intermediates: self.intermediates,
                    memory_limit: self.memory_limit,
                };

                let result = match self.strategy {
                    SearchStrategyEnum::Dfs => {
                        let mut solver: Solver<'_, $task, DfsSearchStrategy> =
                            Solver::new(self.task.clone(), full_view);
                        solver.solve(options)
                    }
                    SearchStrategyEnum::AndOr => {
                        let mut solver: Solver<'_, $task, AndOrSearchStrategy> =
                            Solver::new(self.task.clone(), full_view);
                        solver.solve(options)
                    }
                    SearchStrategyEnum::BfsLb => {
                        let mut solver: Solver<'_, $task, BfsSearchStrategy<LBHeuristic>> =
                            Solver::new(self.task.clone(), full_view);
                        solver.solve(options)
                    }
                    SearchStrategyEnum::BfsCuriosity => {
                        let mut solver: Solver<'_, $task, BfsSearchStrategy<CuriosityHeuristic>> =
                            Solver::new(self.task.clone(), full_view);
                        solver.solve(options)
                    }
                    SearchStrategyEnum::BfsGosdt => {
                        let mut solver: Solver<'_, $task, BfsSearchStrategy<GOSDTHeuristic>> =
                            Solver::new(self.task.clone(), full_view);
                        solver.solve(options)
                    }
                    SearchStrategyEnum::DfsPrio => {
                        let mut solver: Solver<'_, $task, DfsPrioSearchStrategy> =
                            Solver::new(self.task.clone(), full_view);
                        solver.solve(options)
                    }
                };

                self.result = Some(result);
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
