use std::{convert::Infallible, fmt::Debug, marker::PhantomData, str::FromStr};

use codt::{
    model::{dataset::DataSet, dataview::DataView, instance::LabeledInstance, tree::Tree},
    search::{
        solver::{SolveResult, Solver},
        strategy::{
            andor::AndOrSearchStrategy,
            bfs::{BfsSearchStrategy, CuriosityHeuristic, GOSDTHeuristic, LBHeuristic},
            dfs::DfsSearchStrategy,
            dfsprio::DfsPrioSearchStrategy,
        },
    },
    tasks::{OptimizationTask, accuracy::AccuracyTask, regression::RegressionTask},
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, ndarray::Axis};
use pyo3::{prelude::*, types::PyList};

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
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum SearchStrategyEnum {
    Dfs,
    AndOr,
    BfsLb,
    BfsCuriosity,
    BfsGosdt,
    DfsPrio,
}

#[pyclass]
pub struct OptimalDecisionTreeClassifier(OptimalDecisionTree<AccuracyTask, i64>);

#[pymethods]
#[allow(non_snake_case)]
impl OptimalDecisionTreeClassifier {
    #[new]
    #[pyo3(signature = (max_depth=2, strategy=SearchStrategyEnum::Dfs))]
    fn new(max_depth: u32, strategy: SearchStrategyEnum) -> OptimalDecisionTreeClassifier {
        OptimalDecisionTreeClassifier(OptimalDecisionTree::new(max_depth, strategy))
    }

    #[pyo3(signature = (X, y))]
    fn fit<'py>(&mut self, X: PyReadonlyArray2<'py, f64>, y: PyReadonlyArray1<'py, i64>) {
        self.0.fit(X, y);
    }

    #[pyo3(signature = (X))]
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        X: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray1<i64>> {
        self.0.predict(py, X)
    }

    #[pyo3(signature = ())]
    fn tree<'py>(&mut self, py: Python<'py>) -> Result<Bound<'py, PyAny>, PyErr> {
        let tree = &self
            .0
            .result
            .as_ref()
            .expect(".fit(X,y) should be called before this function and should be checked in the python wrapper")
            .tree;

        tree_to_py(tree, py)
    }

    #[pyo3(signature = ())]
    fn intermediate_lbs<'py>(&mut self, py: Python<'py>) -> Result<Bound<'py, PyAny>, PyErr> {
        let intermediates = &self
            .0
            .result
            .as_ref()
            .expect(".fit(X,y) should be called before this function and should be checked in the python wrapper")
            .intermediate_lbs;
        intermediates.into_pyobject(py)
    }

    #[pyo3(signature = ())]
    fn intermediate_ubs<'py>(&mut self, py: Python<'py>) -> Result<Bound<'py, PyAny>, PyErr> {
        let intermediates = &self
            .0
            .result
            .as_ref()
            .expect(".fit(X,y) should be called before this function and should be checked in the python wrapper")
            .intermediate_ubs;
        intermediates.into_pyobject(py)
    }
}

#[pyclass]
pub struct OptimalDecisionTreeRegressor(OptimalDecisionTree<RegressionTask, f64>);

#[pymethods]
#[allow(non_snake_case)]
impl OptimalDecisionTreeRegressor {
    #[new]
    #[pyo3(signature = (max_depth=2, strategy=SearchStrategyEnum::Dfs))]
    fn new(max_depth: u32, strategy: SearchStrategyEnum) -> OptimalDecisionTreeRegressor {
        OptimalDecisionTreeRegressor(OptimalDecisionTree::new(max_depth, strategy))
    }

    #[pyo3(signature = (X, y))]
    fn fit<'py>(&mut self, X: PyReadonlyArray2<'py, f64>, y: PyReadonlyArray1<'py, f64>) {
        self.0.fit(X, y);
    }

    #[pyo3(signature = (X))]
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        X: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        self.0.predict(py, X)
    }

    #[pyo3(signature = ())]
    fn tree<'py>(&mut self, py: Python<'py>) -> Result<Bound<'py, PyAny>, PyErr> {
        let tree = &self
            .0
            .result
            .as_ref()
            .expect(".fit(X,y) should be called before this function and should be checked in the python wrapper")
            .tree;

        tree_to_py(tree, py)
    }

    #[pyo3(signature = ())]
    fn intermediate_lbs<'py>(&mut self, py: Python<'py>) -> Result<Bound<'py, PyAny>, PyErr> {
        let intermediates = &self
            .0
            .result
            .as_ref()
            .expect(".fit(X,y) should be called before this function and should be checked in the python wrapper")
            .intermediate_lbs;
        intermediates.into_pyobject(py)
    }

    #[pyo3(signature = ())]
    fn intermediate_ubs<'py>(&mut self, py: Python<'py>) -> Result<Bound<'py, PyAny>, PyErr> {
        let intermediates = &self
            .0
            .result
            .as_ref()
            .expect(".fit(X,y) should be called before this function and should be checked in the python wrapper")
            .intermediate_ubs;
        intermediates.into_pyobject(py)
    }
}

pub struct OptimalDecisionTree<
    OT: OptimizationTask,
    PyLabelType: TryFrom<OT::LabelType, Error: Debug>
        + TryInto<OT::LabelType, Error: Debug>
        + Copy
        + numpy::Element,
> {
    max_depth: u32,
    strategy: SearchStrategyEnum,
    result: Option<SolveResult<OT>>,
    _phantom: PhantomData<PyLabelType>,
}

#[allow(non_snake_case)]
impl<
    LabelType: FromStr + Clone + Copy + Into<PyLabelType> + TryFrom<PyLabelType>,
    OT: OptimizationTask<InstanceType = LabeledInstance<LabelType>, LabelType = LabelType> + Default,
    PyLabelType: Copy + numpy::Element,
> OptimalDecisionTree<OT, PyLabelType>
where
    <LabelType as FromStr>::Err: Debug,
    <LabelType as TryFrom<PyLabelType>>::Error: Debug,
{
    fn new(max_depth: u32, strategy: SearchStrategyEnum) -> Self {
        Self {
            max_depth,
            strategy,
            result: None,
            _phantom: PhantomData,
        }
    }

    fn fit<'py>(&mut self, X: PyReadonlyArray2<'py, f64>, y: PyReadonlyArray1<'py, PyLabelType>) {
        let x_arr = X.as_array();
        let y_arr = y.as_array();
        let mut dataset = DataSet::<LabeledInstance<LabelType>>::default();
        for i in 0..y_arr.dim() {
            let features = x_arr.index_axis(Axis(0), i);
            dataset.add_instance(
                LabeledInstance::new(y_arr[i].try_into().unwrap()),
                features.iter().copied(),
            );
        }
        dataset.preprocess_after_adding_instances();

        OT::preprocess_dataset(&mut dataset);
        let task = OT::default();
        let full_view = DataView::from_dataset(&dataset);

        let result = match self.strategy {
            SearchStrategyEnum::Dfs => {
                let mut solver: Solver<'_, OT, DfsSearchStrategy> = Solver::new(task, full_view);
                solver.solve(self.max_depth)
            }
            SearchStrategyEnum::AndOr => {
                let mut solver: Solver<'_, OT, AndOrSearchStrategy> = Solver::new(task, full_view);
                solver.solve(self.max_depth)
            }
            SearchStrategyEnum::BfsLb => {
                let mut solver: Solver<'_, OT, BfsSearchStrategy<LBHeuristic>> =
                    Solver::new(task, full_view);
                solver.solve(self.max_depth)
            }
            SearchStrategyEnum::BfsCuriosity => {
                let mut solver: Solver<'_, OT, BfsSearchStrategy<CuriosityHeuristic>> =
                    Solver::new(task, full_view);
                solver.solve(self.max_depth)
            }
            SearchStrategyEnum::BfsGosdt => {
                let mut solver: Solver<'_, OT, BfsSearchStrategy<GOSDTHeuristic>> =
                    Solver::new(task, full_view);
                solver.solve(self.max_depth)
            }
            SearchStrategyEnum::DfsPrio => {
                let mut solver: Solver<'_, OT, DfsPrioSearchStrategy> =
                    Solver::new(task, full_view);
                solver.solve(self.max_depth)
            }
        };

        self.result = Some(result);
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray1<PyLabelType>> {
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
}
