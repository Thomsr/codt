use numpy::{ndarray::Axis, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyclass]
pub struct OptimalDecisionTreeClassifier {
    max_depth: i32,
    max_nodes: i32,
}

#[pymethods]
impl OptimalDecisionTreeClassifier {
    #[new]
    #[pyo3(signature = (max_depth=4, max_nodes=31))]
    fn new(max_depth: i32, max_nodes: i32) -> OptimalDecisionTreeClassifier {
        OptimalDecisionTreeClassifier {
            max_depth,
            max_nodes,
        }
    }

    #[pyo3(signature = (X, y))]
    fn fit<'py>(&mut self, X: PyReadonlyArray2<'py, f64>, y: PyReadonlyArray1<'py, i64>) {}

    #[pyo3(signature = (X))]
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        X: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray1<i64>> {
        X.as_array().map_axis(Axis(1), |_| 5).into_pyarray(py)
    }
}
