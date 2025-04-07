use codt::model::{dataset::DataSet, instance::LabeledInstance};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, ndarray::Axis};
use pyo3::prelude::*;

#[pyclass]
pub struct OptimalDecisionTreeClassifier {
    #[allow(dead_code)]
    max_depth: i32,
    #[allow(dead_code)]
    max_nodes: i32,
    dataset: Option<DataSet<LabeledInstance<i32>>>,
}

#[pymethods]
#[allow(non_snake_case)]
impl OptimalDecisionTreeClassifier {
    #[new]
    #[pyo3(signature = (max_depth=4, max_nodes=31))]
    fn new(max_depth: i32, max_nodes: i32) -> OptimalDecisionTreeClassifier {
        OptimalDecisionTreeClassifier {
            max_depth,
            max_nodes,
            dataset: None,
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

        self.dataset = Some(dataset);
    }

    #[pyo3(signature = (X))]
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        X: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray1<i64>> {
        let ret = self.dataset.as_ref().unwrap().original_feature_values[0][0] as i64;
        X.as_array().map_axis(Axis(1), |_| ret).into_pyarray(py)
    }
}
