mod tree;

use pyo3::prelude::*;
use tree::OptimalDecisionTreeClassifier;

#[pymodule]
fn codt_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OptimalDecisionTreeClassifier>()?;

    Ok(())
}
