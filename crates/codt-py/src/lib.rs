mod tree;

use pyo3::prelude::*;
use tree::{OptimalDecisionTreeClassifier, OptimalDecisionTreeRegressor, SearchStrategyEnum};

#[pymodule]
fn codt_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchStrategyEnum>()?;
    m.add_class::<OptimalDecisionTreeClassifier>()?;
    m.add_class::<OptimalDecisionTreeRegressor>()?;

    Ok(())
}
