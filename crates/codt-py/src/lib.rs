mod tree;

use pyo3::prelude::*;
use tree::{
    OptimalDecisionTreeClassifier, OptimalDecisionTreeRegressor, SearchStrategyEnum,
    search_strategy_from_string, terminal_solver_from_string, ub_from_string,
};

#[pymodule]
fn codt_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchStrategyEnum>()?;
    m.add_class::<OptimalDecisionTreeClassifier>()?;
    m.add_class::<OptimalDecisionTreeRegressor>()?;

    m.add_function(wrap_pyfunction!(search_strategy_from_string, m)?)?;
    m.add_function(wrap_pyfunction!(ub_from_string, m)?)?;
    m.add_function(wrap_pyfunction!(terminal_solver_from_string, m)?)?;

    Ok(())
}
