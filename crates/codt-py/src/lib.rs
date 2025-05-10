mod tree;

use pyo3::prelude::*;
use tree::{
    OptimalDecisionTreeClassifier, OptimalDecisionTreeRegressor, SearchStrategyEnum,
    all_search_strategies, search_strategy_from_string,
};

#[pymodule]
fn codt_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchStrategyEnum>()?;
    m.add_class::<OptimalDecisionTreeClassifier>()?;
    m.add_class::<OptimalDecisionTreeRegressor>()?;

    m.add_function(wrap_pyfunction!(all_search_strategies, m)?)?;
    m.add_function(wrap_pyfunction!(search_strategy_from_string, m)?)?;

    Ok(())
}
