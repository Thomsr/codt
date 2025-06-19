mod tree;

use codt::search::solver::{SearchStrategyEnum, TerminalSolver, UpperboundStrategy};
use pyo3::prelude::*;
use strum::VariantNames;
use tree::{OptimalDecisionTreeClassifier, OptimalDecisionTreeRegressor};

#[pyfunction(signature = ())]
pub fn all_search_strategies() -> &'static [&'static str] {
    SearchStrategyEnum::VARIANTS
}

#[pyfunction(signature = ())]
pub fn all_upperbounds() -> &'static [&'static str] {
    UpperboundStrategy::VARIANTS
}

#[pyfunction(signature = ())]
pub fn all_terminal_solvers() -> &'static [&'static str] {
    TerminalSolver::VARIANTS
}

#[pymodule]
fn codt_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OptimalDecisionTreeClassifier>()?;
    m.add_class::<OptimalDecisionTreeRegressor>()?;

    m.add_function(wrap_pyfunction!(all_search_strategies, m)?)?;
    m.add_function(wrap_pyfunction!(all_upperbounds, m)?)?;
    m.add_function(wrap_pyfunction!(all_terminal_solvers, m)?)?;

    Ok(())
}
