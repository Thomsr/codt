mod tree;

use codt::search::solver::{LowerBoundStrategy, SearchStrategyEnum, UpperboundStrategy};
use pyo3::prelude::*;
use strum::VariantNames;
use tree::OptimalDecisionTreeClassifier;

#[pyfunction(signature = ())]
pub fn all_search_strategies() -> &'static [&'static str] {
    SearchStrategyEnum::VARIANTS
}

#[pyfunction(signature = ())]
pub fn all_lowerbounds() -> &'static [&'static str] {
    LowerBoundStrategy::VARIANTS
}

#[pyfunction(signature = ())]
pub fn all_upperbounds() -> &'static [&'static str] {
    UpperboundStrategy::VARIANTS
}

#[pymodule]
fn codt_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OptimalDecisionTreeClassifier>()?;

    m.add_function(wrap_pyfunction!(all_search_strategies, m)?)?;
    m.add_function(wrap_pyfunction!(all_lowerbounds, m)?)?;
    m.add_function(wrap_pyfunction!(all_upperbounds, m)?)?;

    Ok(())
}
