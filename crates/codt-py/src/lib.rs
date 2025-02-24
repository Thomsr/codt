use std::path::PathBuf;

use pyo3::prelude::*;

#[pyfunction]
fn solve_classification(_path: PathBuf, _max_depth: u32, _max_num_nodes: u32) -> PyResult<String> {
    codt::say_hi();
    Ok("hi".to_string())
}

#[pymodule]
fn codt_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_classification, m)?)?;
    Ok(())
}
