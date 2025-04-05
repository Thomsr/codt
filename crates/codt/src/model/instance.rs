use std::fmt::Debug;
use std::str::FromStr;

pub trait Instance {
    fn read(line: String) -> Self;
}

fn read_label<T: FromStr>(line: String) -> T
where
    <T as FromStr>::Err: Debug,
{
    let parts: Vec<&str> = line.split(' ').collect();
    let label = parts
        .first()
        .expect("Expected at least a label at this line");
    let label: T = label
        .parse()
        .expect("Expected a label of the correct type at the start of the line");

    label
}

// Classification

pub struct ClassificationInstance {
    pub label: i32,
}

impl ClassificationInstance {
    pub fn new(label: i32) -> Self {
        Self { label }
    }
}

impl Instance for ClassificationInstance {
    fn read(line: String) -> Self {
        Self {
            label: read_label(line),
        }
    }
}

// Regression

pub struct RegressionInstance {
    pub label: f64,
}

impl RegressionInstance {
    pub fn new(label: f64) -> Self {
        Self { label }
    }
}

impl Instance for RegressionInstance {
    fn read(line: String) -> Self {
        Self {
            label: read_label(line),
        }
    }
}
