use std::fmt::Debug;
use std::str::FromStr;

pub trait Instance {
    /// Read this instance from a string, return the instance and the remainder of the string with the feature values.
    fn read(line: String) -> (Self, String)
    where
        Self: Sized;
}

fn read_label<T: FromStr>(line: String) -> (T, String)
where
    <T as FromStr>::Err: Debug,
{
    let mut parts = line.split(' ');
    let label = parts
        .next()
        .expect("Expected at least a label at this line");
    let label: T = label
        .parse()
        .expect("Expected a label of the correct type at the start of the line");

    (label, parts.collect::<Vec<_>>().join(" "))
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
    fn read(line: String) -> (Self, String) {
        let (label, remainder) = read_label(line);
        (Self { label }, remainder)
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
    fn read(line: String) -> (Self, String) {
        let (label, remainder) = read_label(line);
        (Self { label }, remainder)
    }
}
