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

/// An instance with a constant label. For example, used in classifcation/regression.
pub struct LabeledInstance<T: FromStr>
where
    <T as FromStr>::Err: Debug,
{
    pub label: T,
}

impl<T: FromStr> LabeledInstance<T>
where
    <T as FromStr>::Err: Debug,
{
    pub fn new(label: T) -> Self {
        Self { label }
    }
}

impl<T: FromStr> Instance for LabeledInstance<T>
where
    <T as FromStr>::Err: Debug,
{
    fn read(line: String) -> (Self, String) {
        let (label, remainder) = read_label(line);
        (Self { label }, remainder)
    }
}
