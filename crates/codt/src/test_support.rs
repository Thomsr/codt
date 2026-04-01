use std::{
    fs::File,
    io::{self, BufRead},
    path::{Path, PathBuf},
};

use crate::model::{dataset::DataSet, instance::Instance};

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../")
}

pub fn read_from_file<I: Instance, P: AsRef<Path>>(
    dataset: &mut DataSet<I>,
    filename: P,
) -> Result<(), io::Error> {
    let file = File::open(filename)?;

    let lines = io::BufReader::new(file).lines();
    for line in lines {
        let line = line?;
        let (instance, feature_string) = I::read(line);
        let features = feature_string.split(' ').map(|value| {
            value
                .parse::<f64>()
                .expect("Expected a real valued feature value while reading instances")
        });
        dataset.add_instance(instance, features);
    }

    if dataset.instances.is_empty() {
        return Err(io::Error::other(
            "File did not result in any instances being read.",
        ));
    }

    dataset.preprocess_after_adding_instances();

    Ok(())
}
