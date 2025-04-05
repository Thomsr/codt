use std::{
    fs::File,
    io::{self, BufRead},
    path::Path,
};

use codt::model::{dataset::DataSet, instance::Instance};

pub fn read_from_file<I: Instance, P: AsRef<Path>>(
    dataset: &mut DataSet<I>,
    filename: P,
) -> Result<(), std::io::Error> {
    let file = File::open(filename)?;

    let lines = io::BufReader::new(file).lines();
    for l in lines {
        let line = l?;
        let (instance, feature_string) = I::read(line);
        let features = feature_string.split(' ').map(|i| {
            i.parse::<f64>()
                .expect("Expected a real valued feature value while reading instances")
        });
        dataset.add_instance(instance, features);
    }

    if dataset.instances.is_empty() {
        return Err(std::io::Error::other(
            "File did not result in any instances being read.",
        ));
    }

    dataset.preprocess_after_adding_instances();

    Ok(())
}
