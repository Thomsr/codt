/// Cart feature-subset diminishing-returns experiment.
///
/// Run with (from the crate root):
///
///   cargo test cart_subset_experiment -- --nocapture \
///       --test-args OUTPUT=/path/to/out.csv \
///                   MAX_ITER=40 \
///                   SEED=42
///
/// All env-style key=value pairs after `--test-args` are read from
/// std::env, so you can also just export them before running:
///
///   export OUTPUT=...  MAX_ITER=...  SEED=...
///   cargo test cart_subset_experiment -- --nocapture
///
/// Datasets are loaded from data/openml/sampled/
#[cfg(test)]
mod cart_subset_experiment {
    use std::{fs, io::Write, path::PathBuf};

    use codt::{
        model::{dataset::DataSet, dataview::DataView, instance::LabeledInstance},
        search,
        tasks::accuracy::AccuracyTask,
        test_support::{read_from_file, repo_root},
    };

    fn cart_size(dataset: &DataSet<LabeledInstance<i32>>, use_subset: bool, seed: u64) -> usize {
        let view = DataView::<AccuracyTask>::from_dataset(dataset, false);
        let task = AccuracyTask::new();
        let tree = search::upper_bounds::cart::cart_upper_bound_with_subset_seed(
            &task,
            &view,
            Some(use_subset),
            seed,
        );
        tree.branch_count()
    }

    // ------------------------------------------------------------------ //
    //  Experiment
    // ------------------------------------------------------------------ //

    #[test]
    fn cart_subset_experiment() {
        fn env(key: &str, default: &str) -> String {
            std::env::var(key).unwrap_or_else(|_| default.to_string())
        }

        let data_dir = repo_root().join("data/openml/sampled");

        let output_path = std::env::var("OUTPUT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("cart_subset_experiment.csv"));

        let max_iter: usize = env("MAX_ITER", "40")
            .parse()
            .expect("MAX_ITER must be an integer");
        let seed: u64 = env("SEED", "42").parse().expect("SEED must be an integer");

        println!("DATA_DIR        = {}", data_dir.display());
        println!("OUTPUT          = {}", output_path.display());
        println!("MAX_ITER        = {max_iter}");
        println!("SEED            = {seed}");

        // Collect all dataset files from the directory
        let mut dataset_files: Vec<PathBuf> = fs::read_dir(&data_dir)
            .unwrap_or_else(|e| panic!("Cannot read data directory {}: {}", data_dir.display(), e))
            .filter_map(|entry| {
                entry.ok().and_then(|e| {
                    let path = e.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("txt") {
                        Some(path)
                    } else {
                        None
                    }
                })
            })
            .collect();

        dataset_files.sort();
        println!("Found {} datasets", dataset_files.len());
        assert!(
            !dataset_files.is_empty(),
            "No datasets found in {}",
            data_dir.display()
        );

        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).unwrap_or_else(|e| {
                panic!("Cannot create output directory {}: {}", parent.display(), e)
            });
        }

        let mut out = fs::File::create(&output_path)
            .unwrap_or_else(|e| panic!("Cannot create output {}: {}", output_path.display(), e));

        writeln!(
            out,
            "dataset,num_dims,iteration,use_subset,cart_size,best_so_far"
        )
        .unwrap();

        for dataset_path in &dataset_files {
            let mut dataset: DataSet<LabeledInstance<i32>> = DataSet::default();
            if let Err(e) = read_from_file(&mut dataset, dataset_path) {
                eprintln!("SKIP {}: {}", dataset_path.display(), e);
                continue;
            }

            // Get number of features from dataset
            let num_dims = dataset.original_feature_values.len();

            let filename = dataset_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");

            println!("Processing {} | dims={}", filename, num_dims);

            let full_size = cart_size(&dataset, false, seed);
            let mut best_so_far = full_size;

            writeln!(
                out,
                "{},{},{},{},{},{}",
                filename, num_dims, 0, false, full_size, best_so_far,
            )
            .unwrap();

            for iter in 1..=max_iter {
                let iter_seed = seed.wrapping_add(iter as u64);
                let size = cart_size(&dataset, true, iter_seed);

                if size < best_so_far {
                    best_so_far = size;
                }

                writeln!(
                    out,
                    "{},{},{},{},{},{}",
                    filename, num_dims, iter, true, size, best_so_far,
                )
                .unwrap();
            }
        }

        println!("Results written to {}", output_path.display());
    }
}
