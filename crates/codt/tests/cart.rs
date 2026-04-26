/// Cart feature-subset diminishing-returns experiment.
///
/// Run with (from the crate root):
///
///   cargo test cart_subset_experiment -- --nocapture \
///       --test-args DATA_DIR=/path/to/data/sampled \
///                   WITTY_RESULTS=/path/to/witty-results \
///                   OUTPUT=/path/to/out.csv \
///                   MAX_ITER=40 \
///                   SEED=42
///
/// All env-style key=value pairs after `--test-args` are read from
/// std::env, so you can also just export them before running:
///
///   export DATA_DIR=...  WITTY_RESULTS=...  OUTPUT=...
///   cargo test cart_subset_experiment -- --nocapture
#[cfg(test)]
mod cart_subset_experiment {
    use std::{
        collections::HashMap,
        fs,
        io::{BufRead, BufReader, Write},
        path::{Path, PathBuf},
    };

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
    //  Witty results parsing
    // ------------------------------------------------------------------ //

    struct WittyRecord {
        filename: String,
        num_dims: usize,
        optimal_size: usize,
    }

    fn parse_witty_results(path: &Path) -> Vec<WittyRecord> {
        let file = fs::File::open(path)
            .unwrap_or_else(|e| panic!("Cannot open witty results {}: {}", path.display(), e));

        BufReader::new(file)
            .lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .filter_map(|line| {
                let fields: Vec<&str> = line.trim().trim_end_matches(';').split(';').collect();
                if fields.len() < 14 {
                    return None;
                }

                let found_optimal = fields[12].trim() == "true";
                if !found_optimal {
                    return None;
                }

                let optimal_size: isize = fields[13].trim().parse().ok()?;
                if optimal_size < 0 {
                    return None;
                }

                let num_dims: usize = fields[6].trim().parse().ok()?;

                let raw_name = fields[2].trim();
                let filename = raw_name.rsplit('/').next().unwrap_or(raw_name).to_string();

                Some(WittyRecord {
                    filename,
                    num_dims,
                    optimal_size: optimal_size as usize,
                })
            })
            .collect()
    }

    // ------------------------------------------------------------------ //
    //  Experiment
    // ------------------------------------------------------------------ //

    #[test]
    fn cart_subset_experiment() {
        fn env(key: &str, default: &str) -> String {
            std::env::var(key).unwrap_or_else(|_| default.to_string())
        }

        let data_dir = PathBuf::from("/home/thoams/Documents/TUDelft/thesis/codt/data/sampled");

        let witty_results_path = repo_root().join("experiments/results/witty-results");

        let output_path = std::env::var("OUTPUT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("cart_subset_experiment.csv"));

        let max_iter: usize = env("MAX_ITER", "40")
            .parse()
            .expect("MAX_ITER must be an integer");
        let seed: u64 = env("SEED", "42").parse().expect("SEED must be an integer");

        println!("DATA_DIR        = {}", data_dir.display());
        println!("WITTY_RESULTS   = {}", witty_results_path.display());
        println!("OUTPUT          = {}", output_path.display());
        println!("MAX_ITER        = {max_iter}");
        println!("SEED            = {seed}");

        let records = parse_witty_results(&witty_results_path);
        println!(
            "Loaded {} solved datasets from witty results",
            records.len()
        );
        assert!(!records.is_empty(), "No solved datasets found");

        let mut seen: HashMap<String, bool> = HashMap::new();
        let records: Vec<_> = records
            .into_iter()
            .filter(|r| seen.insert(r.filename.clone(), true).is_none())
            .collect();

        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).unwrap_or_else(|e| {
                panic!("Cannot create output directory {}: {}", parent.display(), e)
            });
        }

        let mut out = fs::File::create(&output_path)
            .unwrap_or_else(|e| panic!("Cannot create output {}: {}", output_path.display(), e));

        writeln!(
            out,
            "dataset,num_dims,optimal_size,iteration,use_subset,cart_size,best_so_far"
        )
        .unwrap();

        for record in &records {
            let dataset_path = data_dir.join(&record.filename).with_extension("txt");

            let mut dataset = DataSet::default();
            if let Err(e) = read_from_file(&mut dataset, &dataset_path) {
                eprintln!("SKIP {}: {}", record.filename, e);
                continue;
            }

            let num_dims = record.num_dims;

            println!(
                "Processing {} | dims={} | optimal={}",
                record.filename, num_dims, record.optimal_size
            );

            let full_size = cart_size(&dataset, false, seed);
            let mut best_so_far = full_size;

            writeln!(
                out,
                "{},{},{},{},{},{},{}",
                record.filename, num_dims, record.optimal_size, 0, false, full_size, best_so_far,
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
                    "{},{},{},{},{},{},{}",
                    record.filename, num_dims, record.optimal_size, iter, true, size, best_so_far,
                )
                .unwrap();
            }
        }

        println!("Results written to {}", output_path.display());
    }
}
