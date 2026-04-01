use std::{fs, path::PathBuf, time::Duration};

#[path = "dataset-by-difficulty.rs"]
mod dataset_by_difficulty;

use codt::{
    model::{dataset::DataSet, dataview::DataView},
    search::solver::{
        SearchStrategyEnum, SolveStatus, SolverOptions, UpperboundStrategy, solver_with_strategy,
    },
    tasks::accuracy::AccuracyTask,
};

use dataset_by_difficulty::DATASETS_BY_DIFFICULTY;

#[derive(Debug)]
struct WittyRecord {
    optimal: bool,
    tree_size: Option<usize>,
}

fn default_options() -> SolverOptions {
    SolverOptions {
        ub_strategy: UpperboundStrategy::ForRemainingInterval,
        track_intermediates: false,
        timeout: Some(Duration::from_secs(60)),
        memory_limit: None,
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../")
}

fn witty_record_for_dataset(name: &str) -> WittyRecord {
    let target = format!("normal/sampled/{name}");
    let path = repo_root().join("results/witty-results");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read Witty results {}: {}", path.display(), e));

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(';').collect();
        if parts.len() < 15 {
            continue;
        }

        if parts[2] != target {
            continue;
        }

        let optimal = parts[12] == "true";
        let tree_size_raw = parts[13]
            .parse::<isize>()
            .unwrap_or_else(|e| panic!("Invalid tree size for {}: {}", target, e));
        let tree_size = if tree_size_raw >= 0 {
            Some(tree_size_raw as usize)
        } else {
            None
        };

        return WittyRecord { optimal, tree_size };
    }

    panic!("Dataset {} not found in {}", target, path.display());
}

#[test]
fn codt_matches_witty_minimum_tree_size_on_sampled_datasets() {
    let datasets = &DATASETS_BY_DIFFICULTY[..40];

    for dataset_name in datasets {
        let witty = witty_record_for_dataset(dataset_name);
        assert!(
            witty.optimal,
            "Expected Witty to have solved {} to optimality",
            dataset_name
        );
        let expected_size = witty
            .tree_size
            .expect("Expected tree size for an optimal Witty solution");

        let dataset =
            DataSet::from_csv(&repo_root().join("data/normal/sampled").join(dataset_name));
        let full_view = DataView::from_dataset(&dataset);
        let mut solver = solver_with_strategy(
            AccuracyTask::new(),
            full_view,
            SearchStrategyEnum::BfsBalanceSmallLb,
        );

        let result = solver.solve(default_options());
        println!("dataset: {} status: {:?}", dataset_name, result.status);

        assert_eq!(
            result.status,
            SolveStatus::PerfectTreeFound,
            "CODT should find a perfect tree for {}",
            dataset_name
        );

        let tree = result
            .tree
            .expect("Tree should exist for perfect solution status");
        let size = tree.branch_count();
        println!(
            "dataset: {} witty_size: {} codt_size: {} tree: {}",
            dataset_name, expected_size, size, tree
        );

        assert_eq!(
            size, expected_size,
            "CODT tree size should match Witty optimum on {}",
            dataset_name
        );
    }
}
