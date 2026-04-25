use std::collections::HashMap;

use crate::tasks::OptimizationTask;

use super::dataset::DataSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataReductionStrategy {
    Disabled,
    Enabled,
}

#[derive(Debug, Clone, Default)]
pub struct ReductionStats {
    pub before_examples: usize,
    pub after_examples: usize,
    pub before_dimensions: usize,
    pub after_dimensions: usize,
    pub before_cuts: usize,
    pub after_cuts: usize,
    pub removed_duplicate_examples: usize,
    pub removed_dimensions: usize,
    pub removed_cuts_equivalent: usize,
    pub removed_cuts_dimension_reduction: usize,
    pub merged_dimensions: usize,
}

#[derive(Clone)]
pub struct ReducedData {
    pub instance_ids: Vec<usize>,
    pub feature_values: Vec<Vec<i32>>, // [feature][instance_idx]
    pub cut_sources: Vec<Vec<(usize, i32)>>, // [feature][cut_idx] -> (original_feature, original_internal_cut_value)
    pub stats: ReductionStats,
}

struct WorkingData {
    instance_ids: Vec<usize>,
    feature_values: Vec<Vec<i32>>,
    cut_sources: Vec<Vec<(usize, i32)>>,
}

impl WorkingData {
    fn num_cuts(&self) -> usize {
        self.cut_sources.iter().map(Vec::len).sum()
    }

    fn remove_cut(&mut self, feature: usize, cut_idx: usize) {
        for v in &mut self.feature_values[feature] {
            if *v > cut_idx as i32 {
                *v -= 1;
            }
        }
        self.cut_sources[feature].remove(cut_idx);
    }

    fn remove_dimension(&mut self, feature: usize) {
        self.feature_values.remove(feature);
        self.cut_sources.remove(feature);
    }

    fn remove_instances(&mut self, keep: &[bool]) {
        self.instance_ids = self
            .instance_ids
            .iter()
            .zip(keep.iter())
            .filter_map(|(&id, &k)| k.then_some(id))
            .collect();

        for values in &mut self.feature_values {
            *values = values
                .iter()
                .zip(keep.iter())
                .filter_map(|(&v, &k)| k.then_some(v))
                .collect();
        }
    }
}

fn recompute_cut_sources_from_values(
    values: &[i32],
    original_feature_fallback: usize,
    prior_sources: &[(usize, i32)],
) -> Vec<(usize, i32)> {
    let mut unique = values.to_vec();
    unique.sort_unstable();
    unique.dedup();

    if unique.len() <= 1 {
        return Vec::new();
    }

    (0..(unique.len() - 1))
        .map(|k| {
            let src = prior_sources
                .get(k)
                .copied()
                .unwrap_or((original_feature_fallback, unique[k]));
            (src.0, src.1)
        })
        .collect()
}

fn class_of<OT: OptimizationTask>(
    dataset: &DataSet<OT::InstanceType>,
    instance_id: usize,
) -> OT::LabelType {
    OT::label_of_instance(&dataset.instances[instance_id])
}

fn apply_dimension_reduction<OT: OptimizationTask>(
    w: &mut WorkingData,
    dataset: &DataSet<OT::InstanceType>,
) -> usize
where
    OT::LabelType: PartialEq,
{
    let mut removed = 0usize;

    for feature in 0..w.feature_values.len() {
        loop {
            let values = &w.feature_values[feature];
            let max_value = values.iter().copied().max().unwrap_or(0);
            if max_value <= 0 {
                break;
            }

            let n_cuts = max_value as usize;
            let mut left_pure = vec![false; n_cuts];
            let mut right_pure = vec![false; n_cuts];

            for cut in 0..n_cuts {
                let mut left_label: Option<OT::LabelType> = None;
                let mut right_label: Option<OT::LabelType> = None;
                let mut left_ok = true;
                let mut right_ok = true;

                for (idx, &v) in values.iter().enumerate() {
                    let label = class_of::<OT>(dataset, w.instance_ids[idx]);
                    if v <= cut as i32 {
                        if let Some(lbl) = left_label {
                            if lbl != label {
                                left_ok = false;
                            }
                        } else {
                            left_label = Some(label);
                        }
                    } else if let Some(lbl) = right_label {
                        if lbl != label {
                            right_ok = false;
                        }
                    } else {
                        right_label = Some(label);
                    }
                    if !left_ok && !right_ok {
                        break;
                    }
                }

                left_pure[cut] = left_ok;
                right_pure[cut] = right_ok;
            }

            let left_indices: Vec<usize> = (0..n_cuts).filter(|&i| left_pure[i]).collect();
            let right_indices: Vec<usize> = (0..n_cuts).filter(|&i| right_pure[i]).collect();

            let mut to_remove = Vec::new();
            if left_indices.len() > 1 {
                let keep = *left_indices.last().expect("len > 1");
                for idx in left_indices {
                    if idx != keep {
                        to_remove.push(idx);
                    }
                }
            }
            if right_indices.len() > 1 {
                let keep = right_indices[0];
                for idx in right_indices {
                    if idx != keep {
                        to_remove.push(idx);
                    }
                }
            }

            to_remove.sort_unstable();
            to_remove.dedup();
            if to_remove.is_empty() {
                break;
            }

            for &cut in to_remove.iter().rev() {
                w.remove_cut(feature, cut);
                removed += 1;
            }
        }
    }

    removed
}

fn left_signature(values: &[i32], cut_idx: usize) -> Vec<bool> {
    values.iter().map(|&v| v <= cut_idx as i32).collect()
}

fn apply_equivalent_cuts(w: &mut WorkingData) -> usize {
    let mut removed = 0usize;

    loop {
        let mut seen: HashMap<Vec<bool>, (usize, usize)> = HashMap::new();
        let mut duplicate: Option<(usize, usize)> = None;

        'outer: for feature in 0..w.feature_values.len() {
            let max_value = w.feature_values[feature].iter().copied().max().unwrap_or(0);
            for cut in 0..(max_value as usize) {
                let sig = left_signature(&w.feature_values[feature], cut);
                if let Some(&first) = seen.get(&sig) {
                    duplicate = Some(first);
                    break 'outer;
                }
                seen.insert(sig, (feature, cut));
            }
        }

        if let Some((f, c)) = duplicate {
            w.remove_cut(f, c);
            removed += 1;
        } else {
            break;
        }
    }

    removed
}

fn build_left_signatures(values: &[i32]) -> Vec<Vec<bool>> {
    let max_value = values.iter().copied().max().unwrap_or(0);
    (0..(max_value as usize))
        .map(|cut| left_signature(values, cut))
        .collect()
}

fn try_merge_dimensions(
    w: &mut WorkingData,
    i1: usize,
    i2: usize,
) -> Option<(Vec<i32>, Vec<(usize, i32)>)> {
    let x = &w.feature_values[i1];
    let y = &w.feature_values[i2];
    let n = x.len();
    if n == 0 {
        return None;
    }

    let mut idxs: Vec<usize> = (0..n).collect();
    idxs.sort_by_key(|&idx| (x[idx], y[idx], idx));

    // Check existence of an ordering where both dimensions are non-decreasing.
    let mut max_y_seen = i32::MIN;
    let mut start = 0usize;
    while start < n {
        let current_x = x[idxs[start]];
        let mut end = start;
        let mut group_min_y = i32::MAX;
        let mut group_max_y = i32::MIN;
        while end < n && x[idxs[end]] == current_x {
            group_min_y = group_min_y.min(y[idxs[end]]);
            group_max_y = group_max_y.max(y[idxs[end]]);
            end += 1;
        }
        if group_min_y < max_y_seen {
            return None;
        }
        max_y_seen = max_y_seen.max(group_max_y);
        start = end;
    }

    let mut merged_values = vec![0i32; n];
    let mut cur = 0i32;
    merged_values[idxs[0]] = cur;
    for widx in 1..n {
        let prev = idxs[widx - 1];
        let now = idxs[widx];
        if x[now] > x[prev] || y[now] > y[prev] {
            cur += 1;
        }
        merged_values[now] = cur;
    }

    let merged_left_sigs = build_left_signatures(&merged_values);
    let left1 = build_left_signatures(x);
    let left2 = build_left_signatures(y);

    let mut merged_sources = Vec::new();
    for sig in &merged_left_sigs {
        let mut found = None;
        for (cut, s) in left1.iter().enumerate() {
            if s == sig {
                found = w.cut_sources[i1].get(cut).copied();
                if found.is_some() {
                    break;
                }
            }
        }
        if found.is_none() {
            for (cut, s) in left2.iter().enumerate() {
                if s == sig {
                    found = w.cut_sources[i2].get(cut).copied();
                    if found.is_some() {
                        break;
                    }
                }
            }
        }

        let src = found?;
        merged_sources.push(src);
    }

    Some((merged_values, merged_sources))
}

fn apply_dimension_merge(w: &mut WorkingData) -> usize {
    let mut merges = 0usize;

    loop {
        let mut best: Option<(usize, usize, Vec<i32>, Vec<(usize, i32)>, usize)> = None;

        for i1 in 0..w.feature_values.len() {
            for i2 in (i1 + 1)..w.feature_values.len() {
                let Some((merged_values, merged_sources)) = try_merge_dimensions(w, i1, i2) else {
                    continue;
                };
                let domain_size = merged_values.iter().copied().max().unwrap_or(-1) as usize + 1;

                let should_replace = best.as_ref().is_none_or(|(b1, b2, _, _, best_domain)| {
                    domain_size < *best_domain
                        || (domain_size == *best_domain && (i1, i2) < (*b1, *b2))
                });

                if should_replace {
                    best = Some((i1, i2, merged_values, merged_sources, domain_size));
                }
            }
        }

        let Some((i1, i2, merged_values, mut merged_sources, _)) = best else {
            break;
        };

        // Remove high index first so i1 remains valid.
        w.remove_dimension(i2);
        w.remove_dimension(i1);
        w.feature_values.insert(i1, merged_values);
        merged_sources = recompute_cut_sources_from_values(
            &w.feature_values[i1],
            merged_sources.first().map(|s| s.0).unwrap_or(0),
            &merged_sources,
        );
        w.cut_sources.insert(i1, merged_sources);

        merges += 1;
    }

    merges
}

fn apply_remove_duplicate_examples<OT: OptimizationTask>(
    w: &mut WorkingData,
    dataset: &DataSet<OT::InstanceType>,
) -> usize
where
    OT::LabelType: PartialEq,
{
    if w.instance_ids.is_empty() {
        return 0;
    }

    let mut first_seen: HashMap<Vec<i32>, usize> = HashMap::new();
    let mut keep = vec![true; w.instance_ids.len()];
    let mut removed = 0usize;

    for idx in 0..w.instance_ids.len() {
        let key: Vec<i32> = w.feature_values.iter().map(|col| col[idx]).collect();
        if let Some(&first_idx) = first_seen.get(&key) {
            let lbl_first = class_of::<OT>(dataset, w.instance_ids[first_idx]);
            let lbl_now = class_of::<OT>(dataset, w.instance_ids[idx]);
            if lbl_first == lbl_now {
                keep[idx] = false;
                removed += 1;
            }
        } else {
            first_seen.insert(key, idx);
        }
    }

    if removed > 0 {
        w.remove_instances(&keep);
    }

    removed
}

fn apply_remove_dimension(w: &mut WorkingData) -> usize {
    let mut removed = 0usize;
    let mut idx = 0usize;

    while idx < w.feature_values.len() {
        let values = &w.feature_values[idx];
        let min = values.iter().copied().min().unwrap_or(0);
        let max = values.iter().copied().max().unwrap_or(0);
        if min == max || w.cut_sources[idx].is_empty() {
            w.remove_dimension(idx);
            removed += 1;
        } else {
            idx += 1;
        }
    }

    removed
}

pub fn reduce_dataset<OT: OptimizationTask>(
    dataset: &DataSet<OT::InstanceType>,
    strategy: DataReductionStrategy,
) -> ReducedData
where
    OT::LabelType: PartialEq,
{
    let mut stats = ReductionStats {
        before_examples: dataset.instances.len(),
        after_examples: dataset.instances.len(),
        before_dimensions: dataset.feature_values.len(),
        after_dimensions: dataset.feature_values.len(),
        before_cuts: dataset
            .internal_to_original_feature_value
            .iter()
            .map(|v| v.len().saturating_sub(1))
            .sum(),
        after_cuts: dataset
            .internal_to_original_feature_value
            .iter()
            .map(|v| v.len().saturating_sub(1))
            .sum(),
        ..ReductionStats::default()
    };

    let mut w = WorkingData {
        instance_ids: (0..dataset.instances.len()).collect(),
        feature_values: dataset.feature_values.clone(),
        cut_sources: dataset
            .internal_to_original_feature_value
            .iter()
            .enumerate()
            .map(|(f, vals)| {
                vals.iter()
                    .take(vals.len().saturating_sub(1))
                    .enumerate()
                    .map(|(i, _)| (f, i as i32))
                    .collect::<Vec<_>>()
            })
            .collect(),
    };

    if matches!(strategy, DataReductionStrategy::Enabled) {
        stats.removed_cuts_dimension_reduction += apply_dimension_reduction::<OT>(&mut w, dataset);
        stats.removed_cuts_equivalent += apply_equivalent_cuts(&mut w);
        stats.merged_dimensions += apply_dimension_merge(&mut w);
        stats.removed_duplicate_examples += apply_remove_duplicate_examples::<OT>(&mut w, dataset);
        stats.removed_dimensions += apply_remove_dimension(&mut w);
    }

    stats.after_examples = w.instance_ids.len();
    stats.after_dimensions = w.feature_values.len();
    stats.after_cuts = w.num_cuts();

    ReducedData {
        instance_ids: w.instance_ids,
        feature_values: w.feature_values,
        cut_sources: w.cut_sources,
        stats,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::{
        model::{dataset::DataSet, instance::LabeledInstance},
        tasks::accuracy::AccuracyTask,
        test_support::{read_from_file, repo_root},
    };

    use super::{DataReductionStrategy, reduce_dataset};

    fn dataset(features: Vec<Vec<i32>>, labels: Vec<i32>) -> DataSet<LabeledInstance<i32>> {
        let mut ds = DataSet::default();
        for l in labels {
            ds.instances.push(LabeledInstance::new(l));
        }
        ds.feature_values = features;
        ds.internal_to_original_feature_value = ds
            .feature_values
            .iter()
            .map(|col| {
                let mut u = col.clone();
                u.sort_unstable();
                u.dedup();
                u.into_iter().map(|v| v as f64).collect::<Vec<_>>()
            })
            .collect();
        ds
    }

    #[test]
    fn remove_duplicate_examples() {
        // Two identical examples with the same label should be reduced to one.
        let ds = dataset(vec![vec![0, 0, 1], vec![0, 0, 1]], vec![1, 1, 0]);
        let out = reduce_dataset::<AccuracyTask>(&ds, DataReductionStrategy::Enabled);
        assert_eq!(out.instance_ids.len(), 2);
    }

    #[test]
    fn remove_dimension_rule() {
        // Feature 1 has the same value for all examples, so it should be removed.
        let ds = dataset(vec![vec![0, 0, 0], vec![0, 1, 1]], vec![0, 1, 1]);
        let out = reduce_dataset::<AccuracyTask>(&ds, DataReductionStrategy::Enabled);
        assert_eq!(out.feature_values.len(), 1);
    }
    #[test]
    fn equivalent_cuts_rule_witty_example() {
        // B.2 Table 1 example from Staus et al.
        let ds = dataset(
            vec![vec![0, 1, 2, 3], vec![1, 0, 2, 2], vec![0, 0, 2, 1]],
            vec![0, 0, 1, 0],
        );
        let out = reduce_dataset::<AccuracyTask>(&ds, DataReductionStrategy::Enabled);
        assert_eq!(out.feature_values.len(), 2);
    }

    #[test]
    fn equivalent_cuts_rule_reduces() {
        let ds = dataset(vec![vec![0, 0, 1, 1], vec![0, 0, 1, 1]], vec![0, 1, 0, 1]);
        let out = reduce_dataset::<AccuracyTask>(&ds, DataReductionStrategy::Enabled);
        assert!(out.stats.after_cuts < out.stats.before_cuts);
    }

    #[test]
    fn dimension_reduction_rule_witty_example() {
        let ds = dataset(
            vec![vec![0, 1, 2, 3], vec![1, 0, 2, 2], vec![0, 0, 2, 1]],
            vec![0, 0, 1, 0],
        );
        let out = reduce_dataset::<AccuracyTask>(&ds, DataReductionStrategy::Enabled);
        assert_eq!(out.feature_values.len(), 2);
    }

    #[test]
    fn dimension_reduction_rule_reduces_extremes() {
        let ds = dataset(vec![vec![0, 1, 2, 3], vec![0, 0, 0, 0]], vec![1, 1, 1, 0]);
        let out = reduce_dataset::<AccuracyTask>(&ds, DataReductionStrategy::Enabled);
        assert!(out.stats.after_cuts < out.stats.before_cuts);
    }

    #[test]
    fn dimension_merge_rule_can_merge() {
        let ds = dataset(vec![vec![0, 1, 2, 3], vec![0, 1, 2, 3]], vec![0, 0, 1, 1]);
        let out = reduce_dataset::<AccuracyTask>(&ds, DataReductionStrategy::Enabled);
        assert!(out.stats.merged_dimensions >= 1);
    }

    #[derive(Debug)]
    struct ReductionCsvRecord {
        instance_name: String,
        n: usize,
        n_prime: usize,
        d: usize,
        d_prime: usize,
        c: usize,
        c_prime: usize,
        delta: usize,
        delta_prime: usize,
        d_max: usize,
        d_prime_max: usize,
    }

    fn reduction_csv_path() -> std::path::PathBuf {
        let root = repo_root();
        let candidates = [
            root.join("data/stats/dataset_reduction_results.csv"),
            root.join("data/dataset_reduction_results.csv"),
            root.join("data/sampled/dataset_reduction_results.csv"),
        ];
        candidates.into_iter().find(|path| path.exists()).expect(
            "Could not find dataset_reduction_results.csv in data/, data/stats/, or data/sampled/",
        )
    }

    fn parse_reduction_csv() -> Vec<ReductionCsvRecord> {
        let file = reduction_csv_path();
        let csv = fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", file.display()));

        csv.lines()
            .skip(1) // header
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let parts: Vec<&str> = line.split(',').collect();
                assert_eq!(
                    parts.len(),
                    11,
                    "Expected 11 columns in reduction CSV for line: {line}"
                );
                ReductionCsvRecord {
                    instance_name: parts[0].to_string(),
                    n: parts[1].parse().expect("n should be usize"),
                    n_prime: parts[2].parse().expect("n_prime should be usize"),
                    d: parts[3].parse().expect("d should be usize"),
                    d_prime: parts[4].parse().expect("d_prime should be usize"),
                    c: parts[5].parse().expect("c should be usize"),
                    c_prime: parts[6].parse().expect("c_prime should be usize"),
                    delta: parts[7].parse().expect("delta should be usize"),
                    delta_prime: parts[8].parse().expect("delta_prime should be usize"),
                    d_max: parts[9].parse().expect("D should be usize"),
                    d_prime_max: parts[10].parse().expect("D_prime should be usize"),
                }
            })
            .collect()
    }

    fn largest_domain_size(feature_values: &[Vec<i32>]) -> usize {
        feature_values
            .iter()
            .map(|values| values.iter().copied().max().unwrap_or(-1) as isize + 1)
            .max()
            .unwrap_or(0) as usize
    }

    fn max_pairwise_differing_dimensions(feature_values: &[Vec<i32>]) -> usize {
        if feature_values.is_empty() {
            return 0;
        }

        let n = feature_values[0].len();
        let d = feature_values.len();
        if n < 2 {
            return 0;
        }

        let mut best = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut differs = 0usize;
                for f in 0..d {
                    if feature_values[f][i] != feature_values[f][j] {
                        differs += 1;
                    }
                }
                best = best.max(differs);
                if best == d {
                    return best;
                }
            }
        }

        best
    }

    #[test]
    fn same_reductions_as_witty() {
        let records = parse_reduction_csv();
        assert!(
            !records.is_empty(),
            "dataset_reduction_results.csv should contain at least one record"
        );
        let mut mismatches = Vec::new();

        for record in records {
            let mut dataset = DataSet::default();
            let file = repo_root()
                .join("data")
                .join(format!("{}.txt", record.instance_name));
            read_from_file(&mut dataset, &file).unwrap_or_else(|e| {
                panic!(
                    "Failed to read dataset {} at {}: {}",
                    record.instance_name,
                    file.display(),
                    e
                )
            });

            let reduced = reduce_dataset::<AccuracyTask>(&dataset, DataReductionStrategy::Enabled);

            if reduced.stats.before_examples != record.n {
                mismatches.push(format!(
                    "{}: n expected {} got {}",
                    record.instance_name, record.n, reduced.stats.before_examples
                ));
            }
            if reduced.stats.after_examples != record.n_prime {
                mismatches.push(format!(
                    "{}: n_prime expected {} got {}",
                    record.instance_name, record.n_prime, reduced.stats.after_examples
                ));
            }
            if reduced.stats.before_dimensions != record.d {
                mismatches.push(format!(
                    "{}: d expected {} got {}",
                    record.instance_name, record.d, reduced.stats.before_dimensions
                ));
            }
            if reduced.stats.after_dimensions != record.d_prime {
                mismatches.push(format!(
                    "{}: d_prime expected {} got {}",
                    record.instance_name, record.d_prime, reduced.stats.after_dimensions
                ));
            }
            if reduced.stats.before_cuts != record.c {
                mismatches.push(format!(
                    "{}: c expected {} got {}",
                    record.instance_name, record.c, reduced.stats.before_cuts
                ));
            }
            if reduced.stats.after_cuts != record.c_prime {
                mismatches.push(format!(
                    "{}: c_prime expected {} got {}",
                    record.instance_name, record.c_prime, reduced.stats.after_cuts
                ));
            }

            let delta = max_pairwise_differing_dimensions(&dataset.feature_values);
            if delta != record.delta {
                mismatches.push(format!(
                    "{}: delta expected {} got {}",
                    record.instance_name, record.delta, delta
                ));
            }

            let delta_prime = max_pairwise_differing_dimensions(&reduced.feature_values);
            if delta_prime != record.delta_prime {
                mismatches.push(format!(
                    "{}: delta_prime expected {} got {}",
                    record.instance_name, record.delta_prime, delta_prime
                ));
            }

            let d_max = largest_domain_size(&dataset.feature_values);
            if d_max != record.d_max {
                mismatches.push(format!(
                    "{}: D expected {} got {}",
                    record.instance_name, record.d_max, d_max
                ));
            }

            let d_prime_max = largest_domain_size(&reduced.feature_values);
            if d_prime_max != record.d_prime_max {
                mismatches.push(format!(
                    "{}: D_prime expected {} got {}",
                    record.instance_name, record.d_prime_max, d_prime_max
                ));
            }
        }

        assert!(
            mismatches.is_empty(),
            "Reduction stats mismatched {} dataset metrics:\n{}",
            mismatches.len(),
            mismatches.join("\n")
        );
    }
}
