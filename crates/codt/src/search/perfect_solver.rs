use std::{sync::Arc, time::Instant};

use crate::{
    allocator::{current_thread_memory_usage, reset_current_thread_max_memory_usage},
    model::{
        dataview::DataView,
        tree::{BranchNode, LeafNode, Tree},
    },
    search::solver::{SolveResult, Solver, SolverOptions},
    tasks::{Cost, CostSum, OptimizationTask},
};

pub struct PerfectSolverImpl<'a, OT: OptimizationTask> {
    task: OT,
    dataview: Option<DataView<'a, OT>>,
}

struct PerfectSearchContext<'a, OT: OptimizationTask> {
    start: Instant,
    timeout: Option<std::time::Duration>,
    memory_limit: Option<u64>,
    graph_expansions: i32,
    stopped: bool,
    _ot: std::marker::PhantomData<&'a OT>,
}

impl<'a, OT: OptimizationTask> PerfectSearchContext<'a, OT> {
    fn should_stop(&mut self) -> bool {
        if self.stopped {
            return true;
        }

        if self.timeout.is_some_and(|timeout| self.start.elapsed() >= timeout) {
            self.stopped = true;
            return true;
        }

        if self.memory_limit.is_some_and(|limit| {
            current_thread_memory_usage().bytes_current >= limit as i64
        }) {
            self.stopped = true;
            return true;
        }

        false
    }

    fn make_leaf(&self, dataview: &DataView<'a, OT>) -> Arc<Tree<OT>> {
        Arc::new(Tree::Leaf(LeafNode {
            cost: dataview.cost_summer.cost(),
            label: dataview.cost_summer.label(),
        }))
    }

    fn branch_count(tree: &Tree<OT>) -> u32 {
        match tree {
            Tree::Leaf(_) => 0,
            Tree::Branch(branch) => {
                1 + Self::branch_count(branch.left_child.as_ref())
                    + Self::branch_count(branch.right_child.as_ref())
            }
        }
    }

    fn better_by_cost_then_size(current: &Arc<Tree<OT>>, candidate: &Arc<Tree<OT>>) -> bool {
        if candidate.cost().strictly_less_than(&current.cost()) {
            return true;
        }
        if current.cost().strictly_less_than(&candidate.cost()) {
            return false;
        }
        Self::branch_count(candidate.as_ref()) < Self::branch_count(current.as_ref())
    }

    fn solve_with_budget(
        &mut self,
        dataview: &DataView<'a, OT>,
        branch_budget: u32,
    ) -> Arc<Tree<OT>> {
        let mut best = self.make_leaf(dataview);

        if branch_budget == 0 || dataview.num_instances() <= 1 {
            return best;
        }

        for feature in 0..dataview.num_features() {
            for split_index in 0..dataview.possible_split_values[feature].len() {
                if self.should_stop() {
                    return best;
                }

                self.graph_expansions += 1;

                let (left_view, right_view) = dataview.split(feature, split_index);

                for left_budget in 0..branch_budget {
                    let right_budget = branch_budget - 1 - left_budget;
                    let left = self.solve_with_budget(&left_view, left_budget);
                    let right = self.solve_with_budget(&right_view, right_budget);
                    let candidate = Arc::new(Tree::Branch(BranchNode {
                        cost: left.cost() + right.cost(),
                        split_feature: feature,
                        split_threshold: dataview.threshold_from_split(feature, split_index),
                        left_child: left,
                        right_child: right,
                    }));

                    if Self::better_by_cost_then_size(&best, &candidate) {
                        best = candidate;
                    }
                }
            }
        }

        best
    }
}

impl<'a, OT: OptimizationTask> PerfectSolverImpl<'a, OT> {
    pub fn new(task: OT, dataview: DataView<'a, OT>) -> Self {
        Self {
            task,
            dataview: Some(dataview),
        }
    }
}

impl<OT: OptimizationTask> Solver<OT> for PerfectSolverImpl<'_, OT> {
    fn solve(&mut self, options: SolverOptions) -> SolveResult<OT> {
        let mut dataview = self.dataview.take().unwrap();
        self.task.prepare_for_data(&mut dataview);

        reset_current_thread_max_memory_usage();

        let start = Instant::now();
        let mut context = PerfectSearchContext {
            start,
            timeout: options.timeout,
            memory_limit: options.memory_limit,
            graph_expansions: 0,
            stopped: false,
            _ot: std::marker::PhantomData,
        };

        let mut solution = context.make_leaf(&dataview);
        let max_branches = dataview.num_instances().saturating_sub(1) as u32;

        for branch_budget in 0..=max_branches {
            if context.should_stop() {
                break;
            }

            let candidate = context.solve_with_budget(&dataview, branch_budget);
            if PerfectSearchContext::<OT>::better_by_cost_then_size(&solution, &candidate) {
                solution = candidate;
            }

            // The first zero-cost solution found by increasing budget has minimum tree size.
            if solution.cost().is_zero() {
                break;
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let graph_expansions = context.graph_expansions;
        drop(context);

        self.dataview = Some(dataview);

        let memory_usage_bytes = current_thread_memory_usage().bytes_max;

        SolveResult {
            cost_str: self.task.print_cost(&solution.cost()),
            tree: solution.clone(),
            graph_expansions,
            memory_usage_bytes,
            intermediate_lbs: vec![(solution.cost(), graph_expansions, elapsed)],
            intermediate_ubs: vec![(solution.cost(), graph_expansions, elapsed)],
        }
    }

    fn d0d1_lowerbound(&mut self, _max_depth: u32) -> (OT::CostType, OT::CostType) {
        let mut dataview = self.dataview.take().unwrap();
        self.task.prepare_for_data(&mut dataview);

        let d0 = dataview.cost_summer.cost();
        let mut d1 = d0;
        for feature in 0..dataview.num_features() {
            for split_index in 0..dataview.possible_split_values[feature].len() {
                let (left_view, right_view) = dataview.split(feature, split_index);
                let candidate = left_view.cost_summer.cost() + right_view.cost_summer.cost();
                if candidate.strictly_less_than(&d1) {
                    d1 = candidate;
                }
            }
        }

        self.dataview = Some(dataview);

        (d0, d1)
    }
}

#[cfg(test)]
mod tests {
    use super::PerfectSolverImpl;
    use crate::{
        model::{
            dataset::DataSet,
            dataview::DataView,
            instance::LabeledInstance,
            tree::Tree,
        },
        search::solver::{SearchStrategyEnum, Solver, SolverOptions, solver_with_strategy},
        tasks::{Cost, OptimizationTask, accuracy::AccuracyTask},
    };

    fn branch_count(tree: &Tree<AccuracyTask>) -> u32 {
        match tree {
            Tree::Leaf(_) => 0,
            Tree::Branch(branch) => {
                1 + branch_count(branch.left_child.as_ref())
                    + branch_count(branch.right_child.as_ref())
            }
        }
    }

    #[test]
    fn finds_perfect_tree_easy() {
        let mut dataset = DataSet::<LabeledInstance<i32>>::default();
        dataset.add_instance(LabeledInstance::new(0), [0.0]);
        dataset.add_instance(LabeledInstance::new(0), [0.2]);
        dataset.add_instance(LabeledInstance::new(1), [0.8]);
        dataset.add_instance(LabeledInstance::new(1), [1.0]);
        dataset.preprocess_after_adding_instances();

        AccuracyTask::preprocess_dataset(&mut dataset);
        let view = DataView::from_dataset(&dataset);
        let task = AccuracyTask::new(0.0);

        let mut solver = solver_with_strategy(task, view, SearchStrategyEnum::Perfect);
        let result = solver.solve(SolverOptions {
            ub_strategy: crate::search::solver::UpperboundStrategy::ForRemainingInterval,
            terminal_solver: crate::search::solver::TerminalSolver::Leaf,
            branch_relaxation: crate::search::solver::BranchRelaxation::None,
            track_intermediates: false,
            max_depth: 0,
            timeout: None,
            memory_limit: None,
        });

        println!("final tree (strategy): {}", result.tree);

        assert!(
            result.tree.cost().is_zero(),
            "Expected zero training error on a perfectly separable dataset"
        );
        assert_eq!(
            branch_count(result.tree.as_ref()),
            1,
            "Expected one split to be sufficient and minimal for this dataset"
        );
    }

    #[test]
    fn finds_perfect_tree_easy2() {
      let mut dataset = DataSet::<LabeledInstance<i32>>::default();

      dataset.add_instance(LabeledInstance::new(0), [0.0]);
      dataset.add_instance(LabeledInstance::new(0), [0.1]);

      dataset.add_instance(LabeledInstance::new(1), [0.3]);
      dataset.add_instance(LabeledInstance::new(1), [0.4]);

      dataset.add_instance(LabeledInstance::new(0), [0.6]);
      dataset.add_instance(LabeledInstance::new(0), [0.7]);

      dataset.add_instance(LabeledInstance::new(1), [0.9]);
      dataset.add_instance(LabeledInstance::new(1), [1.0]);

      dataset.preprocess_after_adding_instances();

      AccuracyTask::preprocess_dataset(&mut dataset);
      let view = DataView::from_dataset(&dataset);
      let task = AccuracyTask::new(0.0);

      let mut solver = solver_with_strategy(task, view, SearchStrategyEnum::Perfect);
      let result = solver.solve(SolverOptions {
          ub_strategy: crate::search::solver::UpperboundStrategy::ForRemainingInterval,
          terminal_solver: crate::search::solver::TerminalSolver::Leaf,
          branch_relaxation: crate::search::solver::BranchRelaxation::None,
          track_intermediates: false,
          max_depth: 0,
          timeout: None,
          memory_limit: None,
      });

      println!("final tree (strategy): {}", result.tree);

      assert!(
          result.tree.cost().is_zero(),
          "Expected zero training error on a perfectly separable dataset"
      );

      assert!(
          branch_count(result.tree.as_ref()) >= 3,
          "Expected multiple splits due to alternating class regions"
      );
    }

    #[test]
  fn five_feature_tree() {
    let mut dataset = DataSet::<LabeledInstance<i32>>::default();

    // 5 features: [f0, f1, f2, f3, f4]
    // Label rule:
    // class = 1 if (f0 > 0.5 && f1 > 0.5) || (f2 > 0.5 && f3 > 0.5)
    // otherwise 0
    //
    // This forces the tree to branch on multiple different features.

    dataset.add_instance(LabeledInstance::new(0), [0.0, 0.0, 0.0, 0.0, 0.0]);
    dataset.add_instance(LabeledInstance::new(0), [0.6, 0.0, 0.0, 0.0, 0.0]);
    dataset.add_instance(LabeledInstance::new(0), [0.0, 0.6, 0.0, 0.0, 0.0]);

    dataset.add_instance(LabeledInstance::new(1), [0.6, 0.6, 0.0, 0.0, 0.0]); // f0 & f1

    dataset.add_instance(LabeledInstance::new(0), [0.0, 0.0, 0.6, 0.0, 0.0]);
    dataset.add_instance(LabeledInstance::new(0), [0.0, 0.0, 0.0, 0.6, 0.0]);

    dataset.add_instance(LabeledInstance::new(1), [0.0, 0.0, 0.6, 0.6, 0.0]); // f2 & f3

    // Mixed negatives to prevent shortcut splits
    dataset.add_instance(LabeledInstance::new(0), [0.6, 0.0, 0.6, 0.0, 0.0]);
    dataset.add_instance(LabeledInstance::new(0), [0.0, 0.6, 0.0, 0.6, 0.0]);

    dataset.preprocess_after_adding_instances();

    AccuracyTask::preprocess_dataset(&mut dataset);
    let view = DataView::from_dataset(&dataset);
    let task = AccuracyTask::new(0.0);

    let mut solver = solver_with_strategy(task, view, SearchStrategyEnum::Perfect);
    let result = solver.solve(SolverOptions {
        ub_strategy: crate::search::solver::UpperboundStrategy::ForRemainingInterval,
        terminal_solver: crate::search::solver::TerminalSolver::Leaf,
        branch_relaxation: crate::search::solver::BranchRelaxation::None,
        track_intermediates: false,
        max_depth: 0,
        timeout: None,
        memory_limit: None,
    });

    println!("final tree (strategy): {}", result.tree.pretty());

    // Perfect separability
    assert!(
        result.tree.cost().is_zero(),
        "Expected zero training error"
    );

    // Should require multiple splits and multiple features
    assert!(
        branch_count(result.tree.as_ref()) >= 3,
        "Expected at least 3 splits due to multi-feature interaction"
    );
  }
}
