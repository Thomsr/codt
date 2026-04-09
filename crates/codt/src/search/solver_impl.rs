use std::{
    collections::{HashSet, VecDeque},
    marker::PhantomData,
    time::{Duration, Instant},
};

use log::{info, trace};

use crate::{
    allocator::{current_thread_memory_usage, reset_current_thread_max_memory_usage},
    model::dataview::DataView,
    search::{
        lower_bounds::pair::pair_lower_bound,
        node::Node,
        queue::PQ,
        solver::{
            LowerBoundStrategy, SolveResult, SolveStatus, Solver, SolverOptions, UpperboundStrategy,
        },
        strategy::SearchStrategy,
    },
    tasks::{Cost, OptimizationTask},
};

pub struct SolverImpl<'a, OT: OptimizationTask, SS: SearchStrategy> {
    task: OT,
    /// Dataview for which the solver finds an optimal decision tree. None during search.
    dataview: Option<DataView<'a, OT>>,
    _ss: PhantomData<SS>,
}

pub struct SolveContext<'a, OT: OptimizationTask, SS: SearchStrategy> {
    pub task: &'a OT,
    pub lb_strategy: HashSet<LowerBoundStrategy>,
    pub ub_strategy: UpperboundStrategy,
    _ss: PhantomData<SS>,
}

impl<OT: OptimizationTask, SS: SearchStrategy> Solver<OT> for SolverImpl<'_, OT, SS> {
    fn solve(&mut self, options: SolverOptions) -> SolveResult<OT> {
        let mut dataview = self.dataview.take().unwrap();

        self.task.prepare_for_data(&mut dataview);

        let context = SolveContext {
            task: &self.task,
            lb_strategy: options.lb_strategy,
            ub_strategy: options.ub_strategy,
            _ss: PhantomData,
        };

        let mut graph_expansions = 0;

        let mut path = VecDeque::new();

        let start_time = Instant::now();
        let mut elapsed = Duration::ZERO;

        let pair_lb = context
            .lb_strategy
            .contains(&LowerBoundStrategy::Pair)
            .then(|| pair_lower_bound::<OT>(&dataview));

        let mut root: Node<'_, OT, SS> = Node::new(&context, dataview, 0);

        let mut intermediate_lbs = vec![(root.cost_lower_bound, graph_expansions, 0.0)];
        let mut intermediate_ubs = vec![(root.best.cost(), graph_expansions, 0.0)];

        if let Some(pair_lb) = pair_lb {
            info!("Pair lower bound: {}", pair_lb);
            OT::update_lowerbound(&mut root.cost_lower_bound, &pair_lb);
        }

        // Ignore memory usage of previous invocations.
        reset_current_thread_max_memory_usage();

        while !root.is_complete()
            && !options.timeout.is_some_and(|timeout| elapsed >= timeout)
            && !options.memory_limit.is_some_and(|memory_limit| {
                current_thread_memory_usage().bytes_current >= memory_limit as i64
            })
        {
            graph_expansions += 1;

            // The initial source does not matter, since we always substitute the root manually.
            root.select(&mut path, 0);
            trace!("Selected path: {:?}", path);

            let mut current = path.pop_front();
            let mut parent_item = path.pop_front();

            let parent = parent_item
                .as_mut()
                .and_then(|(_, p)| p.child_by_idx(current.as_ref().unwrap().0))
                .unwrap_or(&mut root);

            parent.expand(&context, &mut current.as_mut().unwrap().1);

            // Return ownership of all the items in the selected path to their respective nodes.
            while let Some((parent_node_idx, item)) = current {
                let parent = parent_item
                    .as_mut()
                    .and_then(|(_, p)| p.child_by_idx(parent_node_idx))
                    .unwrap_or(&mut root);

                parent.backtrack_item(&context, item);

                current = parent_item;
                parent_item = path.pop_front();
            }

            elapsed = start_time.elapsed();

            if options.track_intermediates {
                let lowest_remaining_lb =
                    root.queue
                        .iter()
                        .fold(None, |val: Option<OT::CostType>, i| {
                            let mut lb = root.lb_for(i) + context.task.branching_cost();
                            OT::update_lowerbound(&mut lb, &i.cost_lower_bound);
                            if val.is_none() || val.unwrap().strictly_greater_than(&lb) {
                                Some(lb)
                            } else {
                                val
                            }
                        });

                let mut actual_lb = root.cost_lower_bound;
                if let Some(lb) = lowest_remaining_lb {
                    if lb.strictly_greater_than(&actual_lb) {
                        actual_lb = lb
                    }
                }

                if actual_lb.strictly_greater_than(&intermediate_lbs.last().unwrap().0) {
                    intermediate_lbs.push((actual_lb, graph_expansions, elapsed.as_secs_f64()))
                }

                if root
                    .best
                    .cost()
                    .strictly_less_than(&intermediate_ubs.last().unwrap().0)
                {
                    intermediate_ubs.push((
                        root.best.cost(),
                        graph_expansions,
                        elapsed.as_secs_f64(),
                    ))
                }
            }
        }

        let solution = root.best;
        let status = if OT::is_perfect_solution_cost(&solution.cost()) {
            SolveStatus::PerfectTreeFound
        } else {
            SolveStatus::NoPerfectTree
        };

        // Take back ownership of the dataset.
        self.dataview = Some(root.dataview);

        let memory_usage_bytes = current_thread_memory_usage().bytes_max;

        SolveResult {
            status,
            cost_str: match status {
                SolveStatus::PerfectTreeFound => Some(self.task.print_cost(&solution.cost())),
                SolveStatus::NoPerfectTree => None,
            },
            tree: match status {
                SolveStatus::PerfectTreeFound => Some(solution),
                SolveStatus::NoPerfectTree => None,
            },
            graph_expansions,
            intermediate_lbs,
            intermediate_ubs,
            memory_usage_bytes,
        }
    }

    fn d0d1_lowerbound(&mut self) -> (OT::CostType, OT::CostType) {
        let mut dataview = self.dataview.take().unwrap();

        self.task.prepare_for_data(&mut dataview);

        let context = SolveContext {
            task: &self.task,
            lb_strategy: HashSet::from([LowerBoundStrategy::ClassCount]),
            ub_strategy: UpperboundStrategy::ForRemainingInterval,
            _ss: PhantomData,
        };

        let root: Node<'_, OT, SS> = Node::new(&context, dataview, 0);
        let d0lb = root.cost_lower_bound;

        let mut d1lb = None;
        for feature_test in root.queue.iter() {
            for split_value in feature_test.split_points.clone() {
                let (left, right) = root.dataview.split(feature_test.feature, split_value);
                let left = Node::new(&context, left, 0);
                let right = Node::new(&context, right, 0);
                let lb = left.cost_lower_bound + right.cost_lower_bound;
                if d1lb.is_none_or(|x| lb.strictly_less_than(&x)) {
                    d1lb = Some(lb);
                }
            }
        }

        // Take back ownership of the dataset.
        self.dataview = Some(root.dataview);

        (d0lb, d1lb.unwrap())
    }
}

impl<'a, OT: OptimizationTask, SS: SearchStrategy> SolverImpl<'a, OT, SS> {
    pub fn new(task: OT, dataview: DataView<'a, OT>) -> Self {
        Self {
            task,
            dataview: Some(dataview),
            _ss: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, time::Duration};

    use crate::{
        model::{dataset::DataSet, dataview::DataView, instance::LabeledInstance, tree::Tree},
        search::solver::{
            LowerBoundStrategy, SearchStrategyEnum, SolveStatus, SolverOptions, UpperboundStrategy,
            solver_with_strategy,
        },
        tasks::{LexicographicCost, accuracy::AccuracyTask},
        test_support::{read_from_file, repo_root},
    };

    fn default_options() -> SolverOptions {
        SolverOptions {
            lb_strategy: HashSet::from([LowerBoundStrategy::ClassCount, LowerBoundStrategy::Pair]),
            ub_strategy: UpperboundStrategy::ForRemainingInterval,
            track_intermediates: false,
            timeout: Some(Duration::from_secs(5)),
            memory_limit: None,
        }
    }

    #[test]
    fn perfect_tree_pure_dataset_is_single_leaf() {
        let mut dataset = DataSet::<LabeledInstance<i32>>::default();
        dataset.add_instance(LabeledInstance::new(0), [0.0]);
        dataset.add_instance(LabeledInstance::new(0), [1.0]);
        dataset.add_instance(LabeledInstance::new(0), [2.0]);
        dataset.preprocess_after_adding_instances();

        let full_view = DataView::from_dataset(&dataset);
        let mut solver = solver_with_strategy(
            AccuracyTask::new(),
            full_view,
            SearchStrategyEnum::BfsBalanceSmallLb,
        );

        let result = solver.solve(default_options());

        assert_eq!(result.status, SolveStatus::PerfectTreeFound);
        let tree = result.tree.expect("Expected a perfect tree for pure data");
        println!("pure dataset tree: {}", tree);
        assert!(matches!(tree.as_ref(), Tree::Leaf(_)));
        assert_eq!(tree.branch_count(), 0);
        assert_eq!(tree.cost(), LexicographicCost::new(0, 0));

        for x in [0.0, 1.5, 10.0] {
            assert_eq!(tree.predict(vec![x]), 0);
        }
    }

    #[test]
    fn perfect_tree_single_split_dataset_has_one_branch() {
        let mut dataset = DataSet::<LabeledInstance<i32>>::default();
        dataset.add_instance(LabeledInstance::new(0), [0.0]);
        dataset.add_instance(LabeledInstance::new(0), [1.0]);
        dataset.add_instance(LabeledInstance::new(1), [2.0]);
        dataset.add_instance(LabeledInstance::new(1), [3.0]);
        dataset.preprocess_after_adding_instances();

        let full_view = DataView::from_dataset(&dataset);
        let mut solver = solver_with_strategy(
            AccuracyTask::new(),
            full_view,
            SearchStrategyEnum::BfsBalanceSmallLb,
        );

        let result = solver.solve(default_options());

        assert_eq!(result.status, SolveStatus::PerfectTreeFound);
        let tree = result
            .tree
            .expect("Expected a perfect tree for linearly separable data");
        println!("single split dataset tree: {}", tree);
        assert!(matches!(tree.as_ref(), Tree::Branch(_)));
        assert_eq!(tree.branch_count(), 1);
        assert_eq!(tree.cost(), LexicographicCost::new(0, 1));
    }

    #[test]
    fn sampled_dataset_runs_and_reports_status() {
        let mut dataset = DataSet::<LabeledInstance<i32>>::default();
        let file = repo_root().join("data/sampled/appendicitis_0.2_0.txt");
        read_from_file(&mut dataset, &file).unwrap();

        assert!(
            dataset.instances.len() > 10,
            "Expected a larger-than-toy sampled dataset"
        );

        let full_view = DataView::from_dataset(&dataset);
        let mut solver = solver_with_strategy(
            AccuracyTask::new(),
            full_view,
            SearchStrategyEnum::BfsBalanceSmallLb,
        );

        let result = solver.solve(default_options());

        println!("sampled dataset status: {:?}", result.status);
        match result.status {
            SolveStatus::PerfectTreeFound => {
                let tree = result.tree.expect("Tree must exist when status is perfect");
                println!("sampled dataset tree: {}", tree);
                assert_eq!(tree.cost().primary, 0);
            }
            SolveStatus::NoPerfectTree => {
                assert!(result.tree.is_none());
            }
        }
    }
}
