use std::{
    cell::{Cell, RefCell},
    collections::{HashSet, VecDeque},
    marker::PhantomData,
    sync::Arc,
    time::{Duration, Instant},
};

use log::{info, trace};
use rustc_hash::FxHashMap;

use crate::{
    allocator::{current_thread_memory_usage, reset_current_thread_max_memory_usage},
    model::{dataview::DataView, difference_table::DifferenceTable, tree::Tree},
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
    pub cart_ub: bool,
    pub cart_ub_patience: usize,
    pub memory_limit: Option<u64>,
    pub difference_table: Option<&'a DifferenceTable>,
    cache_max_branch_budget: Option<usize>,
    solution_cache: RefCell<FxHashMap<Vec<usize>, Arc<Tree<OT>>>>,
    cache_lookups: Cell<usize>,
    cache_useful_hits: Cell<usize>,
    _ss: PhantomData<SS>,
}

impl<OT: OptimizationTask, SS: SearchStrategy> SolveContext<'_, OT, SS> {
    fn cache_key(dataview: &DataView<OT>) -> Vec<usize> {
        let mut key = dataview.instance_ids.clone();
        key.sort_unstable();
        key
    }

    pub fn cached_solution(&self, dataview: &DataView<OT>) -> Option<Arc<Tree<OT>>> {
        self.cache_lookups.set(self.cache_lookups.get() + 1);
        let solution = self
            .solution_cache
            .borrow()
            .get(&Self::cache_key(dataview))
            .cloned();
        if solution.is_some() {
            self.cache_useful_hits.set(self.cache_useful_hits.get() + 1);
        }
        solution
    }

    pub fn cache_solution(&self, dataview: &DataView<OT>, solution: Arc<Tree<OT>>) {
        self.solution_cache
            .borrow_mut()
            .insert(Self::cache_key(dataview), solution);
    }

    pub fn cache_eligible(&self, lower_bound: &OT::CostType, upper_bound: &OT::CostType) -> bool {
        if self.cache_max_branch_budget.is_some_and(|max_budget| {
            OT::remaining_branch_budget(lower_bound, upper_bound)
                .is_some_and(|budget| budget <= max_budget)
        }) {
            return true;
        }
        false
    }
}

impl<OT: OptimizationTask, SS: SearchStrategy> Solver<OT> for SolverImpl<'_, OT, SS> {
    fn solve(&mut self, options: SolverOptions) -> SolveResult<OT> {
        let mut dataview = self.dataview.take().unwrap();

        self.task.prepare_for_data(&mut dataview);

        let use_pair_lowerbound = options.lb_strategy.contains(&LowerBoundStrategy::Pair);
        let use_improvement_lowerbound = options
            .lb_strategy
            .contains(&LowerBoundStrategy::Improvement);
        let use_one_off_lowerbound = options.lb_strategy.contains(&LowerBoundStrategy::OneOff);

        let difference_table =
            if use_pair_lowerbound || use_improvement_lowerbound || use_one_off_lowerbound {
                Some(DifferenceTable::new(&dataview))
            } else {
                None
            };

        let context = SolveContext {
            task: &self.task,
            lb_strategy: options.lb_strategy,
            ub_strategy: options.ub_strategy,
            cart_ub: options.cart_ub,
            cart_ub_patience: options.cart_ub_patience,
            memory_limit: options.memory_limit,
            difference_table: difference_table.as_ref(),
            cache_max_branch_budget: options.cache_max_branch_budget,
            solution_cache: RefCell::default(),
            cache_lookups: Cell::default(),
            cache_useful_hits: Cell::default(),
            _ss: PhantomData,
        };

        let one_off_witnesses = if use_one_off_lowerbound {
            context
                .difference_table
                .as_ref()
                .map(|table| table.one_off_witnesses())
                .unwrap_or_default()
        } else {
            Default::default()
        };

        info!("One off witnesses: {}", one_off_witnesses.len());

        let mut graph_expansions = 0;

        let mut path = VecDeque::new();

        let start_time = Instant::now();
        let mut elapsed = Duration::ZERO;

        let root_pair_lower_bound = if use_pair_lowerbound {
            context.difference_table.as_ref().map(|table| {
                let view = table.view_for_dataview(&dataview);
                pair_lower_bound::<OT>(&view)
            })
        } else {
            None
        };

        let mut root: Node<'_, OT, SS> = Node::new(&context, dataview, 0, true, one_off_witnesses);

        if let Some(pair_lb) = root_pair_lower_bound {
            info!("Pair lower bound: {}", pair_lb);
            OT::update_lowerbound(&mut root.cost_lower_bound, &pair_lb);
            info!(
                "Root lower bound after pair lower bound: {}",
                root.cost_lower_bound
            );
        }

        let mut intermediate_lbs = vec![(root.cost_lower_bound, graph_expansions, 0.0)];
        let mut intermediate_ubs = vec![(root.best.cost(), graph_expansions, 0.0)];

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
            root.select(&context, &mut path, 0);
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
                    intermediate_lbs.push((actual_lb, graph_expansions, elapsed.as_secs_f64()));
                    info!(
                        "New intermediate LB: {} (expansions: {}, time: {:.2}s)",
                        actual_lb,
                        graph_expansions,
                        elapsed.as_secs_f64()
                    );
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
                    ));
                    info!(
                        "New intermediate UB: {} (expansions: {}, time: {:.2}s)",
                        root.best.cost(),
                        graph_expansions,
                        elapsed.as_secs_f64()
                    );
                }
            }
        }

        info!(
            "Final root bounds: lower {}, upper {}, best {}, complete {}",
            root.cost_lower_bound,
            root.cost_upper_bound,
            root.best.cost(),
            root.is_complete()
        );
        let solution = root.best;
        let status = if OT::is_perfect_solution_cost(&solution.cost()) {
            SolveStatus::PerfectTreeFound
        } else {
            SolveStatus::NoPerfectTree
        };

        // Take back ownership of the dataset.
        self.dataview = Some(root.dataview);

        let memory_usage_bytes = current_thread_memory_usage().bytes_max;
        info!(
            "Solution cache: {} useful hits / {} lookups ({:.1}%), {} entries",
            context.cache_useful_hits.get(),
            context.cache_lookups.get(),
            if context.cache_lookups.get() == 0 {
                0.0
            } else {
                context.cache_useful_hits.get() as f64 / context.cache_lookups.get() as f64 * 100.0
            },
            context.solution_cache.borrow().len(),
        );

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
    use std::{cell::RefCell, collections::HashSet, sync::Arc};

    use rustc_hash::FxHashMap;

    use super::SolveContext;
    use crate::{
        model::{
            dataset::DataSet,
            dataview::DataView,
            instance::LabeledInstance,
            tree::{LeafNode, Tree},
        },
        search::{
            solver::{
                SearchStrategyEnum, SolveStatus, SolverOptions, UpperboundStrategy,
                solver_with_strategy,
            },
            strategy::dfs::DfsSearchStrategy,
        },
        tasks::{LexicographicCost, accuracy::AccuracyTask},
        test_support::{read_from_file, repo_root},
    };

    #[test]
    fn solution_cache_uses_branch_budget_and_subset_ids() {
        let mut dataset = DataSet::<LabeledInstance<i32>>::default();
        dataset.add_instance(LabeledInstance::new(0), [0.0]);
        dataset.add_instance(LabeledInstance::new(1), [1.0]);
        dataset.preprocess_after_adding_instances();

        let view = DataView::from_dataset(&dataset, false);
        let mut reordered_view = DataView::from_dataset(&dataset, false);
        reordered_view.instance_ids.reverse();

        let task = AccuracyTask::new();
        let context: SolveContext<'_, AccuracyTask, DfsSearchStrategy> = SolveContext {
            task: &task,
            lb_strategy: HashSet::new(),
            ub_strategy: UpperboundStrategy::SolutionsOnly,
            cart_ub: false,
            cart_ub_patience: 0,
            memory_limit: None,
            difference_table: None,
            cache_max_branch_budget: Some(3),
            solution_cache: RefCell::new(FxHashMap::default()),
            cache_lookups: Default::default(),
            cache_useful_hits: Default::default(),
            _ss: Default::default(),
        };
        let solution = Arc::new(Tree::Leaf(LeafNode {
            cost: LexicographicCost::new(1, 0),
            label: 0,
        }));

        assert!(
            context.cache_eligible(&LexicographicCost::new(1, 2), &LexicographicCost::new(1, 5))
        );
        assert!(
            !context.cache_eligible(&LexicographicCost::new(1, 2), &LexicographicCost::new(1, 6))
        );
        assert!(
            !context.cache_eligible(&LexicographicCost::new(0, 2), &LexicographicCost::new(1, 2))
        );

        context.cache_solution(&view, solution.clone());
        assert!(Arc::ptr_eq(
            &context.cached_solution(&reordered_view).unwrap(),
            &solution
        ));
    }

    #[test]
    fn perfect_tree_pure_dataset_is_single_leaf() {
        let mut dataset = DataSet::<LabeledInstance<i32>>::default();
        dataset.add_instance(LabeledInstance::new(0), [0.0]);
        dataset.add_instance(LabeledInstance::new(0), [1.0]);
        dataset.add_instance(LabeledInstance::new(0), [2.0]);
        dataset.preprocess_after_adding_instances();

        let full_view = DataView::from_dataset(&dataset, false);
        let mut solver = solver_with_strategy(
            AccuracyTask::new(),
            full_view,
            SearchStrategyEnum::BfsBalanceSmallLb,
        );

        let result = solver.solve(SolverOptions::default());

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

        let full_view = DataView::from_dataset(&dataset, false);
        let mut solver = solver_with_strategy(
            AccuracyTask::new(),
            full_view,
            SearchStrategyEnum::BfsBalanceSmallLb,
        );

        let result = solver.solve(SolverOptions::default());

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
    fn cached_search_finds_minimum_size_xor_tree() {
        let mut dataset = DataSet::<LabeledInstance<i32>>::default();
        dataset.add_instance(LabeledInstance::new(0), [0.0, 0.0]);
        dataset.add_instance(LabeledInstance::new(1), [0.0, 1.0]);
        dataset.add_instance(LabeledInstance::new(1), [1.0, 0.0]);
        dataset.add_instance(LabeledInstance::new(0), [1.0, 1.0]);
        dataset.preprocess_after_adding_instances();

        let full_view = DataView::from_dataset(&dataset, false);
        let mut solver = solver_with_strategy(
            AccuracyTask::new(),
            full_view,
            SearchStrategyEnum::AndOrDfsPrio,
        );
        let result = solver.solve(SolverOptions {
            cart_ub: false,
            cache_max_branch_budget: Some(3),
            ..SolverOptions::default()
        });

        assert_eq!(result.status, SolveStatus::PerfectTreeFound);
        let tree = result.tree.expect("Expected a perfect XOR tree");
        assert_eq!(tree.cost(), LexicographicCost::new(0, 3));
        assert_eq!(tree.branch_count(), 3);
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

        let full_view = DataView::from_dataset(&dataset, false);
        let mut solver = solver_with_strategy(
            AccuracyTask::new(),
            full_view,
            SearchStrategyEnum::BfsBalanceSmallLb,
        );

        let result = solver.solve(SolverOptions::default());

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
