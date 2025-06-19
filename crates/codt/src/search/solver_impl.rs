use std::{
    collections::VecDeque,
    marker::PhantomData,
    time::{Duration, Instant},
};

use log::trace;

use crate::{
    allocator::current_thread_memory_usage,
    model::dataview::DataView,
    search::{
        node::Node,
        queue::PQ,
        solver::{SolveResult, Solver, SolverOptions, TerminalSolver, UpperboundStrategy},
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
    pub ub_strategy: UpperboundStrategy,
    pub terminal_solver: TerminalSolver,
    _ss: PhantomData<SS>,
}

impl<OT: OptimizationTask, SS: SearchStrategy> Solver<OT> for SolverImpl<'_, OT, SS> {
    fn solve(&mut self, options: SolverOptions) -> SolveResult<OT> {
        let mut dataview = self.dataview.take().unwrap();

        self.task.prepare_for_data(&mut dataview);

        let context = SolveContext {
            task: &self.task,
            ub_strategy: UpperboundStrategy::SolutionsOnly,
            terminal_solver: TerminalSolver::LeftRight,
            _ss: PhantomData,
        };

        let mut root: Node<'_, OT, SS> = Node::new(&context, dataview, options.max_depth);

        let mut graph_expansions = 0;

        let mut path = VecDeque::new();

        let start_time = Instant::now();
        let mut elapsed = Duration::ZERO;

        let mut intermediate_lbs = vec![(root.cost_lower_bound, graph_expansions, 0.0)];
        let mut intermediate_ubs = vec![(root.best.cost(), graph_expansions, 0.0)];

        while !root.is_complete()
            && options.timeout.is_none_or(|timeout| elapsed < timeout)
            && options.memory_limit.is_none_or(|memory_limit| {
                current_thread_memory_usage().bytes_current < memory_limit as i64
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
                            let mut lb = root.pruner.lb_for(i.feature, &i.split_points)
                                + context.task.branching_cost();
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

        // Take back ownership of the dataset.
        self.dataview = Some(root.dataview);

        SolveResult {
            cost_str: self.task.print_cost(&solution.cost()),
            tree: solution,
            graph_expansions,
            intermediate_lbs,
            intermediate_ubs,
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
