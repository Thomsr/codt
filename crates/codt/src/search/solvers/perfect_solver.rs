use std::{marker::PhantomData, sync::Arc, time::Instant};

use crate::{
    allocator::{current_thread_memory_usage, reset_current_thread_max_memory_usage},
    model::{
        dataview::DataView,
        tree::{BranchNode, LeafNode, Tree},
    },
    search::{node::Node, solver::{SolveResult, Solver, SolverOptions}, solver::optimal_solver::SolveContext, strategy::SearchStrategy},
    tasks::{Cost, CostSum, OptimizationTask},
};

pub struct PerfectSolverImpl<'a, OT: OptimizationTask, SS: SearchStrategy> {
    task: OT,
    dataview: Option<DataView<'a, OT>>,
    _ss: PhantomData<SS>
}

impl<OT: OptimizationTask, SS: SearchStrategy> Solver<OT> for PerfectSolverImpl<'_, OT, SS> {
    fn solve(&mut self, options: SolverOptions) -> SolveResult<OT> {
        let mut dataview = self.dataview.take().unwrap();
        self.task.prepare_for_data(&mut dataview);

        reset_current_thread_max_memory_usage();

        let start = Instant::now();
        let context = SolveContext {
            task: &self.task,
            ub_strategy: options.ub_strategy,
            terminal_solver: options.terminal_solver,
            branch_relaxation: options.branch_relaxation,
            _ss: PhantomData,
        };

        let mut root: Node<'_, OT, SS> = Node::new(&context, dataview, 0, 0);



        todo!("implementation here");

        let elapsed = start.elapsed().as_secs_f64();
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
}

impl<'a, OT: OptimizationTask, SS: SearchStrategy> PerfectSolverImpl<'a, OT, SS> {
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

}
