use std::{marker::PhantomData, sync::Arc};

use log::trace;

use crate::{
    model::{dataview::DataView, tree::Tree},
    tasks::OptimizationTask,
};

use super::{graph::SearchGraph, node::Node, strategy::SearchStrategy};

pub struct SolveResult<OT: OptimizationTask> {
    pub tree: Arc<Tree<OT>>,
    pub cost_str: String,
    pub graph_expansions: i32,
}

pub struct Solver<'a, OT: OptimizationTask, SS: SearchStrategy> {
    task: OT,
    /// Dataview for which the solver finds an optimal decision tree. None during search.
    dataview: Option<DataView<'a, OT>>,
    _ss: PhantomData<SS>,
}

impl<OT: OptimizationTask, SS: SearchStrategy> Solver<'_, OT, SS> {
    pub fn solve(&mut self, max_depth: u32) -> SolveResult<OT> {
        let mut dataview = self.dataview.take().unwrap();

        self.task.prepare_for_data(&mut dataview);

        let mut graph: SearchGraph<'_, OT, SS> = SearchGraph::new(Node::new(dataview, max_depth));

        let mut graph_expansions = 0;

        let mut path = Vec::new();

        while graph.select(&mut path) {
            graph_expansions += 1;
            trace!("Selected path: {:?}", path);

            let mut current = path.pop().unwrap();
            let mut parent = path.pop();
            graph.expand(
                &mut current,
                parent.as_mut().and_then(|p| p.current_node_mut()),
            );
            if let Some(parent) = parent {
                path.push(parent)
            }
            path.push(current);
            graph.backtrack(&mut path);

            trace!(
                "LB: {:?}, UB: {:?}",
                graph.root.cost_lower_bound,
                graph.root.best.cost()
            );
        }

        let solution = graph.root.best;

        // Take back ownership of the dataset.
        self.dataview = Some(graph.root.dataview);

        SolveResult {
            cost_str: self.task.print_cost(&solution.cost()),
            tree: solution,
            graph_expansions,
        }
    }

    pub fn new(task: OT, dataview: DataView<'_, OT>) -> Solver<OT, SS> {
        Solver::<'_, OT, SS> {
            task,
            dataview: Some(dataview),
            _ss: PhantomData,
        }
    }
}
