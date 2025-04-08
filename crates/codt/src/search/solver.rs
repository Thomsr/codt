use std::marker::PhantomData;

use log::trace;

use crate::{model::dataview::DataView, tasks::OptimizationTask};

use super::{graph::SearchGraph, node::Node, strategy::SearchStrategy};

pub struct SolveResult<OT: OptimizationTask> {
    pub cost: OT::CostType,
    pub cost_str: String,
}

pub struct Solver<'a, OT: OptimizationTask, SS: SearchStrategy> {
    task: OT,
    /// Dataview for which the solver finds an optimal decision tree. None during search.
    dataview: Option<DataView<'a, OT::InstanceType>>,
    _ss: PhantomData<SS>,
}

impl<OT: OptimizationTask, SS: SearchStrategy> Solver<'_, OT, SS> {
    pub fn solve(&mut self, max_depth: u32) -> SolveResult<OT> {
        let mut dataview = self.dataview.take().unwrap();

        self.task.prepare_for_data(&mut dataview);

        let mut graph: SearchGraph<'_, OT, SS> =
            SearchGraph::new(Node::new(&self.task, dataview, max_depth));

        while let Some(mut path) = graph.select() {
            trace!("Selected path: {:?}", path);

            let mut current = path.pop().unwrap();
            let mut parent = path.pop();
            graph.expand(
                &self.task,
                &mut current,
                parent.as_mut().and_then(|p| p.current_node_mut()),
            );
            if let Some(parent) = parent {
                path.push(parent)
            }
            path.push(current);
            graph.backtrack(path);

            trace!(
                "LB: {:?}, UB: {:?}",
                graph.root.cost_lower_bound, graph.root.cost_upper_bound
            );
        }

        let solution = graph.root.cost_upper_bound;

        // Take back ownership of the dataset.
        self.dataview = Some(graph.root.dataview);

        SolveResult {
            cost_str: self.task.print_cost(&solution),
            cost: solution,
        }
    }

    pub fn new(
        task: OT,
        dataview: DataView<'_, <OT as OptimizationTask>::InstanceType>,
    ) -> Solver<OT, SS> {
        Solver::<'_, OT, SS> {
            task,
            dataview: Some(dataview),
            _ss: PhantomData,
        }
    }
}
