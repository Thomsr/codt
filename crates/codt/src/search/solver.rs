use crate::{model::dataview::DataView, tasks::OptimizationTask};

use super::{graph::SearchGraph, node::Node};

pub struct SolveResult<OT: OptimizationTask> {
    pub cost: OT::CostType,
    pub cost_str: String,
}

pub struct Solver<'a, OT: OptimizationTask> {
    task: OT,
    /// Dataview for which the solver finds an optimal decision tree. None during search.
    dataview: Option<DataView<'a, OT::InstanceType>>,
}

impl<OT: OptimizationTask> Solver<'_, OT> {
    pub fn solve(&mut self, max_depth: u32) -> SolveResult<OT> {
        let mut dataview = self.dataview.take().unwrap();

        self.task.prepare_for_data(&mut dataview);

        let _ = max_depth;

        let mut graph = SearchGraph {
            root: Node::new(&self.task, dataview),
        };

        while let Some(mut path) = graph.select() {
            let mut current = path.pop().unwrap();
            let parent = path.pop();
            graph.expand(
                &self.task,
                &mut current,
                parent.as_ref().map(|p| p.left_child.as_ref().unwrap()),
            );
            if let Some(parent) = parent {
                path.push(parent)
            }
            path.push(current);
        }

        let solution = graph.root.cost_upper_bound;

        // Take back ownership of the dataset.
        self.dataview = Some(graph.root.dataview);

        let solution = match solution {
            std::ops::Bound::Included(ub) => ub,
            _ => self.task.leaf_cost(self.dataview.as_ref().unwrap()),
        };

        SolveResult {
            cost_str: self.task.print_cost(&solution),
            cost: solution,
        }
    }

    pub fn new(
        task: OT,
        dataview: DataView<'_, <OT as OptimizationTask>::InstanceType>,
    ) -> Solver<OT> {
        Solver::<'_, OT> {
            task,
            dataview: Some(dataview),
        }
    }
}
