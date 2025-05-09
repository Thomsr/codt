use std::{collections::VecDeque, marker::PhantomData, sync::Arc, time::Instant};

use log::trace;

use crate::{
    model::{dataview::DataView, tree::Tree},
    tasks::OptimizationTask,
};

use super::{node::Node, strategy::SearchStrategy};

pub struct SolveResult<OT: OptimizationTask> {
    pub tree: Arc<Tree<OT>>,
    pub cost_str: String,
    pub graph_expansions: i32,
    pub intermediate_lbs: Vec<(OT::CostType, i32, f64)>,
    pub intermediate_ubs: Vec<(OT::CostType, i32, f64)>,
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

        let mut root: Node<'_, OT, SS> = Node::new(dataview, max_depth);

        let mut graph_expansions = 0;

        let mut path = VecDeque::new();

        let start_time = Instant::now();

        let mut intermediate_lbs = vec![(root.cost_lower_bound, graph_expansions, 0.0)];
        let mut intermediate_ubs = vec![(root.best.cost(), graph_expansions, 0.0)];

        while !root.is_complete() {
            graph_expansions += 1;
            root.select(&mut path);
            trace!("Selected path: {:?}", path);

            let mut current = path.pop_front();
            let mut parent_item = path.pop_front();

            let parent = parent_item
                .as_mut()
                .and_then(|p| p.current_node_mut())
                .unwrap_or(&mut root);
            parent.expand(current.as_mut().unwrap());

            // Return ownership of all the items in the selected path to their respective nodes.
            while let Some(item) = current {
                let parent = parent_item
                    .as_mut()
                    .and_then(|p| p.current_node_mut())
                    .unwrap_or(&mut root);

                parent.backtrack_item(item);

                current = parent_item;
                parent_item = path.pop_front();
            }

            if root.cost_lower_bound > intermediate_lbs.last().unwrap().0 {
                intermediate_lbs.push((root.cost_lower_bound, graph_expansions, start_time.elapsed().as_secs_f64()))
            }

            if root.best.cost() < intermediate_ubs.last().unwrap().0 {
                intermediate_ubs.push((root.best.cost(), graph_expansions, start_time.elapsed().as_secs_f64()))
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

    pub fn new(task: OT, dataview: DataView<'_, OT>) -> Solver<OT, SS> {
        Solver::<'_, OT, SS> {
            task,
            dataview: Some(dataview),
            _ss: PhantomData,
        }
    }
}
