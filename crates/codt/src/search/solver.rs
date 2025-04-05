use crate::{model::dataview::DataView, tasks::OptimizationTask};

pub struct SolveResult<OT: OptimizationTask> {
    pub cost: OT::CostType,
    pub cost_str: String,
}

pub struct Solver<'a, OT: OptimizationTask> {
    task: OT,
    dataview: DataView<'a, OT::InstanceType>,
}

impl<OT: OptimizationTask> Solver<'_, OT> {
    pub fn solve(&mut self) -> SolveResult<OT> {
        self.task.prepare_for_data(&mut self.dataview);

        let solution = self.task.leaf_cost(&self.dataview);

        SolveResult {
            cost_str: self.task.print_cost(&solution),
            cost: solution,
        }
    }

    pub fn new(
        task: OT,
        dataview: DataView<'_, <OT as OptimizationTask>::InstanceType>,
    ) -> Solver<OT> {
        Solver::<'_, OT> { task, dataview }
    }
}
