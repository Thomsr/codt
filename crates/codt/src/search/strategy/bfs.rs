use std::{cmp::Ordering, marker::PhantomData};

use crate::{
    search::node::{Node, QueueItem},
    tasks::OptimizationTask,
};

use super::SearchStrategy;

pub struct LBHeuristic;
pub struct CuriosityHeuristic;
pub struct GOSDTHeuristic;

pub trait BfsHeuristic {
    fn heuristic_from_lb_and_support(lb: f64, support: usize) -> f64;
}

impl BfsHeuristic for LBHeuristic {
    fn heuristic_from_lb_and_support(lb: f64, _support: usize) -> f64 {
        lb
    }
}

impl BfsHeuristic for CuriosityHeuristic {
    fn heuristic_from_lb_and_support(lb: f64, support: usize) -> f64 {
        lb / support as f64
    }
}

impl BfsHeuristic for GOSDTHeuristic {
    fn heuristic_from_lb_and_support(lb: f64, support: usize) -> f64 {
        // TODO figure out why lb + support seems to work well. Maybe due to faster LB discovery
        lb + support as f64
    }
}

pub struct BfsSearchStrategy<H: BfsHeuristic> {
    _heuristic: PhantomData<H>,
}

impl<H: BfsHeuristic> SearchStrategy for BfsSearchStrategy<H> {
    fn cmp_item<'a, OT: OptimizationTask, SS: SearchStrategy>(
        a: &QueueItem<'a, OT, SS>,
        b: &QueueItem<'a, OT, SS>,
    ) -> Ordering {
        a.lowest_descendant_heuristic()
            .partial_cmp(&b.lowest_descendant_heuristic())
            .expect("No NaN allowed in heuristic value")
            .then(a.feature.cmp(&b.feature))
            .then(a.split_points.start.cmp(&b.split_points.start))
    }

    fn child_priority<'a, OT: OptimizationTask, SS: SearchStrategy>(
        _item: &QueueItem<'a, OT, SS>,
        children: &[Node<'a, OT, SS>; 2],
    ) -> usize {
        if children[0].lowest_descendant_heuristic <= children[1].lowest_descendant_heuristic {
            0
        } else {
            1
        }
    }

    fn item_front_of_queue_is_lowest_lb<OT: OptimizationTask, SS: SearchStrategy>(
        _item: &QueueItem<OT, SS>,
    ) -> bool {
        false
    }

    fn heuristic<OT: OptimizationTask, SS: SearchStrategy>(item: &QueueItem<OT, SS>) -> f64 {
        let lb_float: f64 = item
            .cost_lower_bound
            .try_into()
            .expect("Global best first search only works for numeric costs");
        H::heuristic_from_lb_and_support(lb_float, item.support)
    }
}
