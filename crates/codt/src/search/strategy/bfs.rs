use std::{cmp::Ordering, marker::PhantomData};

use crate::{search::node::QueueItem, tasks::OptimizationTask};

use super::SearchStrategy;

pub struct LBHeuristic;
pub struct CuriosityHeuristic;
pub struct GOSDTHeuristic;

pub trait BfsHeuristic {
    fn heuristic_from_lb_and_remaining_fraction(lb: f64, remaining_fraction: f64) -> f64;
}

impl BfsHeuristic for LBHeuristic {
    fn heuristic_from_lb_and_remaining_fraction(lb: f64, _remaining_fraction: f64) -> f64 {
        lb
    }
}

impl BfsHeuristic for CuriosityHeuristic {
    fn heuristic_from_lb_and_remaining_fraction(lb: f64, remaining_fraction: f64) -> f64 {
        lb / remaining_fraction
    }
}

impl BfsHeuristic for GOSDTHeuristic {
    fn heuristic_from_lb_and_remaining_fraction(lb: f64, remaining_fraction: f64) -> f64 {
        // TODO figure out why lb + remaining_fraction seems to work well.
        lb - remaining_fraction
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
        a: &crate::search::node::Node<'a, OT, SS>,
        b: &crate::search::node::Node<'a, OT, SS>,
    ) -> usize {
        if a.lowest_descendant_heuristic <= b.lowest_descendant_heuristic {
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

    fn heuristic_from_lb_and_remaining_fraction<OT: OptimizationTask>(
        lb: OT::CostType,
        remaining_fraction: f64,
    ) -> f64 {
        let lb_float: f64 = lb
            .try_into()
            .expect("Global best first search only works for numeric costs");
        H::heuristic_from_lb_and_remaining_fraction(lb_float, remaining_fraction)
    }
}
