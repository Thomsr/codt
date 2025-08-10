use std::{cmp::Ordering, marker::PhantomData};

use crate::{
    search::{
        node::{FeatureTest, Node},
        strategy::SearchStrategy,
    },
    tasks::OptimizationTask,
};

pub struct CuriosityHeuristic;
pub struct RandomHeuristic;
pub struct LBSupportHeuristic<const LB: i64 = 1, const SUPPORT: i64 = 1>;

/// Info used for the heuristics, so that the heuristic struct does not need to be generic over the tasks.
pub struct HeuristicInfo {
    lb: f64,
    support: usize,
    random_value: u64,
}

pub trait BfsHeuristic {
    fn heuristic(info: HeuristicInfo) -> f64;
    fn generate_random_value() -> bool {
        false
    }
}

impl BfsHeuristic for CuriosityHeuristic {
    fn heuristic(info: HeuristicInfo) -> f64 {
        info.lb / info.support as f64
    }
}

impl<const LB: i64, const SUPPORT: i64> BfsHeuristic for LBSupportHeuristic<LB, SUPPORT> {
    fn heuristic(info: HeuristicInfo) -> f64 {
        LB as f64 * info.lb + SUPPORT as f64 * info.support as f64
    }
}

impl BfsHeuristic for RandomHeuristic {
    fn heuristic(info: HeuristicInfo) -> f64 {
        info.random_value as f64
    }

    fn generate_random_value() -> bool {
        true
    }
}

pub struct BfsSearchStrategy<H: BfsHeuristic> {
    _heuristic: PhantomData<H>,
}

impl<H: BfsHeuristic> SearchStrategy for BfsSearchStrategy<H> {
    fn cmp_item<'a, OT: OptimizationTask, SS: SearchStrategy>(
        a: &FeatureTest<'a, OT, SS>,
        b: &FeatureTest<'a, OT, SS>,
    ) -> Ordering {
        a.lowest_descendant_heuristic()
            .partial_cmp(&b.lowest_descendant_heuristic())
            .expect("No NaN allowed in heuristic value")
            .then(a.feature_rank.cmp(&b.feature_rank))
            .then(a.split_point.cmp(&b.split_point))
    }

    fn child_priority<'a, OT: OptimizationTask, SS: SearchStrategy>(
        _item: &FeatureTest<'a, OT, SS>,
        children: &[Node<'a, OT, SS>; 2],
    ) -> usize {
        if children[0].lowest_descendant_heuristic <= children[1].lowest_descendant_heuristic {
            0
        } else {
            1
        }
    }

    fn item_front_of_queue_is_lowest_lb<OT: OptimizationTask, SS: SearchStrategy>(
        _item: &FeatureTest<OT, SS>,
    ) -> bool {
        false
    }

    fn heuristic<OT: OptimizationTask, SS: SearchStrategy>(item: &FeatureTest<OT, SS>) -> f64 {
        let lb_float: f64 = item
            .cost_lower_bound
            .try_into()
            .expect("Global best first search only works for numeric costs");
        H::heuristic(HeuristicInfo {
            lb: lb_float,
            support: item.support,
            random_value: item.random_value,
        })
    }

    fn generate_random_value() -> bool {
        H::generate_random_value()
    }
}
