use std::{
    collections::BTreeMap,
    ops::{Bound, Range},
};

use crate::tasks::Cost;
use crate::tasks::OptimizationTask;

pub struct BoundStorage<OT: OptimizationTask, const LEFT_SUBTREE: bool, const LEFT_OF: bool> {
    /// Lookup for finding lower bounds. Note: sorted increasing in threshold
    /// and increasing in cost.
    /// The following is a table of how to interpret the presence of a tuple (threshold, cost):
    ///
    /// TODO: Can we do better than closest? Yes, if we add the worst case to the lb for comparing if they are better.
    /// LEFT_SUBTREE , LEFT_OF  : 0..=threshold, cost is the highest lower bound for the left subtree
    /// RIGHT_SUBTREE, LEFT_OF  : 0..=threshold, cost is the lower bound for the closest right subtree
    /// LEFT_SUBTREE , RIGHT_OF : threshold..=inf, cost is the lower bound for the closest left subtree
    /// RIGHT_SUBTREE, RIGHT_OF : threshold..=inf, cost is the highest lower bound for the right subtree
    lookup: BTreeMap<usize, OT::CostType>,
}

impl<OT: OptimizationTask, const LEFT_SUBTREE: bool, const LEFT_OF: bool> Clone
    for BoundStorage<OT, LEFT_SUBTREE, LEFT_OF>
{
    fn clone(&self) -> Self {
        Self {
            lookup: self.lookup.clone(),
        }
    }
}

impl<OT: OptimizationTask, const LEFT_SUBTREE: bool, const LEFT_OF: bool>
    BoundStorage<OT, LEFT_SUBTREE, LEFT_OF>
{
    pub fn new() -> Self {
        Self {
            lookup: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, threshold: usize, lb: OT::CostType) {
        if lb.is_zero() {
            // Don't bother storing trivial lower bounds.
            return;
        }

        // Find point to insert
        let mut cursor = if LEFT_OF {
            self.lookup.upper_bound_mut(Bound::Included(&threshold))
        } else {
            self.lookup.lower_bound_mut(Bound::Included(&threshold))
        };

        // Update or insert the new lower bound.
        let peek = if LEFT_OF {
            cursor.peek_prev()
        } else {
            cursor.peek_next()
        };

        let only_keep_best = LEFT_SUBTREE == LEFT_OF;

        let needs_insert = match peek {
            Some((&k, v)) => {
                if (only_keep_best || k == threshold) && v.greater_or_not_much_less_than(&lb) {
                    // There already exist a lower bound that is usable in strictly more scenarios that is better.
                    return;
                } else if k == threshold {
                    // We found a better lower bound for an existing node, only need to update it
                    *v = lb;
                    false
                } else {
                    // We found a better lower bound at a new position, we need to insert it.
                    true
                }
            }
            // This is the first lower bound we have, we need to insert it.
            None => true,
        };

        if needs_insert {
            let insertion = if LEFT_OF {
                cursor.insert_before(threshold, lb)
            } else {
                cursor.insert_after(threshold, lb)
            };
            insertion.expect("Order should have been preserved by inserting at the correct index");
        }

        // Remove all thresholds after this that have a worse or equal lower bound.
        if only_keep_best {
            while let Some((_, v)) = if LEFT_OF {
                cursor.next()
            } else {
                cursor.prev()
            } {
                if v.less_or_not_much_greater_than(&lb) {
                    if LEFT_OF {
                        cursor.remove_prev();
                    } else {
                        cursor.remove_next();
                    }
                } else {
                    // If we see a better lower bound, all subsequent ones must also be better.
                    break;
                }
            }
        }
    }

    /// Find the lower bound, up to and *including* this threshold.
    pub fn find(&self, threshold: usize) -> Option<(&usize, &<OT as OptimizationTask>::CostType)> {
        if LEFT_OF {
            self.lookup.range(..(threshold + 1)).last()
        } else {
            self.lookup.range((threshold)..).next()
        }
    }
}

pub struct Pruner<OT: OptimizationTask> {
    pub best_left_subtree_left_of: Vec<BoundStorage<OT, true, true>>,
    pub best_right_subtree_right_of: Vec<BoundStorage<OT, false, false>>,
    pub closest_right_subtree_left_of: Vec<BoundStorage<OT, false, true>>,
    pub closest_left_subtree_right_of: Vec<BoundStorage<OT, true, false>>,
}

impl<OT: OptimizationTask> Pruner<OT> {
    pub fn new(num_features: usize) -> Self {
        Self {
            best_left_subtree_left_of: vec![BoundStorage::new(); num_features],
            best_right_subtree_right_of: vec![BoundStorage::new(); num_features],
            closest_right_subtree_left_of: vec![BoundStorage::new(); num_features],
            closest_left_subtree_right_of: vec![BoundStorage::new(); num_features],
        }
    }

    pub fn insert_left_subtree(&mut self, feature: usize, threshold: usize, lb: OT::CostType) {
        self.best_left_subtree_left_of[feature].insert(threshold, lb);
        self.closest_left_subtree_right_of[feature].insert(threshold, lb);
    }

    pub fn insert_right_subtree(&mut self, feature: usize, threshold: usize, lb: OT::CostType) {
        self.best_right_subtree_right_of[feature].insert(threshold, lb);
        self.closest_right_subtree_left_of[feature].insert(threshold, lb);
    }

    pub fn closest_left_lb(&self, feature: usize, threshold: usize) -> (usize, OT::CostType) {
        let lb_left = self.best_left_subtree_left_of[feature].find(threshold - 1);
        let lb_right = self.closest_right_subtree_left_of[feature].find(threshold - 1);

        match (lb_left, lb_right) {
            (Some((_, &l)), Some((&t, &r))) => (t, l + r),
            _ => (threshold, OT::CostType::ZERO),
        }
    }

    pub fn closest_right_lb(&self, feature: usize, threshold: usize) -> (usize, OT::CostType) {
        let lb_left = self.closest_left_subtree_right_of[feature].find(threshold + 1);
        let lb_right = self.best_right_subtree_right_of[feature].find(threshold + 1);

        match (lb_left, lb_right) {
            (Some((&t, &l)), Some((_, &r))) => (t, l + r),
            _ => (threshold, OT::CostType::ZERO),
        }
    }

    pub fn lb_for(&self, feature: usize, threshold: &Range<usize>) -> OT::CostType {
        // Any left subtree from a tree with threshold <= this one is a lower bound. Vice versa for right.
        let lb_left = self.best_left_subtree_left_of[feature].find(threshold.start);
        // end - 1 as it is an exclusive range
        let lb_right = self.best_right_subtree_right_of[feature].find(threshold.end - 1);

        match (lb_left, lb_right) {
            (Some((_, &l)), Some((_, &r))) => l + r,
            (Some((_, &l)), None) => l,
            (None, Some((_, &r))) => r,
            (None, None) => OT::CostType::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tasks::accuracy::AccuracyTask;

    use super::*;

    #[test]
    fn test_correctness() {
        let mut pruner: Pruner<AccuracyTask> = Pruner::new(1);
        assert_eq!(pruner.lb_for(0, &(10..11)), 0.0.into());
        pruner.insert_left_subtree(0, 5, 4.0.into());
        assert_eq!(pruner.lb_for(0, &(10..11)), 4.0.into());
        pruner.insert_left_subtree(0, 10, 5.0.into());
        assert_eq!(pruner.lb_for(0, &(10..11)), 5.0.into());
        pruner.insert_left_subtree(0, 8, 6.0.into());
        assert_eq!(pruner.lb_for(0, &(10..11)), 6.0.into());
        pruner.insert_left_subtree(0, 11, 20.0.into());
        assert_eq!(pruner.lb_for(0, &(10..11)), 6.0.into());
        pruner.insert_right_subtree(0, 9, 6.0.into());
        assert_eq!(pruner.lb_for(0, &(10..11)), 6.0.into());
        pruner.insert_right_subtree(0, 10, 6.0.into());
        assert_eq!(pruner.lb_for(0, &(10..11)), 12.0.into());
        pruner.insert_right_subtree(0, 10, 8.0.into());
        assert_eq!(pruner.lb_for(0, &(10..11)), 14.0.into());
    }

    #[test]
    fn test_bound_storage_left_leftof() {
        // LEFT_SUBTREE = true, LEFT_OF = true
        let mut bs: BoundStorage<AccuracyTask, true, true> = BoundStorage::new();
        // Insert a non-zero lower bound
        bs.insert(5, 4.0.into());
        assert_eq!(bs.find(5), Some((&5, &4.0.into())));
        // Insert a better lower bound at same threshold (should update)
        bs.insert(5, 6.0.into());
        assert_eq!(bs.find(5), Some((&5, &6.0.into())));
        // Insert a worse lower bound at same threshold (should not update)
        bs.insert(5, 2.0.into());
        assert_eq!(bs.find(5), Some((&5, &6.0.into())));
        // Insert a worse lower bound that reaches more
        bs.insert(3, 5.0.into());
        assert_eq!(bs.find(3), Some((&3, &5.0.into())));
        assert_eq!(bs.find(5), Some((&5, &6.0.into())));
        // Insert a equal lower bound that reaches less
        bs.insert(10, 6.0.into());
        assert_eq!(bs.find(10), Some((&5, &6.0.into())));
        // Insert a better lower bound that reaches more
        bs.insert(2, 7.0.into());
        assert_eq!(bs.find(10), Some((&2, &7.0.into())));
        // Insert a zero lower bound (should not store)
        bs.insert(1, 0.0.into());
        assert_eq!(bs.find(1), None);
    }

    #[test]
    fn test_bound_storage_left_rightof() {
        // LEFT_SUBTREE = true, LEFT_OF = false
        let mut bs: BoundStorage<AccuracyTask, true, false> = BoundStorage::new();
        bs.insert(5, 4.0.into());
        assert_eq!(bs.find(5), Some((&5, &4.0.into())));
        // Insert a better lower bound at same threshold (should update)
        bs.insert(5, 8.0.into());
        assert_eq!(bs.find(5), Some((&5, &8.0.into())));
        // Insert a zero lower bound (should not store)
        bs.insert(20, 0.0.into());
        assert_eq!(bs.find(20), None);
        // Insert a worse lower bound that reaches less (should update only closest)
        bs.insert(2, 2.0.into());
        assert_eq!(bs.find(2), Some((&2, &2.0.into())));
        assert_eq!(bs.find(4), Some((&5, &8.0.into())));
    }

    #[test]
    fn test_bound_storage_right_leftof() {
        // LEFT_SUBTREE = false, LEFT_OF = true
        let mut bs: BoundStorage<AccuracyTask, false, true> = BoundStorage::new();
        bs.insert(3, 2.0.into());
        assert_eq!(bs.find(3), Some((&3, &2.0.into())));
        // Insert a better lower bound at same threshold (should update)
        bs.insert(3, 5.0.into());
        assert_eq!(bs.find(3), Some((&3, &5.0.into())));
        // Insert a worse lower bound at same threshold (should not update)
        bs.insert(3, 1.0.into());
        assert_eq!(bs.find(3), Some((&3, &5.0.into())));
        // Insert a zero lower bound (should not store)
        bs.insert(2, 0.0.into());
        assert_eq!(bs.find(2), None);
    }

    #[test]
    fn test_bound_storage_right_rightof() {
        // LEFT_SUBTREE = false, LEFT_OF = false
        let mut bs: BoundStorage<AccuracyTask, false, false> = BoundStorage::new();
        bs.insert(2, 1.0.into());
        assert_eq!(bs.find(2), Some((&2, &1.0.into())));
        // Insert a better lower bound at same threshold (should update)
        bs.insert(2, 3.0.into());
        assert_eq!(bs.find(2), Some((&2, &3.0.into())));
        // Insert a worse lower bound at same threshold (should not update)
        bs.insert(2, 2.0.into());
        assert_eq!(bs.find(2), Some((&2, &3.0.into())));
        // Insert a zero lower bound (should not store)
        bs.insert(5, 0.0.into());
        assert_eq!(bs.find(5), None);
    }

    #[test]
fn test_pruner_closest_left_lb_threshold() {
    let mut pruner: Pruner<AccuracyTask> = Pruner::new(1);
    // Insert left and right subtree bounds at different thresholds
    pruner.insert_left_subtree(0, 3, 2.0.into());
    pruner.insert_right_subtree(0, 5, 3.0.into());
    pruner.insert_left_subtree(0, 7, 4.0.into());
    pruner.insert_right_subtree(0, 8, 5.0.into());

    // Query for threshold 6: should find left at 3, right at 5, so threshold returned is 5
    let (thresh, cost) = pruner.closest_left_lb(0, 6);
    assert_eq!(thresh, 5);
    assert_eq!(cost, (2.0 + 3.0).into());

    // Query for threshold 7: should find left at 7, right at 5, so threshold returned is 5
    let (thresh, cost) = pruner.closest_left_lb(0, 8);
    assert_eq!(thresh, 5);
    assert_eq!(cost, (4.0 + 3.0).into());

    // Query for threshold 2: no left or right, should return threshold and zero
    let (thresh, cost) = pruner.closest_left_lb(0, 2);
    assert_eq!(thresh, 2);
    assert_eq!(cost, 0.0.into());
}

#[test]
fn test_pruner_closest_right_lb_threshold() {
    let mut pruner: Pruner<AccuracyTask> = Pruner::new(1);
    // Insert left and right subtree bounds at different thresholds
    pruner.insert_left_subtree(0, 2, 2.0.into());
    pruner.insert_right_subtree(0, 4, 3.0.into());
    pruner.insert_left_subtree(0, 6, 4.0.into());
    pruner.insert_right_subtree(0, 9, 5.0.into());

    // Query for threshold 3: should find closest left at 6, best right at 9, so threshold returned is 6
    let (thresh, cost) = pruner.closest_right_lb(0, 2);
    assert_eq!(thresh, 6);
    assert_eq!(cost, (4.0 + 5.0).into());

    // Query for threshold 8: should find closest left at 6, best right at 9, so threshold returned is 6
    let (thresh, cost) = pruner.closest_right_lb(0, 5);
    assert_eq!(thresh, 6);
    assert_eq!(cost, (4.0 + 5.0).into());

    // Query for threshold 20: no left, should return threshold and zero
    let (thresh, cost) = pruner.closest_right_lb(0, 8);
    assert_eq!(thresh, 8);
    assert_eq!(cost, 0.0.into());
}
}
