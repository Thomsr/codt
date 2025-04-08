use std::{
    collections::BTreeMap,
    ops::{Bound, Range},
};

use crate::tasks::OptimizationTask;

pub struct Pruner<OT: OptimizationTask> {
    /// Lookup for finding lower bounds. Note: sorted increasing in threshold
    /// and increasing in cost.
    /// Presence of tuple (threshold, cost) means that from 0 to
    /// threshold inclusive, cost is the highest lower bound for
    /// the left subtree found so far.
    pub best_left_subtree_left_of: Vec<BTreeMap<i32, OT::CostType>>,

    /// Lookup for finding lower bounds. Note: sorted increasing in threshold
    /// and decreasing in cost.
    /// Presence of tuple (threshold, cost) means that from threshold to
    /// inf inclusive, cost is the highest lower bound for
    /// the right subtree found so far.
    pub best_right_subtree_right_of: Vec<BTreeMap<i32, OT::CostType>>,
}

impl<OT: OptimizationTask> Pruner<OT> {
    pub fn new(num_features: usize) -> Self {
        Self {
            best_left_subtree_left_of: vec![BTreeMap::new(); num_features],
            best_right_subtree_right_of: vec![BTreeMap::new(); num_features],
        }
    }

    pub fn insert_left_subtree(&mut self, feature: usize, threshold: i32, lb: OT::CostType) {
        // Find point to insert
        let mut cursor =
            self.best_left_subtree_left_of[feature].upper_bound_mut(Bound::Included(&threshold));

        // Update or insert the new lower bound.
        let needs_insert = match cursor.peek_prev() {
            Some((&k, v)) => {
                if *v >= lb {
                    return;
                } else if k == threshold {
                    *v = lb;
                    false
                } else {
                    true
                }
            }
            None => true,
        };

        if needs_insert {
            cursor
                .insert_before(threshold, lb)
                .expect("Order should have been preserved by inserting at the correct index");
        }

        // Remove all thresholds after this that have a worse or equal lower bound.
        while let Some((_, v)) = cursor.next() {
            if *v <= lb {
                cursor.remove_prev();
            } else {
                // If we see a better lower bound, all subsequent ones must also be better.
                break;
            }
        }
    }

    pub fn insert_right_subtree(&mut self, feature: usize, threshold: i32, lb: OT::CostType) {
        // Find point to insert
        let mut cursor =
            self.best_right_subtree_right_of[feature].lower_bound_mut(Bound::Included(&threshold));

        // Update or insert the new lower bound.
        let needs_insert = match cursor.peek_next() {
            Some((&k, v)) => {
                if *v >= lb {
                    return;
                } else if k == threshold {
                    *v = lb;
                    false
                } else {
                    true
                }
            }
            None => true,
        };

        if needs_insert {
            cursor
                .insert_after(threshold, lb)
                .expect("Order should have been preserved by inserting at the correct index");
        }

        // Remove all thresholds after this that have a worse or equal lower bound.
        while let Some((_, v)) = cursor.prev() {
            if *v <= lb {
                cursor.remove_next();
            } else {
                // If we see a better lower bound, all subsequent ones must also be better.
                break;
            }
        }
    }

    pub fn lb_for(&self, feature: usize, threshold: &Range<i32>) -> OT::CostType {
        // Any left subtree from a tree with threshold <= this one is a lower bound. Vice versa for right.
        let lb_left = self.best_left_subtree_left_of[feature]
            .range(..(threshold.start + 1))
            .last();
        let lb_right = self.best_right_subtree_right_of[feature]
            .range((threshold.end - 1)..)
            .next();

        match (lb_left, lb_right) {
            (Some((_, &l)), Some((_, &r))) => l + r,
            (Some((_, &l)), None) => l,
            (None, Some((_, &r))) => r,
            (None, None) => OT::MIN_COST,
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
        assert_eq!(pruner.lb_for(0, &(10..11)), 0);
        pruner.insert_left_subtree(0, 5, 4);
        assert_eq!(pruner.lb_for(0, &(10..11)), 4);
        pruner.insert_left_subtree(0, 10, 5);
        assert_eq!(pruner.lb_for(0, &(10..11)), 5);
        pruner.insert_left_subtree(0, 8, 6);
        assert_eq!(pruner.lb_for(0, &(10..11)), 6);
        pruner.insert_left_subtree(0, 11, 20);
        assert_eq!(pruner.lb_for(0, &(10..11)), 6);
        pruner.insert_right_subtree(0, 9, 6);
        assert_eq!(pruner.lb_for(0, &(10..11)), 6);
        pruner.insert_right_subtree(0, 10, 6);
        assert_eq!(pruner.lb_for(0, &(10..11)), 12);
        pruner.insert_right_subtree(0, 10, 8);
        assert_eq!(pruner.lb_for(0, &(10..11)), 14);
    }
}
