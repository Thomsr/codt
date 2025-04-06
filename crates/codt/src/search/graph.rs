use crate::tasks::OptimizationTask;

use super::node::{Node, QueueItem};

pub struct SearchGraph<'a, OT: OptimizationTask> {
    pub root: Node<'a, OT>,
}

impl<'a, OT: OptimizationTask> SearchGraph<'a, OT> {
    pub fn find_lowest_cost_split(&self, item: &QueueItem<'a, OT>) -> i32 {
        // get rs = right solution
        // get ls = left solution
        // for x = distance from a solution:
        // lower bound for left = max(ls.left, rs.left-worst*x)
        // lower bound for right = max(rs.right, ls.right-worst*x)
        // find minimum index of left_lb + right_lb. If in a flat area, pick the center of that area.
        item.split_points.start
    }

    pub fn select(&mut self) -> Option<Vec<QueueItem<'a, OT>>> {
        let next_from_root = self.root.queue.pop();

        if let Some(next_from_root) = next_from_root {
            let mut path = Vec::new();
            let mut current = next_from_root;

            while let Some(left_child) = &mut current.left_child {
                let next = left_child.queue.pop();
                if let Some(item) = next {
                    path.push(current);
                    current = item;
                } else {
                    break;
                }
            }
            path.push(current);
            Some(path)
        } else {
            // done
            None
        }
    }

    pub fn expand(
        &mut self,
        task: &OT,
        item: &mut QueueItem<'a, OT>,
        parent: Option<&Node<'a, OT>>,
    ) {
        // if expanded: error;

        let parent = parent.unwrap_or(&self.root);
        let x = self.find_lowest_cost_split(item);
        // if x - start > 0: split off threshold and add to queue
        // if end - x > 0: split off threshold and add to queue
        item.split_points = x..(x + 1);

        let (left, right) = parent.split(task, item.feature, x);
        item.left_child = Some(left);
        item.right_child = Some(right);
    }
}
