use std::marker::PhantomData;

use crate::tasks::OptimizationTask;

use super::{
    node::{Node, QueueItem},
    strategy::SearchStrategy,
};

pub struct SearchGraph<'a, OT: OptimizationTask, SS: SearchStrategy> {
    pub root: Node<'a, OT, SS>,
    _ss: PhantomData<SS>,
}

impl<'a, OT: OptimizationTask, SS: SearchStrategy> SearchGraph<'a, OT, SS> {
    pub fn new(root: Node<'a, OT, SS>) -> Self {
        Self {
            root,
            _ss: PhantomData,
        }
    }

    pub fn find_lowest_cost_split(&self, item: &QueueItem<'a, OT, SS>) -> i32 {
        // get rs = right solution
        // get ls = left solution
        // for x = distance from a solution:
        // lower bound for left = max(ls.left, rs.left-worst*x)
        // lower bound for right = max(rs.right, ls.right-worst*x)
        // find minimum index of left_lb + right_lb. If in a flat area, pick the center of that area.
        item.split_points.start + (item.split_points.end - item.split_points.start) / 2
    }

    pub fn select(&mut self) -> Option<Vec<QueueItem<'a, OT, SS>>> {
        if self.root.is_complete() {
            return None;
        }

        let next_from_root = self.root.queue.pop();

        if let Some(next_from_root) = next_from_root {
            let mut path = Vec::new();
            let mut current = next_from_root;

            while let Some(children) = &mut current.children {
                let item = children[current.current_child]
                    .queue
                    .pop()
                    .expect("Queue cannot be empty since the root is not complete");
                path.push(current);
                current = item;
            }
            path.push(current);
            Some(path)
        } else {
            unreachable!("Queue cannot be empty since the root is not complete")
        }
    }

    pub fn expand(
        &mut self,
        task: &OT,
        item: &mut QueueItem<'a, OT, SS>,
        parent: Option<&mut Node<'a, OT, SS>>,
    ) {
        assert!(!item.is_expanded());

        let x = self.find_lowest_cost_split(item);
        assert!(item.split_points.start <= x && x < item.split_points.end);

        let (left_item, right_item) = item.split_at(x);

        let parent = parent.unwrap_or(&mut self.root);
        if let Some(left_item) = left_item {
            if parent
                .dataview
                .feature_range_remains(item.feature, &left_item.split_points)
            {
                parent.queue.push(left_item);
            }
        }
        if let Some(right_item) = right_item {
            if parent
                .dataview
                .feature_range_remains(item.feature, &right_item.split_points)
            {
                parent.queue.push(right_item);
            }
        }

        let (left, right) = parent.split(task, item.feature, x);
        item.children = Some([left, right]);
    }

    pub fn backtrack(&mut self, mut path: Vec<QueueItem<'a, OT, SS>>) {
        let mut current = path.pop();
        let mut parent_item = path.pop();

        // Return ownership of all the items in the selected path to their respective nodes.
        while let Some(item) = current {
            let parent = parent_item
                .as_mut()
                .and_then(|p| p.current_node_mut())
                .unwrap_or(&mut self.root);

            parent.backtrack_item(item);

            current = parent_item;
            parent_item = path.pop();
        }
    }
}
