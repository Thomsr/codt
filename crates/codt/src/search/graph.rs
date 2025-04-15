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

    /// Select the path to the node to expand next.
    pub fn select(&mut self, path_buffer: &mut Vec<QueueItem<'a, OT, SS>>) -> bool {
        assert!(path_buffer.is_empty());

        if self.root.is_complete() {
            return false;
        }

        let next_from_root = self.root.queue.pop();

        if let Some(next_from_root) = next_from_root {
            let mut current = next_from_root;

            while let Some(children) = &mut current.children {
                let item = children[current.current_child]
                    .queue
                    .pop()
                    .expect("Queue cannot be empty since the root is not complete");
                path_buffer.push(current);
                current = item;
            }
            path_buffer.push(current);
            true
        } else {
            unreachable!("Queue cannot be empty since the root is not complete")
        }
    }

    /// Expand an item from the queue one level by selecting a concrete split and instantiating its children.
    pub fn expand(
        &mut self,
        item: &mut QueueItem<'a, OT, SS>,
        parent: Option<&mut Node<'a, OT, SS>>,
    ) {
        let parent = parent.unwrap_or(&mut self.root);

        assert!(!item.is_expanded());
        assert!(parent.remaining_depth_budget > 0);

        let x = parent.find_lowest_cost_split(item.feature, &item.split_points);
        assert!(item.split_points.start <= x && x < item.split_points.end);

        let (left_item, right_item) = item.split_at(x);

        if let Some(left_item) = left_item {
            parent.queue.push(left_item);
        }
        if let Some(right_item) = right_item {
            parent.queue.push(right_item);
        }

        let (left, right) = parent.split(item.feature, x);
        item.children = Some([left, right]);
    }

    pub fn backtrack(&mut self, path: &mut Vec<QueueItem<'a, OT, SS>>) {
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
