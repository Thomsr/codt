use std::collections::BinaryHeap;

pub trait PQ<T> {
    fn is_empty(&self) -> bool;
    fn pop(&mut self) -> Option<T>;
    fn push(&mut self, item: T);
    fn peek(&self) -> Option<&T>;
    fn clear(&mut self);
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T> + 'a
    where
        T: 'a;
}

pub struct BinaryHeapQueue<T: Ord> {
    queue: BinaryHeap<T>,
}

impl<T: Ord> Default for BinaryHeapQueue<T> {
    fn default() -> Self {
        Self {
            queue: BinaryHeap::new(),
        }
    }
}

impl<T: Ord> PQ<T> for BinaryHeapQueue<T> {
    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    fn pop(&mut self) -> Option<T> {
        self.queue.pop()
    }

    fn push(&mut self, item: T) {
        self.queue.push(item);
    }

    fn peek(&self) -> Option<&T> {
        self.queue.peek()
    }

    fn clear(&mut self) {
        self.queue.clear();
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T> + 'a
    where
        T: 'a,
    {
        self.queue.iter()
    }
}

pub struct VecQueue<T: Ord> {
    queue: Vec<T>,
    next: Option<usize>,
}

impl<T: Ord> Default for VecQueue<T> {
    fn default() -> Self {
        Self {
            queue: Vec::new(),
            next: None,
        }
    }
}

impl<T: Ord> VecQueue<T> {
    fn set_next(&mut self) {
        if let Some((next, _)) = self.queue.iter().enumerate().min_by_key(|item| item.1) {
            self.next = Some(next);
        } else {
            self.next = None;
        }
    }
}

impl<T: Ord> PQ<T> for VecQueue<T> {
    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    fn pop(&mut self) -> Option<T> {
        if let Some(next) = self.next {
            let ret = self.queue.swap_remove(next);
            self.set_next();
            Some(ret)
        } else {
            None
        }
    }

    fn push(&mut self, item: T) {
        if let Some(next) = self.peek() {
            if item < *next {
                // If the new item is smaller than the current next, it should be the new next.
                self.next = Some(self.queue.len());
            }
        } else {
            // If there is no next item, this is the first item.
            self.next = Some(0);
        };
        self.queue.push(item);
    }

    fn peek(&self) -> Option<&T> {
        if let Some(next) = self.next {
            Some(&self.queue[next])
        } else {
            None
        }
    }

    fn clear(&mut self) {
        self.queue.clear();
        self.next = None;
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T> + 'a
    where
        T: 'a,
    {
        self.queue.iter()
    }
}
