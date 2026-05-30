use std::{cell::Cell, cmp::Ordering};

use log::info;

use crate::{
    allocator::current_thread_memory_usage,
    search::{
        node::{FeatureTest, Node},
        strategy::{SearchStrategy, andor::AndOrSearchStrategy, dfsprio::DfsPrioSearchStrategy},
    },
    tasks::OptimizationTask,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAdaptiveMode {
    AndOr,
    DfsPrio,
}

thread_local! {
    static CURRENT_MODE: Cell<MemoryAdaptiveMode> = const { Cell::new(MemoryAdaptiveMode::AndOr) };
}

fn mode_for_memory_usage(memory_limit: Option<u64>, bytes_current: i64) -> MemoryAdaptiveMode {
    match memory_limit {
        Some(memory_limit) if (bytes_current as u128) * 4 >= (memory_limit as u128) * 3 => {
            MemoryAdaptiveMode::DfsPrio
        }
        _ => MemoryAdaptiveMode::AndOr,
    }
}

fn mode_for_memory_limit(memory_limit: Option<u64>) -> MemoryAdaptiveMode {
    mode_for_memory_usage(memory_limit, current_thread_memory_usage().bytes_current)
}

fn set_current_mode(mode: MemoryAdaptiveMode) {
    CURRENT_MODE.with(|current_mode| current_mode.set(mode));
}

fn current_mode() -> MemoryAdaptiveMode {
    CURRENT_MODE.with(Cell::get)
}

pub struct AndOrDfsPrioSearchStrategy;

impl SearchStrategy for AndOrDfsPrioSearchStrategy {
    fn cmp_item<'a, OT: OptimizationTask, SS: SearchStrategy>(
        a: &FeatureTest<'a, OT, SS>,
        b: &FeatureTest<'a, OT, SS>,
    ) -> Ordering {
        match current_mode() {
            MemoryAdaptiveMode::AndOr => AndOrSearchStrategy::cmp_item(a, b),
            MemoryAdaptiveMode::DfsPrio => DfsPrioSearchStrategy::cmp_item(a, b),
        }
    }

    fn child_priority<'a, OT: OptimizationTask, SS: SearchStrategy>(
        item: &FeatureTest<'a, OT, SS>,
        children: &[Node<'a, OT, SS>; 2],
    ) -> usize {
        match current_mode() {
            MemoryAdaptiveMode::AndOr => AndOrSearchStrategy::child_priority(item, children),
            MemoryAdaptiveMode::DfsPrio => DfsPrioSearchStrategy::child_priority(item, children),
        }
    }

    fn item_front_of_queue_is_lowest_lb<OT: OptimizationTask, SS: SearchStrategy>(
        item: &FeatureTest<OT, SS>,
    ) -> bool {
        match current_mode() {
            MemoryAdaptiveMode::AndOr => {
                AndOrSearchStrategy::item_front_of_queue_is_lowest_lb(item)
            }
            MemoryAdaptiveMode::DfsPrio => {
                DfsPrioSearchStrategy::item_front_of_queue_is_lowest_lb(item)
            }
        }
    }

    fn refresh_memory_mode(memory_limit: Option<u64>) -> Option<MemoryAdaptiveMode> {
        let mode = mode_for_memory_limit(memory_limit);
        let previous_mode = current_mode();
        if previous_mode != mode {
            info!(
                "Switching search strategy from {:?} to {:?} at {} bytes current memory (limit: {:?})",
                previous_mode,
                mode,
                current_thread_memory_usage().bytes_current,
                memory_limit,
            );
        }
        set_current_mode(mode);
        Some(mode)
    }
}

#[cfg(test)]
mod tests {
    use super::{MemoryAdaptiveMode, mode_for_memory_usage};

    #[test]
    fn switches_at_three_quarters_of_memory_limit() {
        assert_eq!(mode_for_memory_usage(None, 123), MemoryAdaptiveMode::AndOr);
        assert_eq!(
            mode_for_memory_usage(Some(100), 74),
            MemoryAdaptiveMode::AndOr
        );
        assert_eq!(
            mode_for_memory_usage(Some(100), 75),
            MemoryAdaptiveMode::DfsPrio
        );
    }
}
