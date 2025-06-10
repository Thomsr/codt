//! Replaces global memory allocator with one that counts the allocated bytes.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::RefCell;

#[derive(Clone, Copy, Default, Debug)]
pub struct MemoryUsage {
    /// Total memory allocated in bytes
    pub bytes_total: i64,
    /// Current memory use in bytes
    pub bytes_current: i64,
    /// Maximum simultaneous memory used in bytes
    pub bytes_max: i64,
}

thread_local! {
    static USAGE: RefCell<MemoryUsage> = RefCell::new(MemoryUsage::default());
}

pub fn current_thread_memory_usage() -> MemoryUsage {
    USAGE.with(|usage| *usage.borrow())
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        USAGE.with(|usage| {
            let mut usage = usage.borrow_mut();
            usage.bytes_total += l.size() as i64;
            usage.bytes_current += l.size() as i64;
            if usage.bytes_current > 0 {
                usage.bytes_max = usage.bytes_max.max(usage.bytes_current);
            }
        });

        unsafe { System.alloc(l) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, l: Layout) {
        USAGE.with(|usage| {
            let mut usage = usage.borrow_mut();
            usage.bytes_current -= l.size() as i64;
        });

        unsafe {
            System.dealloc(ptr, l);
        }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator {};
