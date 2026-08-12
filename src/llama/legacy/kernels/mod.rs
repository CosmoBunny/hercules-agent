//! Pure-Rust fused quant kernels (no C/FFI).
//!
//! Stream-dequant into a tiny stack buffer and accumulate GEMV so we never
//! allocate a full f32 weight matrix.

mod gemv_fused;
mod slice;

pub use gemv_fused::{gemv_quant_fused, supports_fused_gemv};
pub use slice::dequant_slice;
