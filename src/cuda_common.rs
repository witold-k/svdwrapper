// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

use anyhow::{anyhow, Result};
use cudarc::driver::sys::*;
use std::mem::{size_of, size_of_val};

pub(crate) fn check_cuda(status: CUresult, operation: &str) -> Result<()> {
    if status == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(anyhow!("{operation} failed with CUDA status {status:?}"))
    }
}

pub(crate) fn checked_bytes<T>(elements: usize, what: &str) -> Result<usize> {
    elements
        .checked_mul(size_of::<T>())
        .ok_or_else(|| anyhow!("byte-size overflow while allocating {what}"))
}

pub(crate) struct DeviceBuffer {
    ptr: CUdeviceptr,
    bytes: usize,
    ctx: CUcontext,
}

impl DeviceBuffer {
    pub(crate) fn new(bytes: usize, what: &str, ctx: CUcontext) -> Result<Self> {
        if bytes == 0 {
            return Err(anyhow!("cannot allocate zero bytes for {what}"));
        }

        let mut ptr = 0;
        unsafe {
            check_cuda(cuMemAlloc_v2(&mut ptr, bytes), &format!("allocating {what}"))?;
        }
        Ok(Self { ptr, bytes, ctx })
    }

    pub(crate) fn ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    pub(crate) fn copy_from<T>(&self, source: &[T], what: &str) -> Result<()> {
        let bytes = size_of_val(source);
        if bytes > self.bytes {
            return Err(anyhow!(
                "host-to-device copy for {what} exceeds device buffer: {bytes} > {}",
                self.bytes
            ));
        }
        unsafe {
            check_cuda(cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent before host-to-device copy")?;
            check_cuda(
                cuMemcpyHtoD_v2(self.ptr, source.as_ptr().cast(), bytes),
                &format!("copying {what} to device"),
            )
        }
    }

    pub(crate) fn copy_to<T>(&self, destination: &mut [T], what: &str) -> Result<()> {
        let bytes = size_of_val(destination);
        if bytes > self.bytes {
            return Err(anyhow!(
                "device-to-host copy for {what} exceeds device buffer: {bytes} > {}",
                self.bytes
            ));
        }
        unsafe {
            check_cuda(cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent before device-to-host copy")?;
            check_cuda(
                cuMemcpyDtoH_v2(destination.as_mut_ptr().cast(), self.ptr, bytes),
                &format!("copying {what} to host"),
            )
        }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe {
                let _ = cuCtxSetCurrent(self.ctx);
                let _ = cuMemFree_v2(self.ptr);
            }
        }
    }
}
