// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

use anyhow::{anyhow, Result};
use jlrs::memory::target::frame::GcFrame;
use jlrs::prelude::*;
use jlrs::runtime::handle::async_handle::AsyncHandle;
use std::sync::OnceLock;

const JULIA_SVD_CODE: &str = concat!(
    include_str!("svd_julia_f32.jl"),
    "\n",
    include_str!("svd_julia_f64.jl")
);

static JULIA_RUNTIME: OnceLock<Result<AsyncHandle, String>> = OnceLock::new();

fn runtime() -> Result<&'static AsyncHandle> {
    JULIA_RUNTIME
        .get_or_init(|| {
            let n_threads = std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1);

            let (handle, _runtime_thread) = Builder::new()
                .async_runtime(Tokio::<3>::new(false))
                .n_threads(n_threads)
                .spawn()
                .map_err(|e| format!("failed to start Julia runtime: {e:?}"))?;

            handle
                .blocking_task(|mut frame| -> Result<()> {
                    unsafe {
                        Value::eval_string(&mut frame, JULIA_SVD_CODE).map_err(|e| {
                            anyhow!(
                                "Julia exception loading SVD code: {}",
                                e.display_string_or("<unknown Julia error>")
                            )
                        })?;
                    }
                    Ok(())
                })
                .try_dispatch()
                .map_err(|_| "failed to dispatch Julia initialization task".to_string())?
                .blocking_recv()
                .map_err(|e| format!("failed to receive Julia initialization result: {e}"))?
                .map_err(|e| format!("failed to initialize Julia SVD code: {e:#}"))?;

            Ok(handle)
        })
        .as_ref()
        .map_err(|e| anyhow!(e.clone()))
}

pub(crate) fn blocking_task<T, F>(task: F) -> Result<T>
where
    T: Send + 'static,
    F: for<'base> FnOnce(GcFrame<'base>) -> Result<T> + Send + 'static,
{
    runtime()?
        .blocking_task(task)
        .try_dispatch()
        .map_err(|_| anyhow!("failed to dispatch task to Julia runtime"))?
        .blocking_recv()
        .map_err(|e| anyhow!("failed to receive result from Julia runtime: {e}"))?
}
