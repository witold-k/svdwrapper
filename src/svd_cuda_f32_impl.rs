// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

use crate::cuda_common::{check_cuda, checked_bytes, CurrentContextGuard, DeviceBuffer};
use crate::svd::{SvdBackend, SvdMode, SvdOutput, SvdResult};
use anyhow::{anyhow, Result};
use cudarc::cusolver::sys::*;
use cudarc::driver::sys::*;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};

pub struct CudaF32Svd {
    handle: cusolverDnHandle_t,
    ctx: CUcontext,
    device: CUdevice,
}

impl CudaF32Svd {
    pub fn new() -> Result<Self> {
        unsafe {
            check_cuda(cuInit(0), "cuInit")?;

            let mut device: CUdevice = 0;
            check_cuda(cuDeviceGet(&mut device, 0), "cuDeviceGet")?;

            let mut ctx: CUcontext = std::ptr::null_mut();
            check_cuda(
                cuDevicePrimaryCtxRetain(&mut ctx, device),
                "cuDevicePrimaryCtxRetain",
            )?;

            let _context = match CurrentContextGuard::activate(ctx) {
                Ok(context) => context,
                Err(error) => {
                    let _ = cuDevicePrimaryCtxRelease_v2(device);
                    return Err(error);
                }
            };

            let mut handle: cusolverDnHandle_t = std::ptr::null_mut();
            let status = cusolverDnCreate(&mut handle);
            if status != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                let _ = cuDevicePrimaryCtxRelease_v2(device);
                return Err(anyhow!("cusolverDnCreate failed with status {status:?}"));
            }

            Ok(Self {
                handle,
                ctx,
                device,
            })
        }
    }
}

impl Drop for CudaF32Svd {
    fn drop(&mut self) {
        if let Ok(_context) = CurrentContextGuard::activate(self.ctx) {
            unsafe {
                let _ = cusolverDnDestroy(self.handle);
            }
        }
        unsafe {
            let _ = cuDevicePrimaryCtxRelease_v2(self.device);
        }
    }
}

impl SvdBackend<f32> for CudaF32Svd {
    fn compute_svd(
        &self,
        a: &ArrayBase<impl Data<Elem = f32>, Ix2>,
        mode: SvdMode,
    ) -> SvdResult<f32> {
        let _context = CurrentContextGuard::activate(self.ctx)?;

        let original_m = a.nrows();
        let original_n = a.ncols();
        if original_m == 0 || original_n == 0 {
            return Err(anyhow!("CUDA SVD requires a non-empty matrix"));
        }

        let transposed = original_m < original_n;
        let solver_m_usize = original_m.max(original_n);
        let solver_n_usize = original_m.min(original_n);
        let m = i32::try_from(solver_m_usize)
            .map_err(|_| anyhow!("matrix row count exceeds cuSOLVER i32 limits"))?;
        let n = i32::try_from(solver_n_usize)
            .map_err(|_| anyhow!("matrix column count exceeds cuSOLVER i32 limits"))?;
        let k = solver_n_usize;

        // gesvd only supports m >= n. For a wide input, decompose A^T and map
        // the singular-vector matrices back to the SVD of A.
        let elements_a = solver_m_usize
            .checked_mul(solver_n_usize)
            .ok_or_else(|| anyhow!("matrix element count overflow"))?;
        let mut a_col = vec![0.0f32; elements_a];
        for r in 0..solver_m_usize {
            for c in 0..solver_n_usize {
                let value = if transposed { a[[c, r]] } else { a[[r, c]] };
                a_col[c * solver_m_usize + r] = value;
            }
        }

        let (solver_u_cols, solver_vt_rows, job) = match mode {
            SvdMode::Full => (solver_m_usize, solver_n_usize, b'A' as i8),
            SvdMode::Reduced => (k, k, b'S' as i8),
        };
        let elements_u = solver_m_usize
            .checked_mul(solver_u_cols)
            .ok_or_else(|| anyhow!("U element count overflow"))?;
        let elements_vt = solver_vt_rows
            .checked_mul(solver_n_usize)
            .ok_or_else(|| anyhow!("Vt element count overflow"))?;

        let d_a = DeviceBuffer::new(checked_bytes::<f32>(elements_a, "A")?, "A")?;
        let d_s = DeviceBuffer::new(checked_bytes::<f32>(k, "singular values")?, "singular values")?;
        let d_u = DeviceBuffer::new(checked_bytes::<f32>(elements_u, "U")?, "U")?;
        let d_vt = DeviceBuffer::new(checked_bytes::<f32>(elements_vt, "Vt")?, "Vt")?;
        d_a.copy_from(&a_col, "A")?;

        let mut lwork = 0;
        let status = unsafe { cusolverDnSgesvd_bufferSize(self.handle, m, n, &mut lwork) };
        if status != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(anyhow!(
                "cusolverDnSgesvd_bufferSize failed with status {status:?}"
            ));
        }
        if lwork <= 0 {
            return Err(anyhow!("cuSOLVER returned invalid workspace size {lwork}"));
        }

        let d_work = DeviceBuffer::new(
            checked_bytes::<f32>(lwork as usize, "cuSOLVER workspace")?,
            "cuSOLVER workspace"
        )?;
        let d_info = DeviceBuffer::new(
            checked_bytes::<i32>(1, "cuSOLVER devInfo")?,
            "cuSOLVER devInfo"
        )?;

        let status = unsafe {
            cusolverDnSgesvd(
                self.handle,
                job,
                job,
                m,
                n,
                d_a.ptr() as *mut f32,
                m,
                d_s.ptr() as *mut f32,
                d_u.ptr() as *mut f32,
                m,
                d_vt.ptr() as *mut f32,
                n,
                d_work.ptr() as *mut f32,
                lwork,
                std::ptr::null_mut(),
                d_info.ptr() as *mut i32,
            )
        };
        if status != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(anyhow!("cusolverDnSgesvd failed with status {status:?}"));
        }

        let mut info = [0i32; 1];
        d_info.copy_to(&mut info, "cuSOLVER devInfo")?;
        match info[0] {
            0 => {}
            value if value < 0 => {
                return Err(anyhow!(
                    "cusolverDnSgesvd rejected argument {}",
                    -value
                ));
            }
            value => {
                return Err(anyhow!(
                    "cusolverDnSgesvd did not converge; {value} superdiagonals did not converge"
                ));
            }
        }

        let mut singular_values = vec![0.0f32; k];
        let mut u_col = vec![0.0f32; elements_u];
        let mut vt_col = vec![0.0f32; elements_vt];
        d_s.copy_to(&mut singular_values, "singular values")?;
        d_u.copy_to(&mut u_col, "U")?;
        d_vt.copy_to(&mut vt_col, "Vt")?;

        let solver_u = Array2::from_shape_fn((solver_m_usize, solver_u_cols), |(r, c)| {
            u_col[c * solver_m_usize + r]
        });
        let solver_vt = Array2::from_shape_fn((solver_vt_rows, solver_n_usize), |(r, c)| {
            vt_col[c * solver_n_usize + r]
        });

        let (u, vt) = if transposed {
            (solver_vt.t().to_owned(), solver_u.t().to_owned())
        } else {
            (solver_u, solver_vt)
        };

        Ok(SvdOutput { u, s: Array1::from_vec(singular_values), vt })
    }
}
