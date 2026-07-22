// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

use crate::svd::{SvdBackend, SvdResult};
use anyhow::{anyhow, Result};
use cudarc::cusolver::sys::*;
use cudarc::driver::sys::*;
use ndarray::{Array2, ArrayBase, Data, Ix2};

pub struct CudaF64Svd {
    handle: cusolverDnHandle_t,
    #[allow(unused)]
    ctx: CUcontext,
}

impl CudaF64Svd {
    pub fn new() -> Result<Self> {
        unsafe {
            cuInit(0);
            let mut dev: CUdevice = 0;
            if cuDeviceGet(&mut dev, 0) != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuDeviceGet failed"));
            }

            let mut ctx: CUcontext = std::ptr::null_mut();
            if cuDevicePrimaryCtxRetain(&mut ctx, dev) != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuDevicePrimaryCtxRetain failed"));
            }

            if cuCtxSetCurrent(ctx) != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuCtxSetCurrent failed"));
            }

            let mut handle: cusolverDnHandle_t = std::ptr::null_mut();
            if cusolverDnCreate(&mut handle) != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                cuDevicePrimaryCtxRelease_v2(dev);
                return Err(anyhow!("cusolverDnCreate failed"));
            }

            Ok(Self { handle, ctx })
        }
    }
}

impl Drop for CudaF64Svd {
    fn drop(&mut self) {
        unsafe {
            cusolverDnDestroy(self.handle);
            let mut dev: CUdevice = 0;
            cuCtxGetDevice(&mut dev);
            cuDevicePrimaryCtxRelease_v2(dev);
        }
    }
}

impl SvdBackend<f64> for CudaF64Svd {
    fn compute_svd(
        &self,
        a: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> SvdResult<f64> {
        let (m, n) = (a.nrows() as i32, a.ncols() as i32);
        let k = std::cmp::min(m, n);
        let lda = m;

        // FIX 1 & 2: Konvertierung in Column-Major Layout fängt auch fragmentierte Slices ab
        let mut a_col = vec![0.0f64; (m * n) as usize];
        for r in 0..a.nrows() {
            for c in 0..a.ncols() {
                a_col[c * (m as usize) + r] = a[[r, c]];
            }
        }

        let elem_size = std::mem::size_of::<f64>();
        let bytes_a = m as usize * n as usize * elem_size;
        let bytes_s = k as usize * elem_size; // FIX 3: k statt n nutzen
        let bytes_u = m as usize * m as usize * elem_size;
        let bytes_vt = n as usize * n as usize * elem_size;

        unsafe {
            let mut d_a: CUdeviceptr = 0;
            let mut d_s: CUdeviceptr = 0;
            let mut d_u: CUdeviceptr = 0;
            let mut d_vt: CUdeviceptr = 0;
            let mut d_work: CUdeviceptr = 0;
            let mut d_info: CUdeviceptr = 0;

            // FIX 4: Sequentieller Check verhindert Teilleaks auf der GPU
            if cuMemAlloc_v2(&mut d_a, bytes_a) != CUresult::CUDA_SUCCESS ||
               cuMemcpyHtoD_v2(d_a, a_col.as_ptr() as *const _, bytes_a) != CUresult::CUDA_SUCCESS ||
               cuMemAlloc_v2(&mut d_s, bytes_s) != CUresult::CUDA_SUCCESS ||
               cuMemAlloc_v2(&mut d_u, bytes_u) != CUresult::CUDA_SUCCESS ||
               cuMemAlloc_v2(&mut d_vt, bytes_vt) != CUresult::CUDA_SUCCESS {
                cuMemFree_v2(d_a); cuMemFree_v2(d_s); cuMemFree_v2(d_u); cuMemFree_v2(d_vt);
                return Err(anyhow!("GPU Allocation or Copy failed"));
            }

            let mut lwork: i32 = 0;
            let stat = cusolverDnDgesvd_bufferSize(self.handle, m, n, &mut lwork);
            if stat != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                cuMemFree_v2(d_a); cuMemFree_v2(d_s); cuMemFree_v2(d_u); cuMemFree_v2(d_vt);
                return Err(anyhow!("cusolverDnDgesvd_bufferSize failed: {:?}", stat));
            }

            let bytes_work = lwork as usize * elem_size;
            cuMemAlloc_v2(&mut d_work, bytes_work);
            cuMemAlloc_v2(&mut d_info, std::mem::size_of::<i32>());

            let stat = cusolverDnDgesvd(
                self.handle, b'A' as i8, b'A' as i8, m, n,
                d_a as *mut f64, lda, d_s as *mut f64, d_u as *mut f64, m,
                d_vt as *mut f64, n, d_work as *mut f64, lwork,
                std::ptr::null_mut(), d_info as *mut i32,
            );

            if stat != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                cuMemFree_v2(d_a); cuMemFree_v2(d_s); cuMemFree_v2(d_u); cuMemFree_v2(d_vt);
                cuMemFree_v2(d_work); cuMemFree_v2(d_info);
                return Err(anyhow!("cusolverDnDgesvd failed: {:?}", stat));
            }

            let mut s_vec = vec![0f64; k as usize];
            let mut u_col_res = vec![0f64; (m * m) as usize];
            let mut vt_col_res = vec![0f64; (n * n) as usize];

            cuMemcpyDtoH_v2(s_vec.as_mut_ptr() as *mut _, d_s, bytes_s);
            cuMemcpyDtoH_v2(u_col_res.as_mut_ptr() as *mut _, d_u, bytes_u);
            cuMemcpyDtoH_v2(vt_col_res.as_mut_ptr() as *mut _, d_vt, bytes_vt);

            cuMemFree_v2(d_a); cuMemFree_v2(d_s); cuMemFree_v2(d_u); cuMemFree_v2(d_vt);
            cuMemFree_v2(d_work); cuMemFree_v2(d_info);

            // Re-Konvertierung von Column-Major zu Row-Major ndarray
            let mut u = Array2::zeros((m as usize, m as usize));
            for r in 0..(m as usize) {
                for c in 0..(m as usize) {
                    u[[r, c]] = u_col_res[c * (m as usize) + r];
                }
            }

            let mut vt = Array2::zeros((n as usize, n as usize));
            for r in 0..(n as usize) {
                for c in 0..(n as usize) {
                    vt[[r, c]] = vt_col_res[c * (n as usize) + r];
                }
            }

            let mut sigma = Array2::zeros((m as usize, n as usize));
            for i in 0..(k as usize) {
                sigma[[i, i]] = s_vec[i];
            }

            Ok((u, sigma, vt))
        }
    }
}

