// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Witold Kaminski

use crate::svd::SvdBackend;
use anyhow::{anyhow, Result};
use cudarc::cusolver::sys::*;
use cudarc::driver::sys::*;
use ndarray::{Array2, ArrayBase, Data, Ix2};

pub struct CudaSvd {
    handle: cusolverDnHandle_t,
    #[allow(unused)]
    ctx: CUcontext,
}

impl CudaSvd {
    pub fn new() -> Result<Self> {
        unsafe {
            // Initialize CUDA
            cuInit(0);

            // Get device 0
            let mut dev: CUdevice = 0;
            let res = cuDeviceGet(&mut dev, 0);
            if res != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuDeviceGet failed: {:?}", res));
            }

            // Retain the primary context (modern API)
            let mut ctx: CUcontext = std::ptr::null_mut();
            let res = cuDevicePrimaryCtxRetain(&mut ctx, dev);
            if res != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuDevicePrimaryCtxRetain failed: {:?}", res));
            }

            // Make it current
            let res = cuCtxSetCurrent(ctx);
            if res != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuCtxSetCurrent failed: {:?}", res));
            }

            // Create cuSOLVER handle
            let mut handle: cusolverDnHandle_t = std::ptr::null_mut();
            let res = cusolverDnCreate(&mut handle);
            if res != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                cuDevicePrimaryCtxRelease_v2(dev);
                return Err(anyhow!("cusolverDnCreate failed: {:?}", res));
            }

            Ok(Self { handle, ctx })
        }
    }
}

impl Drop for CudaSvd {
    fn drop(&mut self) {
        unsafe {
            cusolverDnDestroy(self.handle);

            // Release primary context
            let mut dev: CUdevice = 0;
            cuCtxGetDevice(&mut dev);
            cuDevicePrimaryCtxRelease_v2(dev);
        }
    }
}

impl SvdBackend for CudaSvd {
    fn compute_svd(
        &self,
        a: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), anyhow::Error> {
        println!("CUDA");
        let (m, n) = (a.nrows() as i32, a.ncols() as i32);
        let lda = m;

        let a_slice = a
            .as_slice()
            .ok_or_else(|| anyhow!("Matrix A must be contiguous"))?;

        let elem_size = std::mem::size_of::<f64>();
        let bytes_a = m as usize * n as usize * elem_size;

        unsafe {
            // Device pointers
            let mut d_a: CUdeviceptr = 0;
            let mut d_s: CUdeviceptr = 0;
            let mut d_u: CUdeviceptr = 0;
            let mut d_vt: CUdeviceptr = 0;
            let mut d_work: CUdeviceptr = 0;
            let mut d_info: CUdeviceptr = 0;

            // Allocate A
            let res = cuMemAlloc_v2(&mut d_a, bytes_a);
            if res != CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuMemAlloc_v2(d_a) failed: {:?}", res));
            }

            // Copy A to device
            let res = cuMemcpyHtoD_v2(d_a, a_slice.as_ptr() as *const _, bytes_a);
            if res != CUresult::CUDA_SUCCESS {
                cuMemFree_v2(d_a);
                return Err(anyhow!("cuMemcpyHtoD_v2(d_a) failed: {:?}", res));
            }

            // Allocate S, U, VT
            let bytes_s = n as usize * elem_size;
            let bytes_u = m as usize * m as usize * elem_size;
            let bytes_vt = n as usize * n as usize * elem_size;

            cuMemAlloc_v2(&mut d_s, bytes_s);
            cuMemAlloc_v2(&mut d_u, bytes_u);
            cuMemAlloc_v2(&mut d_vt, bytes_vt);

            // --- gesvdj info/params ---
            let mut gesvdj_info: gesvdjInfo_t = std::ptr::null_mut();
            let stat = cusolverDnCreateGesvdjInfo(&mut gesvdj_info);
            if stat != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                cuMemFree_v2(d_a);
                cuMemFree_v2(d_s);
                cuMemFree_v2(d_u);
                cuMemFree_v2(d_vt);
                return Err(anyhow!("cusolverDnCreateGesvdjInfo failed: {:?}", stat));
            }

            // optional: set tolerance / max sweeps if you want
            // cusolverDnXgesvdjSetTolerance(gesvdj_info, 1e-7);
            // cusolverDnXgesvdjSetMaxSweeps(gesvdj_info, 100);

            // Workspace size for Jacobi SVD (full U, VT)
            let jobz = cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;

            let econ = 0;          // 0 = full, 1 = econ

            let mut lwork: i32 = 0;
            let stat = cusolverDnDgesvdj_bufferSize(
                self.handle,
                jobz,
                econ,
                m,
                n,
                d_a as *mut f64,
                lda,
                d_s as *mut f64,
                d_u as *mut f64,
                m,
                d_vt as *mut f64,
                n,
                &mut lwork,
                gesvdj_info,
            );
            if stat != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                cuMemFree_v2(d_a);
                cuMemFree_v2(d_s);
                cuMemFree_v2(d_u);
                cuMemFree_v2(d_vt);
                cusolverDnDestroyGesvdjInfo(gesvdj_info);
                return Err(anyhow!(
                    "cusolverDnDgesvdj_bufferSize failed: {:?}",
                    stat
                ));
            }

            let bytes_work = lwork as usize * elem_size;
            cuMemAlloc_v2(&mut d_work, bytes_work);
            cuMemAlloc_v2(&mut d_info, std::mem::size_of::<i32>());

            // --- SVD (Jacobi) ---
            let stat = cusolverDnDgesvdj(
                self.handle,
                jobz,
                econ,
                m,
                n,
                d_a as *mut f64,
                lda,
                d_s as *mut f64,
                d_u as *mut f64,
                m,
                d_vt as *mut f64,
                n,
                d_work as *mut f64,
                lwork,
                d_info as *mut i32,
                gesvdj_info,
            );

            if stat != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                cuMemFree_v2(d_a);
                cuMemFree_v2(d_s);
                cuMemFree_v2(d_u);
                cuMemFree_v2(d_vt);
                cuMemFree_v2(d_work);
                cuMemFree_v2(d_info);
                cusolverDnDestroyGesvdjInfo(gesvdj_info);
                return Err(anyhow!("cusolverDnDgesvdj failed: {:?}", stat));
            }

            // Copy back
            let mut s_vec = vec![0f64; n as usize];
            let mut u_vec = vec![0f64; (m * m) as usize];
            let mut vt_vec = vec![0f64; (n * n) as usize];

            cuMemcpyDtoH_v2(
                s_vec.as_mut_ptr() as *mut _,
                d_s,
                bytes_s,
            );
            cuMemcpyDtoH_v2(
                u_vec.as_mut_ptr() as *mut _,
                d_u,
                bytes_u,
            );
            cuMemcpyDtoH_v2(
                vt_vec.as_mut_ptr() as *mut _,
                d_vt,
                bytes_vt,
            );

            // Free device memory
            cuMemFree_v2(d_a);
            cuMemFree_v2(d_s);
            cuMemFree_v2(d_u);
            cuMemFree_v2(d_vt);
            cuMemFree_v2(d_work);
            cuMemFree_v2(d_info);
            cusolverDnDestroyGesvdjInfo(gesvdj_info);

            // Build ndarray
            let s = Array2::from_shape_vec((n as usize, 1), s_vec)?;
            let u = Array2::from_shape_vec((m as usize, m as usize), u_vec)?;
            let vt = Array2::from_shape_vec((n as usize, n as usize), vt_vec)?;

            Ok((u, s, vt))
        }
    }
}

