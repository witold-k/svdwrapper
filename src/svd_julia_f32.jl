# call julia and execute first
# using Pkg
# Pkg.add("CUDA")

# file content as a string in Rust
#using CUDA
using LinearAlgebra

function svd_cuda_f32(a::Array{Float32,2})
    dA = a # CuArray(a)              # move to GPU
    F  = svd(dA)                 # SVD on GPU (CUSOLVER under the hood)
    # bring results back to CPU as regular Arrays
    return (Array(F.U), Array(F.S), Array(F.Vt))
end

function svd_cpu_f32(a::Array{Float32,2})
    dA = a # CuArray(a)              # move to GPU
    F  = svd(dA)                 # SVD on GPU (CUSOLVER under the hood)
    # bring results back to CPU as regular Arrays
    return (Array(F.U), Array(F.S), Array(F.Vt))
end

