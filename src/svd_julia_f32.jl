# call julia and execute first
# using Pkg
# Pkg.add("CUDA")

#using CUDA
using LinearAlgebra

function svd_cuda_f32(a::Array{Float32,2})
    dA = CuArray(a)              # move to GPU
    F  = svd(dA)                 # SVD on GPU (CUSOLVER under the hood)
    # bring results back to CPU as regular Arrays
    return (Array(F.U), Array(F.S), Array(F.Vt))
end

function svd_cpu_f32(a::Array{Float32,2})
    dA = a 
    F  = svd(dA)                 
    # bring results back to CPU as regular Arrays
    return (Array(F.U), Array(F.S), Array(F.Vt))
end

