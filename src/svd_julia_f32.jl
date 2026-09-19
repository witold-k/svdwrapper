# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Witold Kaminski

using LinearAlgebra

function svd_cpu_f32(a::Array{Float32,2}, full::Bool)
    f = svd(a; full=full)
    return (f.U, f.S, f.Vt)
end
