# doc

- human rewrite
- only for dense SVD

# bugs

- only english language

# optimizations

- Singular values must not be stored in a matrix, instead vector

# features

- provide julia wrapper
- provide AMD rocm wrapper
- provide opencl wrapper


# missing tests

UᵀU ≈ I
V Vᵀ ≈ I
A ≈ U diag(S) Vᵀ
S descending
S >= 0

m > n
m < n
m = n
rank deficient
zero matrix
identity
ill-conditioned
non-contiguous ndarray slice
