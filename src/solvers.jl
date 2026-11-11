# =============================================================================
# src/solvers.jl
# =============================================================================
# Defines the `Solvers` module, which provides three implementations of the
# Conjugate Gradient (CG) iterative solver for symmetric positive definite
# (SPD) linear systems A x = b:
#
#   1. cg_serial   — single-threaded reference implementation
#   2. cg_parallel — multi-threaded CPU implementation (Threads.@threads)
#   3. cg_gpu      — GPU implementation via CUDA.jl
#
# All three solvers share the same mathematical algorithm (standard CG with
# Fletcher-Reeves direction update) and expose an identical interface so that
# they can be swapped transparently for benchmarking.
# =============================================================================

module Solvers

using LinearAlgebra  # dot, norm, etc.
using CUDA           # GPU arrays (CuArray) and cu() transfer helper

# Public API — only these symbols are visible outside the module
export cg_serial, cg_parallel, cg_gpu


# -----------------------------------------------------------------------------
# 1. Serial CG Solver
# -----------------------------------------------------------------------------

"""
    cg_serial(A, b; tol=1e-6, max_iter=1000)

Solve the SPD linear system `A x = b` using the standard single-threaded
Conjugate Gradient method.

### Algorithm (standard CG)
```
Initialise:
  x₀   = 0            (zero initial guess)
  r₀   = b − A x₀     (initial residual)
  p₀   = r₀            (initial search direction)
  ρ₀   = r₀ · r₀       (squared residual norm)

For i = 1, 2, …, max_iter:
  Ap      = A pᵢ                   (matrix–vector product — the expensive step)
  α       = ρᵢ / (pᵢ · Ap)        (optimal step length along pᵢ)
  x_{i+1} = xᵢ + α pᵢ            (advance solution)
  r_{i+1} = rᵢ − α Ap             (update residual without re-multiplying A)
  ρ_{i+1} = r_{i+1} · r_{i+1}
  if ‖r_{i+1}‖ < tol → converged
  β       = ρ_{i+1} / ρᵢ          (Fletcher-Reeves coefficient)
  p_{i+1} = r_{i+1} + β pᵢ        (new A-conjugate search direction)
```

### Parameters
- `A`        : SPD coefficient matrix (n × n)
- `b`        : right-hand side vector (length n)
- `tol`      : convergence tolerance on ‖r‖₂  (default 1e-6)
- `max_iter` : maximum number of iterations    (default 1000)

### Returns
`(x, iterations, residual_norm)`
"""
function cg_serial(A, b; tol=1e-6, max_iter=1000)
    n = length(b)                      # Problem dimension

    # Initial guess: x₀ = 0 (matches element type of b for generality)
    x = zeros(eltype(b), n)

    # Initial residual: r₀ = b − A x₀ = b  (since x₀ = 0)
    r = b - A * x

    # Initial search direction equals the residual
    p = copy(r)

    # ρ₀ = r₀ · r₀  (squared ℓ₂-norm of residual)
    rsold = dot(r, r)

    # ------------------------------------------------------------------
    # Main CG iteration
    # ------------------------------------------------------------------
    for i in 1:max_iter
        # Matrix–vector product: Ap = A * pᵢ
        Ap = A * p

        # Step length: α = ρᵢ / (pᵢᵀ A pᵢ)
        alpha = rsold / dot(p, Ap)

        # Update solution:  x_{i+1} = xᵢ + α pᵢ
        x = x + alpha * p

        # Update residual:  r_{i+1} = rᵢ − α A pᵢ
        r = r - alpha * Ap

        # New squared residual norm: ρ_{i+1} = r_{i+1} · r_{i+1}
        rsnew = dot(r, r)

        # Convergence check: ‖r‖₂ = √ρ < tol
        if sqrt(rsnew) < tol
            return x, i, sqrt(rsnew)   # Return early on convergence
        end

        # Fletcher-Reeves direction update:
        #   p_{i+1} = r_{i+1} + (ρ_{i+1} / ρᵢ) pᵢ
        p = r + (rsnew / rsold) * p

        # Shift ρ forward for the next iteration
        rsold = rsnew
    end

    # Reached max_iter without converging — return best result
    return x, max_iter, sqrt(rsold)
end


# -----------------------------------------------------------------------------
# 2. Multi-Threaded (Parallel CPU) CG Solver
# -----------------------------------------------------------------------------

"""
    cg_parallel(A, b; tol=1e-6, max_iter=1000)

Multi-threaded CPU implementation of the Conjugate Gradient algorithm.

The only difference from `cg_serial` is that the matrix–vector product
`A * p` is parallelised across rows using `Threads.@threads` (see the helper
`threaded_matvec!`).  All vector arithmetic uses Julia's in-place broadcasting
(`.+=`, `.-=`, `.=`) which avoids extra heap allocations every iteration.

Start Julia with multiple threads (e.g. `julia -t 4 main.jl`) to benefit
from this implementation.

### Parameters / Returns — identical to `cg_serial`.
"""
function cg_parallel(A, b; tol=1e-6, max_iter=1000)
    n = length(b)
    x = zeros(eltype(b), n)    # Zero initial guess

    # Pre-allocate the Ap buffer once and reuse it every iteration.
    # This avoids triggering the garbage collector inside the hot loop.
    Ap = similar(b)

    # Initial residual and search direction
    r = b - A * x
    p = copy(r)
    rsold = dot(r, r)

    # ------------------------------------------------------------------
    # Main CG iteration (parallel matrix–vector product)
    # ------------------------------------------------------------------
    for i in 1:max_iter

        # Parallelised matrix–vector multiply: Ap = A * p
        # Rows of A are distributed across available CPU threads.
        threaded_matvec!(Ap, A, p)

        # Step length (scalar, computed on a single thread)
        alpha = rsold / dot(p, Ap)

        # In-place solution / residual updates using Julia broadcasting.
        # Broadcasting over arrays is already optimised by the compiler
        # (loop fusion), so no extra allocations occur here.
        x .+= alpha .* p       # x_{i+1} = xᵢ + α pᵢ
        r .-= alpha .* Ap      # r_{i+1} = rᵢ − α Ap

        rsnew = dot(r, r)

        # Convergence check
        if sqrt(rsnew) < tol
            return x, i, sqrt(rsnew)
        end

        # In-place direction update: p_{i+1} = r_{i+1} + β pᵢ
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end

    return x, max_iter, sqrt(rsold)
end


"""
    threaded_matvec!(y, A, x)

Compute the matrix–vector product `y = A * x` in-place using multi-threading.

Each thread handles a contiguous subset of the rows of `A`, accumulating the
dot product for that row into `y[i]`.

- `@inbounds` suppresses bounds-checking inside the inner loop for speed.
- `Threads.@threads` distributes the outer loop across all available threads.

### Parameters
- `y` : output vector (pre-allocated, length = rows of A)
- `A` : coefficient matrix (m × n)
- `x` : input vector (length n)
"""
function threaded_matvec!(y, A, x)
    # Distribute rows of A across Julia threads
    Threads.@threads for i in 1:size(A, 1)
        s = zero(eltype(A))              # Thread-local accumulator
        @inbounds for j in 1:size(A, 2) # Skip bounds check for performance
            s += A[i, j] * x[j]
        end
        y[i] = s                         # Write result to shared output vector
    end
end


# -----------------------------------------------------------------------------
# 3. GPU CG Solver (CUDA)
# -----------------------------------------------------------------------------

"""
    cg_gpu(A, b; tol=1e-6, max_iter=1000)

GPU-accelerated implementation of the Conjugate Gradient algorithm via
CUDA.jl.

All heavy computation (matrix–vector products, vector updates) is offloaded to
the GPU.  The convergence check (`sqrt(rsnew) < tol`) requires a scalar value
on the CPU, so `rsnew` is implicitly transferred from device to host each
iteration — this is the main performance trade-off.

The solution vector is transferred back to CPU memory (`Array(x)`) before
returning so the caller does not need to manage device memory.

### Parameters / Returns — identical to `cg_serial`.
"""
function cg_gpu(A, b; tol=1e-6, max_iter=1000)
    # Transfer input matrix and vector from host (CPU) RAM to device (GPU) VRAM
    d_A = cu(A)   # CuArray — lives on the GPU
    d_b = cu(b)   # CuArray — lives on the GPU

    n = length(d_b)

    # Allocate initial solution vector on the GPU (all zeros)
    x = CUDA.zeros(eltype(d_b), n)

    # Initial residual and search direction (all on GPU)
    r = d_b - d_A * x
    p = copy(r)
    rsold = dot(r, r)   # GPU reduction — returns a scalar on the CPU

    # ------------------------------------------------------------------
    # Main CG iteration (all arithmetic on GPU)
    # ------------------------------------------------------------------
    for i in 1:max_iter

        # GPU matrix–vector product: Ap = d_A * p
        Ap = d_A * p

        # Step length (scalar transferred to CPU automatically by dot())
        alpha = rsold / dot(p, Ap)

        # In-place solution and residual updates on the GPU
        x .+= alpha .* p    # x_{i+1} = xᵢ + α pᵢ   (GPU kernel)
        r .-= alpha .* Ap   # r_{i+1} = rᵢ − α Ap    (GPU kernel)

        # New squared residual norm (GPU reduction, scalar on CPU)
        rsnew = dot(r, r)

        # Convergence check on CPU (rsnew was already transferred by dot())
        if sqrt(rsnew) < tol
            # Transfer solution from GPU back to CPU RAM before returning
            return Array(x), i, sqrt(rsnew)
        end

        # Direction update on GPU: p_{i+1} = r_{i+1} + β pᵢ
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end

    # Transfer solution back to CPU RAM
    return Array(x), max_iter, sqrt(rsold)
end

end # module Solvers
