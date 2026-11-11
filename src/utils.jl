# =============================================================================
# src/utils.jl
# =============================================================================
# Defines the `Utils` module, which provides helper functions used across the
# project:
#
#   - generate_spd_matrix : constructs a random Symmetric Positive Definite
#                           (SPD) matrix suitable for CG testing.
#   - validate_solution   : computes the residual norm to confirm solver
#                           accuracy after a solve.
# =============================================================================

module Utils

using LinearAlgebra   # norm(), I (identity), etc.
using SparseArrays    # imported for potential sparse-matrix extensions

# Symbols exported to callers (e.g. `using .Utils`)
export generate_spd_matrix, validate_solution


# -----------------------------------------------------------------------------
# Matrix Generation
# -----------------------------------------------------------------------------

"""
    generate_spd_matrix(n)

Generate a random Symmetric Positive Definite (SPD) matrix of size `n × n`.

The Conjugate Gradient method is mathematically guaranteed to converge only
for SPD systems, so every test case must use an SPD matrix.

### Construction strategy
1. Draw `A` from a standard normal distribution (`randn`).
   - Using `randn` (Gaussian) creates a harder, more realistic test than
     uniform random entries.
2. Form `A' * A` — the product of any matrix with its transpose is always
   symmetric positive semi-definite (SPSD).
3. Divide by `n` to prevent entries from growing with problem size.
4. Add `0.01 * I` (a small identity shift) to push all eigenvalues strictly
   above zero, ensuring *positive definiteness* (not just semi-definiteness).

### Parameters
- `n` : dimension of the desired square matrix.

### Returns
An `n × n` dense `Matrix{Float64}` that is symmetric and positive definite.
"""
function generate_spd_matrix(n)
    # Step 1: Draw a random n×n matrix from a standard normal distribution.
    # randn gives harder conditioning than rand (heavier tails, more variety).
    A = randn(n, n)

    # Step 2: A' * A  →  symmetric positive semi-definite.
    # Step 3: divide by n  →  scales entries to O(1) regardless of n.
    # Step 4: add 0.01 * I →  shifts smallest eigenvalue above 0 (strictly SPD).
    A = (A' * A) ./ n + 0.01 * I

    return A
end


# -----------------------------------------------------------------------------
# Solution Validation
# -----------------------------------------------------------------------------

"""
    validate_solution(A, b, x; tol=1e-5)

Validate an approximate solution `x` to the linear system `A x = b` by
computing the 2-norm of the residual vector `r = b − A x`.

A solver is considered successful when `‖r‖₂ ≤ tol`.  This function provides
an independent verification step that is separate from the solver's own
internal convergence tracking.

### Parameters
- `A` : coefficient matrix (n × n), same matrix passed to the solver.
- `b` : right-hand side vector (length n).
- `x` : approximate solution returned by the solver.

### Returns
`norm_res :: Float64` — the Euclidean norm ‖b − A x‖₂.
A small value (close to zero) confirms a good solution.
"""
function validate_solution(A, b, x)
    # Compute the residual vector r = b − A x
    residual = b - A * x

    # Return the Euclidean (ℓ₂) norm of the residual
    norm_res = norm(residual)
    return norm_res
end

end # module Utils
