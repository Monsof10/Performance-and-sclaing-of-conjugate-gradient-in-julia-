# =============================================================================
# cg_python.py
# =============================================================================
# Pure Python / NumPy implementation of the Conjugate Gradient (CG) method.
#
# Purpose:
#   - Serve as the baseline reference implementation in Python (NumPy).
#   - Benchmark execution time across several matrix sizes.
#   - Generate a plot of execution time vs. matrix size for comparison
#     against the Julia serial/parallel/GPU implementations.
#
# Usage:
#   python cg_python.py
#
# Output:
#   Console benchmark results + outputs/python_benchmark_results.png
# =============================================================================

import numpy as np       # Numerical arrays and linear algebra operations
import time              # Wall-clock timing for benchmarking
import matplotlib.pyplot as plt  # Plotting benchmark results
import os                # File-system path manipulation


# -----------------------------------------------------------------------------
# Matrix Generation
# -----------------------------------------------------------------------------

def generate_spd_matrix(n):
    """
    Generate a random Symmetric Positive Definite (SPD) matrix of size n x n.

    The Conjugate Gradient method is only guaranteed to converge for SPD
    systems, so every test matrix must satisfy this property.

    Construction steps:
      1. Start with a random matrix A with entries in [0, 1).
      2. Symmetrise: A = 0.5 * (A + A^T)  →  A is now symmetric.
      3. Add a scaled identity: A += n * I  →  guarantees positive definiteness
         because the diagonal dominance shifts all eigenvalues above zero.

    Parameters
    ----------
    n : int
        Dimension of the square matrix.

    Returns
    -------
    A : numpy.ndarray, shape (n, n)
        A symmetric positive definite matrix.
    """
    # Step 1: Random matrix with uniform entries in [0, 1)
    A = np.random.rand(n, n)

    # Step 2: Symmetrise by averaging with its transpose
    A = 0.5 * (A + A.T)

    # Step 3: Add n * I to ensure positive definiteness (diagonal dominance)
    A = A + n * np.eye(n)

    return A


# -----------------------------------------------------------------------------
# Conjugate Gradient Solver
# -----------------------------------------------------------------------------

def cg_python(A, b, tol=1e-6, max_iter=1000):
    """
    Solve the linear system A x = b using the Conjugate Gradient method.

    The CG method is an iterative Krylov-subspace solver that is optimal for
    large sparse (or dense) SPD systems. Each iteration requires only one
    matrix-vector product and a few vector operations.

    Algorithm outline (standard CG):
      Initialise:
        x_0  = 0          (initial guess — all zeros)
        r_0  = b - A x_0  (initial residual)
        p_0  = r_0        (initial search direction)
        ρ_0  = r_0 · r_0  (squared residual norm)

      For i = 0, 1, 2, …, max_iter-1:
        Ap      = A p_i                        (matrix-vector product)
        α       = ρ_i / (p_i · Ap)            (step length)
        x_{i+1} = x_i + α p_i                 (update solution)
        r_{i+1} = r_i − α Ap                  (update residual)
        ρ_{i+1} = r_{i+1} · r_{i+1}
        if ||r_{i+1}|| < tol → converged, stop
        β       = ρ_{i+1} / ρ_i               (Fletcher-Reeves coefficient)
        p_{i+1} = r_{i+1} + β p_i             (update search direction)

    Parameters
    ----------
    A        : numpy.ndarray, shape (n, n)
               Symmetric positive definite coefficient matrix.
    b        : numpy.ndarray, shape (n,)
               Right-hand side vector.
    tol      : float, optional (default 1e-6)
               Convergence tolerance on the residual 2-norm ||r||.
    max_iter : int, optional (default 1000)
               Maximum number of CG iterations before giving up.

    Returns
    -------
    x    : numpy.ndarray, shape (n,)
           Approximate solution vector.
    iters : int
           Number of iterations performed (≤ max_iter).
    res  : float
           Final residual norm ||r|| at exit.
    """
    n = len(b)                     # Problem size (number of unknowns)
    x = np.zeros(n)                # Initial guess x_0 = 0
    r = b - A.dot(x)               # Initial residual r_0 = b - A x_0
    p = r.copy()                   # Initial search direction p_0 = r_0
    rsold = np.dot(r, r)           # ρ_0 = r_0 · r_0 (squared residual norm)

    # -------------------------------------------------------------------------
    # Main CG iteration loop
    # -------------------------------------------------------------------------
    for i in range(max_iter):

        # Compute matrix-vector product: Ap = A * p_i
        Ap = A.dot(p)

        # Step length: α = ρ_i / (p_i · A p_i)
        alpha = rsold / np.dot(p, Ap)

        # Update solution: x_{i+1} = x_i + α p_i
        x = x + alpha * p

        # Update residual: r_{i+1} = r_i − α A p_i
        r = r - alpha * Ap

        # New squared residual norm: ρ_{i+1} = r_{i+1} · r_{i+1}
        rsnew = np.dot(r, r)

        # Convergence check: stop if ||r_{i+1}|| < tol
        if np.sqrt(rsnew) < tol:
            # Return solution, iteration count (1-indexed), and residual norm
            return x, i + 1, np.sqrt(rsnew)

        # Fletcher-Reeves update for the search direction:
        #   p_{i+1} = r_{i+1} + (ρ_{i+1} / ρ_i) * p_i
        p = r + (rsnew / rsold) * p

        # Carry ρ_{i+1} forward for the next iteration
        rsold = rsnew

    # If we reach here, max_iter was exceeded without converging.
    # Return the best solution found so far together with the current residual.
    return x, max_iter, np.sqrt(rsold)


# =============================================================================
# Benchmark Entry Point
# =============================================================================

if __name__ == "__main__":

    # Matrix sizes to benchmark (n × n systems)
    sizes = [1000, 2000, 4000, 5000]

    # List to collect wall-clock times for each size
    times = []

    print("=== Python Conjugate Gradient Benchmark ===")

    # ------------------------------------------------------------------
    # Benchmark loop — iterate over each problem size
    # ------------------------------------------------------------------
    for n in sizes:
        # Build a fresh SPD matrix and random right-hand side for each size
        A = generate_spd_matrix(n)
        b = np.random.rand(n)

        # Warm-up run (2 iterations only) to avoid measuring JIT / cache effects
        # on the very first call.  NumPy does not JIT-compile, but this keeps
        # the methodology consistent with the Julia benchmarks.
        cg_python(A, b, max_iter=2)

        # --- Timed run ---
        start_time = time.time()
        x, iters, res = cg_python(A, b)        # Full solve
        end_time = time.time()

        # Compute and store elapsed wall-clock time in seconds
        exec_time = end_time - start_time
        times.append(exec_time)

        # Print a one-line summary for this size
        print(f"Size: {n}x{n} | Time: {exec_time:.4f} s | Iters: {iters} | Res: {res:.2e}")

    # ------------------------------------------------------------------
    # Plotting — execution time vs. matrix size
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    # Plot the collected timing data as a line chart with circle markers
    plt.plot(sizes, times, marker='o', linestyle='-', color='r', label='Python (NumPy)')

    plt.title("Python CG Benchmark Execution Time")
    plt.xlabel("Matrix Size (n)")
    plt.ylabel("Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.6)   # Dashed grid for readability
    plt.legend()

    # ------------------------------------------------------------------
    # Save the plot to the shared outputs/ directory
    # ------------------------------------------------------------------
    # Build path: <project_root>/outputs/  (two levels up from this file)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")

    # Create the directory if it does not already exist
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, "python_benchmark_results.png")
    plt.savefig(plot_path)
    print(f"\nPlot saved to '{plot_path}'")
