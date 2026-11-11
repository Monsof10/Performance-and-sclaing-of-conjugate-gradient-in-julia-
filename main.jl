# =============================================================================
# main.jl  —  Project Entry Point
# =============================================================================
# Orchestrates the full Conjugate Gradient (CG) performance study:
#
#   Phase 1 — Dependency management
#             Activates the local Pkg environment and installs any missing
#             packages listed in REQUIRED_PACKAGES.
#
#   Phase 2 — Solver validation
#             Runs all three CG implementations (serial, parallel, GPU) on a
#             1000 × 1000 test system to verify correctness before benchmarking.
#
#   Phase 3 — Performance benchmarking
#             Calls run_benchmarks() with a range of matrix sizes and collects
#             wall-clock timings for each solver variant.
#
#   Phase 4 — Visualisation
#             Produces a two-panel plot (execution time + speedup) saved to
#             outputs/cg_benchmark_results.png.
#
# Run from the project root:
#   julia main.jl
# or with multiple CPU threads for the parallel solver:
#   julia -t 4 main.jl
# =============================================================================

using Pkg

# Activate the project-local environment (./env/) so that all package
# versions are pinned to those recorded in env/Project.toml / Manifest.toml.
Pkg.activate("env")

# List of external packages that must be present in the environment.
# These are not Julia standard-library packages — they need to be added via Pkg.
const REQUIRED_PACKAGES = ["CUDA", "BenchmarkTools", "Plots"]

println("Checking/Installing dependencies... (This might take a few minutes on the first run)")

# ─────────────────────────────────────────────────────────────────────────────
# Auto-install any missing external packages
# ─────────────────────────────────────────────────────────────────────────────
# Pkg.project().dependencies is a Dict mapping package name → UUID for all
# packages currently declared in the active Project.toml.  We only call
# Pkg.add if the package is absent, to avoid re-resolving on every run.
for pkg in REQUIRED_PACKAGES
    if !haskey(Pkg.project().dependencies, pkg)
        println("Adding package: $pkg")
        Pkg.add(pkg)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Ensure standard library packages are recorded in the environment
# ─────────────────────────────────────────────────────────────────────────────
# LinearAlgebra, SparseArrays, and Statistics ship with Julia itself, so
# Pkg.add may fail (they are not registered in General).  The try/catch
# silently skips the failure — the packages will still be importable from
# Julia's base installation.
for pkg in ["LinearAlgebra", "SparseArrays", "Statistics"]
    if !haskey(Pkg.project().dependencies, pkg)
        try
            Pkg.add(pkg)
        catch
            # If Pkg.add fails for a stdlib, it's likely already available in base
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Load project source files and external packages
# ─────────────────────────────────────────────────────────────────────────────
include("src/solvers.jl")      # Defines module Solvers (cg_serial, cg_parallel, cg_gpu)
include("src/utils.jl")        # Defines module Utils  (generate_spd_matrix, validate_solution)
include("scripts/benchmarks.jl")  # Defines run_benchmarks()

using .Solvers   # Bring solver functions into scope
using .Utils     # Bring utility functions into scope
using Plots      # Plotting library (GR backend by default)
using CUDA       # GPU support; functions gracefully degrade when unavailable

# ─────────────────────────────────────────────────────────────────────────────
# CUDA runtime version compatibility fix
# ─────────────────────────────────────────────────────────────────────────────
# On some systems the installed CUDA.jl version may mismatch the system's CUDA
# runtime.  CUDA.set_runtime_version! instructs CUDA.jl to target a specific
# runtime version, which can resolve "version mismatch" errors at startup.
# We wrap it in a try/catch because:
#   (a) The function may not exist in older CUDA.jl versions.
#   (b) It is only meaningful when CUDA is actually functional.
if CUDA.functional()
    try
        # This tells CUDA.jl to use the system runtime if possible
        CUDA.set_runtime_version!(v"13.1")
    catch
        # Ignore if this specific function isn't available
    end
end


# =============================================================================
# main() — Top-level orchestration function
# =============================================================================

function main()
    println("\n=== Conjugate Gradient Project ===")

    # ──────────────────────────────────────────────────────────────────────
    # Phase 2 — Solver Validation
    # ──────────────────────────────────────────────────────────────────────
    # Verify that all solver implementations produce converged solutions
    # on a moderately sized system before running expensive benchmarks.
    println("\n[Phase 2] Validating implementations on a 1000x1000 system...")
    n_test = 1000

    # Build a 1000×1000 SPD matrix and a random right-hand side
    A = generate_spd_matrix(n_test)
    b = rand(n_test)

    # Run each solver; tol=1e-6 is the convergence threshold on ‖r‖₂.
    # Using a slightly larger tolerance for randn matrices (harder conditioning)
    x_ser, iter_ser, res_ser = cg_serial(A, b, tol=1e-6)
    x_par, iter_par, res_par = cg_parallel(A, b, tol=1e-6)

    # Print validation summary: iteration count and final residual norm
    println("Serial:   Iterations=$iter_ser, Residual=$res_ser")
    println("Parallel: Iterations=$iter_par, Residual=$res_par")

    # GPU validation — only runs if CUDA is available and functional
    if CUDA.functional()
        x_gpu, iter_gpu, res_gpu = cg_gpu(A, b, tol=1e-6)
        println("GPU:      Iterations=$iter_gpu, Residual=$res_gpu")
    else
        println("GPU:      Not functional or not found. Skipping.")
    end

    # ──────────────────────────────────────────────────────────────────────
    # Phase 3 — Performance Benchmarking
    # ──────────────────────────────────────────────────────────────────────
    # run_benchmarks() uses BenchmarkTools.@belapsed for accurate timings.
    # Larger sizes are used here to make parallel/GPU speedup more visible.
    println("\n[Phase 4] Running performance benchmarks...")
    sizes = [1000, 2000, 4000, 5000]    # n × n system dimensions to benchmark
    results = run_benchmarks(sizes)

    # ──────────────────────────────────────────────────────────────────────
    # Phase 4 — Visualisation
    # ──────────────────────────────────────────────────────────────────────
    println("\n[Phase 4] Generating plots...")

    # --- Panel 1: Execution Time vs. Matrix Size ---
    # Shows raw wall-clock time for each solver so we can see how each scales.
    p1 = plot(
        results["size"], results["serial"],
        label="Serial", marker=:circle,
        title="Execution Time", xlabel="Matrix Size (n)", ylabel="Time (s)"
    )
    plot!(p1, results["size"], results["parallel"], label="Parallel (CPU)", marker=:square)
    if CUDA.functional()
        # Add the GPU line only when a GPU was actually benchmarked
        plot!(p1, results["size"], results["gpu"], label="GPU (CUDA)", marker=:triangle)
    end

    # --- Panel 2: Speedup vs. Matrix Size ---
    # Speedup = t_serial / t_variant.  Values > 1 indicate faster-than-serial.
    speedup_par = results["serial"] ./ results["parallel"]   # Element-wise division
    p2 = plot(
        results["size"], speedup_par,
        label="Parallel Speedup", marker=:square,
        title="Speedup (vs Serial)", xlabel="Matrix Size (n)", ylabel="Speedup"
    )
    if CUDA.functional()
        speedup_gpu = results["serial"] ./ results["gpu"]
        plot!(p2, results["size"], speedup_gpu, label="GPU Speedup", marker=:triangle)
    end

    # Combine both panels into a single 2-row figure and save to disk.
    # layout=(2,1) stacks the panels vertically; size=(800,800) gives good resolution.
    final_plot = plot(p1, p2, layout=(2,1), size=(800, 800))
    savefig("outputs/cg_benchmark_results.png")
    println("Benchmark plot saved to 'outputs/cg_benchmark_results.png'")

    # ──────────────────────────────────────────────────────────────────────
    # Phase 5 — Summary
    # ──────────────────────────────────────────────────────────────────────
    println("\n[Phase 5] Summary")
    println("Project structure complete. Report and slides can be prepared based on 'outputs/cg_benchmark_results.png'.")
end


# =============================================================================
# Script Entry-Point Guard
# =============================================================================
# Only call main() when this file is executed directly (julia main.jl).
# When included from another script or the REPL, main() is defined but
# not invoked automatically, allowing callers to control execution.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
