# =============================================================================
# scripts/profiling_analysis.jl
# =============================================================================
# Deep-dive profiling script for the CG solvers.  While benchmarks.jl focuses
# on *wall-clock time comparisons* across problem sizes, this script focuses on
# *where time is actually spent* inside a single solver invocation.
#
# Three analyses are performed and saved as a single 3-panel plot:
#
#   Panel 1 — Execution Time vs. Matrix Size
#             Compares serial vs. parallel wall-clock time across several sizes.
#
#   Panel 2 — Memory Allocation vs. Matrix Size
#             Compares heap bytes allocated per call (serial vs. parallel).
#             High allocation counts indicate room for in-place optimisation.
#
#   Panel 3 — Operation Time Breakdown (fixed size n = 2000)
#             Times each key CG sub-operation individually (matvec, dot
#             products, vector updates) to show where time is spent.
#
# A statistical profile (Julia's Profile module) is also run and printed
# to the console as a flat text report.
#
# Output:
#   outputs/profiling_results.png  — three-panel visual profiling report
#
# Run standalone:
#   julia scripts/profiling_analysis.jl
# or with multiple threads:
#   julia -t 4 scripts/profiling_analysis.jl
# =============================================================================

using Pkg
# Activate the project-local environment so that the correct package versions
# (BenchmarkTools, Plots, etc.) are loaded rather than any globally installed ones.
Pkg.activate(joinpath(@__DIR__, "..", "env"))

using Profile            # Built-in Julia statistical call-stack profiler
using BenchmarkTools     # @belapsed / @allocated — accurate micro-benchmarking
using Plots              # Plotting library — GR backend used by default
using LinearAlgebra      # dot, norm — needed for the operation breakdown

# Load solver and utility modules from the src/ directory
include("../src/solvers.jl")
include("../src/utils.jl")
using .Solvers   # cg_serial, cg_parallel, cg_gpu
using .Utils     # generate_spd_matrix


# =============================================================================
# Main Profiling Function
# =============================================================================

"""
    profile_cg()

Run a comprehensive profiling session and save a visual report to
`outputs/profiling_results.png`.

### What is produced
- **Panel 1**: Execution time (ms) — serial vs. parallel, across matrix sizes.
- **Panel 2**: Memory allocated (MB) — serial vs. parallel, across matrix sizes.
- **Panel 3**: Per-operation time breakdown — how time is split inside one CG
               iteration (matvec, two dot products, vector update) at n=2000.
- **Console**: Julia Profile flat report — which lines/functions take most time.
"""
function profile_cg()

    println("=== CG Profiling Analysis ===")
    println("Threads available: $(Threads.nthreads())")

    # ------------------------------------------------------------------
    # Part 1 & 2: Sweep over matrix sizes
    # ------------------------------------------------------------------
    # Use a moderate range: large enough for meaningful data,
    # small enough for the profiling script to finish quickly.
    sizes = [500, 1000, 2000, 3000]

    serial_times    = Float64[]   # wall-clock time in seconds (serial)
    parallel_times  = Float64[]   # wall-clock time in seconds (parallel)
    serial_allocs   = Float64[]   # heap bytes allocated (serial),   → MB
    parallel_allocs = Float64[]   # heap bytes allocated (parallel), → MB

    println("\n[1/3] Timing & allocation sweep across matrix sizes...")

    for n in sizes
        println("  Benchmarking $n × $n ...")
        A = generate_spd_matrix(n)
        b = rand(n)

        # @belapsed: runs multiple trials with JIT warm-up, returns minimum
        # elapsed time in seconds.  The $ interpolation passes variables by
        # reference so variable-lookup cost is not measured.
        t_ser = @belapsed cg_serial($A, $b)
        t_par = @belapsed cg_parallel($A, $b)

        # @allocated: returns total heap bytes allocated in a single call.
        # One call is enough here — we want steady-state allocation, not timing.
        mem_ser = @allocated(cg_serial(A, b)) / 1e6   # bytes → MB
        mem_par = @allocated(cg_parallel(A, b)) / 1e6 # bytes → MB

        push!(serial_times,    t_ser)
        push!(parallel_times,  t_par)
        push!(serial_allocs,   mem_ser)
        push!(parallel_allocs, mem_par)

        println("    Serial:   $(round(t_ser*1000, digits=2)) ms | $(round(mem_ser, digits=2)) MB")
        println("    Parallel: $(round(t_par*1000, digits=2)) ms | $(round(mem_par, digits=2)) MB")
    end

    # ------------------------------------------------------------------
    # Part 3: Operation-level breakdown at n = 2000
    # ------------------------------------------------------------------
    # Time each of the four operations that occur every CG iteration:
    #   (a) Matrix-vector product   A * p        — the expensive step
    #   (b) Dot product             p · Ap       — used to compute step length
    #   (c) Dot product             r · r        — residual norm squared
    #   (d) Vector update           x += α * p   — solution/residual advance
    println("\n[2/3] Operation-level breakdown at n = 2000...")

    n_fixed = 2000
    A_fixed = generate_spd_matrix(n_fixed)
    b_fixed = rand(n_fixed)

    # Build representative vectors for timing individual operations
    x_fixed  = zeros(n_fixed)
    r_fixed  = b_fixed - A_fixed * x_fixed
    p_fixed  = copy(r_fixed)
    Ap_fixed = A_fixed * p_fixed   # pre-compute so we can time dot(p, Ap) alone

    # Minimum time (seconds) for each sub-operation
    t_matvec  = @belapsed $A_fixed * $p_fixed
    t_dot_pAp = @belapsed dot($p_fixed, $Ap_fixed)
    t_dot_rr  = @belapsed dot($r_fixed, $r_fixed)
    t_vecupd  = @belapsed $x_fixed .+ 1.0 .* $p_fixed  # mimics x += α*p

    # Convert to milliseconds for readability
    op_names  = ["Matvec\n(A·p)", "Dot\n(p·Ap)", "Dot\n(r·r)", "Vec Update\n(x+=α·p)"]
    op_ms     = [t_matvec, t_dot_pAp, t_dot_rr, t_vecupd] .* 1000

    println("    Matvec   (A·p):    $(round(t_matvec*1000,  digits=3)) ms")
    println("    Dot      (p·Ap):   $(round(t_dot_pAp*1000, digits=3)) ms")
    println("    Dot      (r·r):    $(round(t_dot_rr*1000,  digits=3)) ms")
    println("    Vec upd  (x+=αp):  $(round(t_vecupd*1000,  digits=3)) ms")

    # ------------------------------------------------------------------
    # Part 4: Statistical profiling (console only)
    # ------------------------------------------------------------------
    println("\n[3/3] Statistical profile of cg_serial at n = 2000...")

    # Warm-up: the first call triggers JIT compilation, which would dominate
    # the profile.  Run once beforehand so only steady-state is captured.
    cg_serial(A_fixed, b_fixed)

    Profile.clear()                         # Remove any stale sampling data
    @profile cg_serial(A_fixed, b_fixed)    # Sample call stack during a full run

    println("\n--- Flat profile report (sorted by sample count, most time first) ---")
    Profile.print(format=:flat, sortedby=:count, maxdepth=15)

    # ------------------------------------------------------------------
    # Build the 3-panel figure
    # ------------------------------------------------------------------
    println("\nGenerating profiling plot...")

    # ── Panel 1: Execution Time vs. Matrix Size ───────────────────────
    # Both lines share the same axes so trends are directly comparable.
    p1 = plot(
        sizes, serial_times .* 1000,
        label     = "Serial",
        marker    = :circle,
        markersize = 6,
        linewidth = 2,
        color     = :steelblue,
        linestyle = :solid,
        title     = "Execution Time",
        xlabel    = "Matrix Size (n)",
        ylabel    = "Time (ms)",
        legend    = :topleft,
        grid      = true,
        framestyle = :box
    )
    plot!(p1,
        sizes, parallel_times .* 1000,
        label     = "Parallel",
        marker    = :square,
        markersize = 6,
        linewidth = 2,
        color     = :darkorange,
        linestyle = :dash
    )

    # ── Panel 2: Memory Allocation vs. Matrix Size ────────────────────
    # Two line series with distinct markers — no StatsPlots required.
    p2 = plot(
        sizes, serial_allocs,
        label     = "Serial",
        marker    = :circle,
        markersize = 6,
        linewidth = 2,
        color     = :steelblue,
        linestyle = :solid,
        title     = "Memory Allocation",
        xlabel    = "Matrix Size (n)",
        ylabel    = "Allocated (MB)",
        legend    = :topleft,
        grid      = true,
        framestyle = :box
    )
    plot!(p2,
        sizes, parallel_allocs,
        label     = "Parallel",
        marker    = :square,
        markersize = 6,
        linewidth = 2,
        color     = :darkorange,
        linestyle = :dash
    )

    # ── Panel 3: Per-Operation Breakdown ─────────────────────────────
    # Vertical bar chart with one bar per CG sub-operation.
    # Colour-coded so each operation is visually distinct.
    bar_colors = [:crimson, :mediumpurple, :mediumseagreen, :goldenrod]

    p3 = bar(
        op_names, op_ms,
        color      = bar_colors,
        legend     = false,
        title      = "Op Breakdown (n=2000)",
        xlabel     = "Operation",
        ylabel     = "Time (ms)",
        grid       = true,
        bar_width  = 0.55,
        framestyle = :box
    )
    # Annotate each bar with its value so exact numbers are visible
    for (i, val) in enumerate(op_ms)
        annotate!(p3, i, val + maximum(op_ms)*0.02,
                  text("$(round(val, digits=3)) ms", 7, :center))
    end

    # ── Combine into a single figure ─────────────────────────────────
    final_plot = plot(
        p1, p2, p3,
        layout     = (1, 3),
        size       = (1350, 420),
        margin     = 6Plots.mm,
        title      = ["Execution Time" "Memory Allocation" "Op Breakdown (n=2000)"],
        titleloc   = :center
    )

    # Ensure the outputs directory exists before saving
    out_dir  = joinpath(@__DIR__, "..", "outputs")
    mkpath(out_dir)
    out_path = joinpath(out_dir, "profiling_results.png")
    savefig(final_plot, out_path)

    println("Profiling plot saved to '$out_path'")
    println("\nDone.")
end


# =============================================================================
# Script Entry Point
# =============================================================================

# Guard: only call profile_cg() when this file is executed directly.
# When included from another script (e.g. during testing), the function is
# defined but not automatically invoked.
if abspath(PROGRAM_FILE) == @__FILE__
    profile_cg()
end
