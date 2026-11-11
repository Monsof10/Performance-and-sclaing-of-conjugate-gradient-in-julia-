# =============================================================================
# scripts/benchmarks.jl
# =============================================================================
# Benchmarking harness for the three CG solver implementations:
#   - Serial   (single-threaded CPU)
#   - Parallel (multi-threaded CPU via Threads.@threads)
#   - GPU      (CUDA, if a compatible GPU is available)
#
# The script defines a single function `run_benchmarks(sizes)` that:
#   1. Iterates over a list of matrix sizes.
#   2. Constructs an SPD system for each size.
#   3. Uses BenchmarkTools.@belapsed to obtain reliable, GC-compensated
#      wall-clock timings for each solver variant.
#   4. Collects results in a Dict and prints per-size speedup summaries.
#
# Called from main.jl; can also be run standalone:
#   julia scripts/benchmarks.jl
# =============================================================================

# Include source modules relative to this script's location.
# When invoked from main.jl these may already be included, but Julia's
# module system is idempotent so duplicate includes are harmless.
include("../src/solvers.jl")
include("../src/utils.jl")

using .Solvers         # cg_serial, cg_parallel, cg_gpu
using .Utils           # generate_spd_matrix
using BenchmarkTools   # @belapsed — accurate micro-benchmarking macro
using CUDA             # CUDA.functional() — GPU availability check
using Statistics       # mean, median (available for extended analysis)


"""
    run_benchmarks(sizes = [500, 1000, 2000, 3000, 4000, 5000])

Run timed benchmarks of all three CG solver variants over a range of matrix
sizes and return the collected timing data.

### BenchmarkTools note
`@belapsed expr` is used instead of `@elapsed` or `@time` because it:
  - Runs the expression multiple times and returns the *minimum* elapsed time.
  - Automatically warms up the JIT before timing.
  - Accounts for garbage-collection pauses.
This gives the most reproducible and meaningful performance measurement.

### Parameters
- `sizes` : vector of integers, each specifying the dimension `n` of the
            `n × n` SPD system to benchmark.

### Returns
A `Dict` with keys:
  - `"size"`     → the input `sizes` vector
  - `"serial"`   → `Vector{Float64}` of serial CG times (seconds)
  - `"parallel"` → `Vector{Float64}` of parallel CG times (seconds)
  - `"gpu"`      → `Vector{Float64}` of GPU CG times (seconds), or `NaN`
                   entries when CUDA is unavailable
"""
function run_benchmarks(sizes = [500, 1000, 2000, 3000, 4000, 5000])

    # Initialise results dictionary.
    # `sizes` is stored by reference so the caller can correlate times to sizes.
    results = Dict(
        "size"     => sizes,
        "serial"   => Float64[],   # One entry per size, filled below
        "parallel" => Float64[],
        "gpu"      => Float64[]
    )

    # Check once whether a CUDA-capable GPU is available.
    # CUDA.functional() returns false if no GPU is found, drivers are missing,
    # or the CUDA toolkit version is incompatible.
    has_gpu = CUDA.functional()
    if !has_gpu
        println("CUDA is not available. GPU benchmarks will be skipped.")
    end

    # Report how many CPU threads Julia was started with.
    # The parallel solver benefits from more threads (start with julia -t N).
    num_threads = Threads.nthreads()
    println("Running benchmarks with $num_threads CPU threads.")

    # ------------------------------------------------------------------
    # Main benchmarking loop — one iteration per problem size
    # ------------------------------------------------------------------
    for n in sizes
        println("Benchmarking size: $n x $n...")

        # Build a fresh n×n SPD matrix and a random right-hand side.
        # Using a fresh matrix each time prevents cache-warming effects
        # from biasing later benchmark runs.
        A = generate_spd_matrix(n)
        b = rand(n)

        # --- Serial benchmark ---
        # @belapsed automatically warms up and returns the minimum time in seconds.
        t_serial = @belapsed cg_serial($A, $b)
        push!(results["serial"], t_serial)

        # --- Parallel (multi-threaded CPU) benchmark ---
        t_parallel = @belapsed cg_parallel($A, $b)
        push!(results["parallel"], t_parallel)

        # --- GPU benchmark (conditional) ---
        if has_gpu
            # Time the full GPU solve, including host→device data transfer,
            # because that is part of the real-world cost.
            t_gpu = @belapsed cg_gpu($A, $b)
            push!(results["gpu"], t_gpu)
        else
            # Record NaN as a sentinel so the results dict stays consistent in
            # shape even when GPU is unavailable (NaN is excluded from plots).
            push!(results["gpu"], NaN)
        end

        # --- Print per-size summary ---
        println("  Serial:   $(round(t_serial,   digits=4)) s")
        println("  Parallel: $(round(t_parallel, digits=4)) s " *
                "(Speedup: $(round(t_serial/t_parallel, digits=2))x)")

        if has_gpu
            println("  GPU:      $(round(t_gpu, digits=4)) s " *
                    "(Speedup: $(round(t_serial/t_gpu, digits=2))x)")
        end
    end

    # Return the populated results dictionary for use by the caller
    # (e.g. main.jl uses it to generate plots).
    return results
end
