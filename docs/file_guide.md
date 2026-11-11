# Project File Guide
> A plain-language explanation of every script in this project.

---

## 📁 Project Structure at a Glance

```
project/
├── main.jl                      ← the "start button" — runs everything
├── src/
│   ├── solvers.jl               ← the 3 CG solver implementations
│   └── utils.jl                 ← helper functions (matrix builder, validator)
├── scripts/
│   ├── benchmarks.jl            ← measures how fast each solver is
│   └── profiling_analysis.jl    ← shows WHERE time is spent inside a solver
└── python/
    └── cg_python.py             ← Python version for comparison
```

---

## 1. `main.jl` — The Entry Point

**What it does:** You run this file. It wires everything together in 4 phases.

| Phase | What happens |
|-------|-------------|
| **Phase 1** | Checks that all required packages (CUDA, BenchmarkTools, Plots) are installed. Installs any that are missing. |
| **Phase 2** | Tests all 3 solvers on a 1000×1000 matrix to make sure they give correct answers before wasting time benchmarking. |
| **Phase 3** | Calls `run_benchmarks()` with sizes [1000, 2000, 4000, 5000] and collects timing results. |
| **Phase 4** | Draws two plots and saves them to `outputs/cg_benchmark_results.png`: ① Execution time vs. matrix size, ② Speedup vs. matrix size. |

---

## 2. `src/solvers.jl` — The 3 CG Solvers

**What it does:** Contains all three versions of the Conjugate Gradient algorithm. They all solve the same math problem (`A·x = b`) but in different ways.

### `cg_serial` — Single-threaded (baseline)
- Uses **one CPU core**.
- Follows the standard CG algorithm step-by-step.
- Returns the answer `x`, the number of iterations, and the final error (residual).

### `cg_parallel` — Multi-threaded CPU
- Same algorithm as serial, but the **matrix-vector multiply** (the slowest step) is split across multiple CPU cores using `Threads.@threads`.
- Uses in-place updates (`.+=`, `.-=`) to avoid creating extra memory copies every loop.
- Requires starting Julia with `-t N` (e.g. `julia -t 4`) to use N threads.

### `threaded_matvec!` — Helper for parallel
- Called inside `cg_parallel`.
- Splits the rows of matrix `A` across threads so each thread computes its own chunk of `A·p`.

### `cg_gpu` — GPU via CUDA
- Copies the matrix and vector to the **GPU memory** first (`cu(A)`, `cu(b)`).
- All the heavy math runs on the GPU.
- Copies the final answer back to the CPU before returning.
- Skipped automatically if no GPU is found.

---

## 3. `src/utils.jl` — Helper Functions

**What it does:** Provides two support functions used by all the solvers and benchmarks.

### `generate_spd_matrix(n)`
- Builds a random **n×n test matrix** that is guaranteed to work with CG.
- *(CG only works on "Symmetric Positive Definite" matrices, so we can't just use any random matrix.)*
- Steps: generate random → multiply by its transpose → add a small identity shift → done.

### `validate_solution(A, b, x)`
- After solving, checks how good the answer is.
- Computes `‖b - A·x‖` (the residual norm). A value near 0 means the answer is correct.

---

## 4. `scripts/benchmarks.jl` — Timing Measurements

**What it does:** Measures how long each solver takes across different problem sizes.

### Key parts:
- **`run_benchmarks(sizes)`** — the main function. Loops over each matrix size and times all 3 solvers.
- Uses **`@belapsed`** (from BenchmarkTools) instead of a simple timer — it runs the solver multiple times, warms up the JIT compiler, and returns the *minimum* time. This is more accurate than `@elapsed`.
- Stores results in a dictionary `{"size", "serial", "parallel", "gpu"}` and returns it to `main.jl` for plotting.
- Prints a speedup summary per size, e.g. `Parallel: 0.42s (Speedup: 2.3x)`.
- GPU entries are filled with `NaN` when no GPU is found, so the dictionary shape stays consistent.

---

## 5. `scripts/profiling_analysis.jl` — Where Time is Spent

**What it does:** Instead of comparing solvers, this script zooms *inside* one solver to find which lines of code are slowest.

### Key parts:
- **`@btime`** — runs the solver and reports: time taken, memory allocated, number of heap allocations. Useful to spot wasted memory usage.
- **`@profile`** + `Profile.print()` — samples the Julia call stack hundreds of times during a solve. Tells you what percentage of time was spent in each function/line.
- A **warm-up call** is done before profiling so Julia's JIT compiler doesn't count in the results.
- Suggests using `ProfileView` for an interactive flame graph.

---

## 6. `python/cg_python.py` — Python Comparison

**What it does:** A pure Python (NumPy) version of the CG algorithm. Used to compare Julia's speed against Python.

### Key parts:

| Part | What it does |
|------|-------------|
| `generate_spd_matrix(n)` | Same idea as the Julia version — builds a valid test matrix. Uses `np.random.rand` + symmetrise + add `n·I`. |
| `cg_python(A, b)` | The CG solver itself. Pure NumPy. Same math as the Julia serial version. |
| **Benchmark loop** (`__main__`) | Runs `cg_python` for sizes [1000, 2000, 4000, 5000], records wall-clock time with `time.time()`. |
| **Plotting** | Draws a simple line chart of time vs. size using `matplotlib` and saves it to `outputs/python_benchmark_results.png`. |

---

## How the files connect

```
main.jl
  ├── includes src/solvers.jl    → cg_serial, cg_parallel, cg_gpu
  ├── includes src/utils.jl      → generate_spd_matrix, validate_solution
  └── includes scripts/benchmarks.jl → run_benchmarks()
                                          └── uses solvers + utils internally

scripts/profiling_analysis.jl   (standalone, includes solvers + utils itself)
python/cg_python.py             (completely standalone, no Julia dependency)
```
