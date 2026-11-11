# =============================================================================
# scripts/matrix_solve_demo.jl
# =============================================================================
# Demonstrates the Conjugate Gradient (CG) algorithm by explicitly showing
# the matrix, the right-hand side vector, and the step-by-step results of
# solving the system with all three implementations:
#   1. Serial CPU
#   2. Parallel CPU
#   3. GPU (CUDA)
#
# This script is meant for educational/debugging purposes to verify that
# all three algorithms compute identical solutions on a small, readable matrix.
# =============================================================================

using Pkg

# Activate the project-local environment
Pkg.activate("env")

# Bring in standard libraries
using LinearAlgebra
using Printf

# Include project modules
include("../src/solvers.jl")
include("../src/utils.jl")

using .Solvers
using .Utils
using CUDA

function main()
    println("================================================================")
    println("        Conjugate Gradient Matrix Solve Demonstration")
    println("================================================================\n")

    # 1. Create a small, readable SPD matrix (e.g., 5x5)
    n = 5
    println("Generating a small $n × $n Symmetric Positive Definite (SPD) matrix A...")
    A = generate_spd_matrix(n)
    
    # Generate a random right-hand side vector b
    b = rand(n)

    # Display the matrix A and vector b
    println("\nMatrix A ($n × $n):")
    display(round.(A, digits=4))
    
    println("\nVector b (Right-Hand Side):")
    display(round.(b, digits=4))
    
    # Also calculate the exact solution using Julia's built-in direct solver (\)
    x_exact = A \ b
    println("\nExact Solution x (calculated via direct solver A \\ b):")
    display(round.(x_exact, digits=6))
    println("\n----------------------------------------------------------------")

    # =========================================================================
    # 1. Solve using Serial CG
    # =========================================================================
    println("\n>>> Running Serial CG Solver...")
    x_ser, iter_ser, res_ser = cg_serial(A, b, tol=1e-6)
    
    println("  Iterations required: $iter_ser")
    println("  Final Residual Norm: $(@sprintf(\"%.2e\", res_ser))")
    println("  Solution x_serial:")
    display(round.(x_ser, digits=6))
    
    # Calculate difference from exact solution
    err_ser = norm(x_ser - x_exact)
    println("  Difference from exact solution: $(@sprintf(\"%.2e\", err_ser))")


    # =========================================================================
    # 2. Solve using Parallel CG
    # =========================================================================
    println("\n>>> Running Parallel CG Solver...")
    x_par, iter_par, res_par = cg_parallel(A, b, tol=1e-6)
    
    println("  Iterations required: $iter_par")
    println("  Final Residual Norm: $(@sprintf(\"%.2e\", res_par))")
    println("  Solution x_parallel:")
    display(round.(x_par, digits=6))

    # Calculate difference from exact solution
    err_par = norm(x_par - x_exact)
    println("  Difference from exact solution: $(@sprintf(\"%.2e\", err_par))")


    # =========================================================================
    # 3. Solve using GPU CG (if available)
    # =========================================================================
    println("\n>>> Running GPU CG Solver...")
    if CUDA.functional()
        x_gpu, iter_gpu, res_gpu = cg_gpu(A, b, tol=1e-6)
        
        println("  Iterations required: $iter_gpu")
        println("  Final Residual Norm: $(@sprintf(\"%.2e\", res_gpu))")
        println("  Solution x_gpu:")
        display(round.(x_gpu, digits=6))
        
        # Calculate difference from exact solution
        err_gpu = norm(x_gpu - x_exact)
        println("  Difference from exact solution: $(@sprintf(\"%.2e\", err_gpu))")
    else
        println("  [!] CUDA GPU not available or functional. Skipping GPU solve.")
    end
    
    println("\n================================================================")
    println("                       Demonstration Complete")
    println("================================================================")
end

# Run the demonstration
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
