#!/usr/bin/env julia

using HomotopyContinuation
include(joinpath(@__DIR__, "..", "src", "HessianSimplex.jl"))
using .HessianSimplex

d = parse(Int, get(ARGS, 1, "1"))
timeout = parse(Int, get(ARGS, 2, d == 1 ? "60" : "600"))
flags = Set(ARGS[3:end])
verify = !isempty(intersect(flags, Set(["true", "verify", "yes"])))
use_symmetry = !isempty(intersect(flags, Set(["symmetry", "symmetric", "quotient"])))

function flag_value(name)
    prefix = name * "="
    for arg in ARGS[3:end]
        startswith(arg, prefix) && return arg[(lastindex(prefix) + 1):end]
    end
    return nothing
end

checkpoint_path = flag_value("checkpoint")
resume_path = flag_value("resume")
checkpoint_interval_arg = flag_value("checkpoint_interval")
checkpoint_interval = isnothing(checkpoint_interval_arg) ? 5 : parse(Float64, checkpoint_interval_arg)
duplicate_check_arg = flag_value("duplicate_check")
duplicate_check = isnothing(duplicate_check_arg) ? :heuristic : Symbol(duplicate_check_arg)
duplicate_check in (:heuristic, :certified) ||
    error("duplicate_check must be heuristic or certified")
reuse_loops_arg = flag_value("reuse_loops")
reuse_loops = isnothing(reuse_loops_arg) ? :all : Symbol(reuse_loops_arg)
reuse_loops in (:all, :random, :none) ||
    error("reuse_loops must be all, random, or none")
max_loops_no_progress_arg = flag_value("max_loops_no_progress")
max_loops_no_progress =
    isnothing(max_loops_no_progress_arg) ? 5 : parse(Int, max_loops_no_progress_arg)
min_solutions_arg = flag_value("min_solutions")
min_solutions = isnothing(min_solutions_arg) ? nothing : parse(Int, min_solutions_arg)

start_solutions = nothing
start_parameters = nothing
gauge_coefficients = nothing
if !isnothing(resume_path)
    metadata, start_solutions, start_parameters = read_checkpoint(resume_path)
    metadata["d"] == d || error("checkpoint d=$(metadata["d"]) does not match requested d=$d")
    gauge_coefficients = HessianSimplex.pairs_to_complex(metadata["gauge_coefficients"])
    checkpoint_path = isnothing(checkpoint_path) ? resume_path : checkpoint_path
    use_symmetry = use_symmetry || get(metadata, "use_symmetry", false)
end

println("Hessian simplex map, d = $d")
println("naive Plucker count before base correction = $(expected_degree(d))")
println("Julia threads = $(Threads.nthreads())")
println("duplicate_check = $duplicate_check")
println("reuse_loops = $reuse_loops")
if use_symmetry
    println("using S_$(d + 1) symmetry; reported solutions are orbit classes")
end
if !isnothing(checkpoint_path)
    println("checkpoint = $checkpoint_path")
end
if !isnothing(resume_path)
    println("resuming from $resume_path with $(length(start_solutions)) saved solutions")
end

monodromy_kwargs = Dict{Symbol, Any}(
    :timeout => timeout,
    :use_symmetry => use_symmetry,
    :compile => true,
    :duplicate_check => duplicate_check,
    :reuse_loops => reuse_loops,
    :max_loops_no_progress => max_loops_no_progress,
    :checkpoint_path => checkpoint_path,
    :checkpoint_interval => checkpoint_interval,
    :start_solutions => start_solutions,
    :start_parameters => start_parameters,
)
if !isnothing(min_solutions)
    monodromy_kwargs[:min_solutions] = min_solutions
end
if !isnothing(gauge_coefficients)
    monodromy_kwargs[:gauge_coefficients] = gauge_coefficients
end

model, result = monodromy_degree(d; monodromy_kwargs...)

println(result)
println("solutions found = $(nsolutions(result))")
if use_symmetry
    println("ordered count if all stabilizers are trivial = $(nsolutions(result) * factorial(big(d + 1)))")
end
println("parameters = $(parameters(result))")

if nsolutions(result) > 0
    sol = first(solutions(result))
    pars = parameters(result)
    println("max system residual = $(maximum(abs.(chart_residuals(model, sol, pars))))")
    println("max omitted-product residual = $(maximum(abs.(omitted_product_residuals(model, sol, pars))))")
end

if verify
    complete = verify_solution_completeness(model.system, result; show_progress = true)
    println("trace-test complete = $complete")
end
