#!/usr/bin/env julia

using HomotopyContinuation
using LinearAlgebra
using Printf
using Random
using Serialization
using Dates

const HC = HomotopyContinuation

struct Options
    degree::Int
    quotient_labels::Bool
    target_solutions_count::Union{Nothing,Int}
    max_loops_no_progress::Int
    timeout::Union{Nothing,Float64}
    seed::Int
    threading::Bool
    compile::Bool
    verify::Bool
    certified_duplicates::Bool
    save_path::Union{Nothing,String}
    checkpoint_dir::Union{Nothing,String}
    checkpoint_every_seconds::Float64
    target_step::Union{Nothing,Int}
    start_checkpoint::Union{Nothing,String}
    stop_after::Union{Nothing,Float64}
end

function usage()
    println("""
    Count pentagrams on a general plane curve by monodromy.

    Usage:
      julia --project=. scripts/count_pentagrams.jl [options]

    Options:
      --degree D                 Degree of the plane curve. Default: 5.
      --ordered                  Count ordered 5-tuples of lines. Default counts modulo S_5.
      --target N                 Stop when N solutions/equivalence classes are found.
                                 Default: 1968 for degree 5 modulo S_5; unset otherwise.
      --no-target                Do not set a target; use monodromy's heuristic stop only.
      --max-loops-no-progress N  Monodromy stopping heuristic. Default: 200.
      --timeout SECONDS          Wall-clock timeout passed to monodromy_solve.
      --seed N                   Random seed for reproducibility. Default: 20260521.
      --threading true|false     Enable HomotopyContinuation threading. Default: true.
      --compile true|false       Compile the system. Default: true.
      --verify                   Run verify_solution_completeness after monodromy.
      --certified-duplicates     Use duplicate_check = :certified.
      --save PATH                Serialize the MonodromyResult to PATH.
      --checkpoint-dir DIR       Periodically serialize partial path results to DIR.
      --checkpoint-every-seconds N
                                 Seconds between checkpoints. Default: 60.
      --target-step N            Milestone checkpoint mode: solve to current+N, save, then continue.
      --start-checkpoint PATH    Resume starts from a serialized checkpoint/result payload.
      --stop-after SECONDS       Gracefully stop from the checkpoint callback after this many seconds.
      --help                     Show this message.

    Example:
      julia --project=. scripts/count_pentagrams.jl --degree 5 --target 1968
    """)
end

function parse_bool(s::AbstractString)
    lower = lowercase(s)
    if lower in ("1", "true", "yes", "y")
        return true
    elseif lower in ("0", "false", "no", "n")
        return false
    else
        error("expected a boolean, got '$s'")
    end
end

function parse_options(args)
    degree = 5
    quotient_labels = true
    target = nothing
    target_was_set = false
    no_target = false
    max_loops_no_progress = 200
    timeout = nothing
    seed = 20260521
    threading = true
    compile = true
    verify = false
    certified_duplicates = false
    save_path = nothing
    checkpoint_dir = nothing
    checkpoint_every_seconds = 60.0
    target_step = nothing
    start_checkpoint = nothing
    stop_after = nothing

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif arg == "--degree"
            i += 1
            degree = parse(Int, args[i])
        elseif arg == "--ordered"
            quotient_labels = false
        elseif arg == "--target"
            i += 1
            target = parse(Int, args[i])
            target_was_set = true
        elseif arg == "--no-target"
            target = nothing
            no_target = true
        elseif arg == "--max-loops-no-progress"
            i += 1
            max_loops_no_progress = parse(Int, args[i])
        elseif arg == "--timeout"
            i += 1
            timeout = parse(Float64, args[i])
        elseif arg == "--seed"
            i += 1
            seed = parse(Int, args[i])
        elseif arg == "--threading"
            i += 1
            threading = parse_bool(args[i])
        elseif arg == "--compile"
            i += 1
            compile = parse_bool(args[i])
        elseif arg == "--verify"
            verify = true
        elseif arg == "--certified-duplicates"
            certified_duplicates = true
        elseif arg == "--save"
            i += 1
            save_path = args[i]
        elseif arg == "--checkpoint-dir"
            i += 1
            checkpoint_dir = args[i]
        elseif arg == "--checkpoint-every-seconds"
            i += 1
            checkpoint_every_seconds = parse(Float64, args[i])
        elseif arg == "--target-step"
            i += 1
            target_step = parse(Int, args[i])
        elseif arg == "--start-checkpoint"
            i += 1
            start_checkpoint = args[i]
        elseif arg == "--stop-after"
            i += 1
            stop_after = parse(Float64, args[i])
        else
            error("unknown argument '$arg'; run with --help")
        end
        i += 1
    end

    if degree < 1
        error("--degree must be positive")
    end

    if !target_was_set && !no_target && degree == 5
        target = quotient_labels ? 1968 : 1968 * factorial(5)
    end

    Options(
        degree,
        quotient_labels,
        target,
        max_loops_no_progress,
        timeout,
        seed,
        threading,
        compile,
        verify,
        certified_duplicates,
        save_path,
        checkpoint_dir,
        checkpoint_every_seconds,
        target_step,
        start_checkpoint,
        stop_after,
    )
end

function monomial_exponents(d::Int)
    exps = NTuple{3,Int}[]
    for a in d:-1:0
        for b in (d-a):-1:0
            c = d - a - b
            push!(exps, (a, b, c))
        end
    end
    exps
end

function cross_product(u, v)
    [
        u[2] * v[3] - u[3] * v[2],
        u[3] * v[1] - u[1] * v[3],
        u[1] * v[2] - u[2] * v[1],
    ]
end

function curve_value(coeffs, exps, point)
    sum(coeffs[k] * point[1]^exps[k][1] * point[2]^exps[k][2] * point[3]^exps[k][3] for k in eachindex(exps))
end

function random_charts(rng::AbstractRNG)
    charts = randn(rng, ComplexF64, 5, 4)
    for i in 1:5
        charts[i, 4] += 1.0 + 0im
    end
    charts
end

function affine_charts()
    charts = zeros(ComplexF64, 5, 4)
    charts[:, 3] .= 1.0 + 0im
    charts[:, 4] .= -1.0 + 0im
    charts
end

function random_lines_in_charts(charts, rng::AbstractRNG)
    lines = zeros(ComplexF64, 3, 5)
    for i in 1:5
        line = randn(rng, ComplexF64, 3)
        denominator = sum(charts[i, k] * line[k] for k in 1:3)
        while abs(denominator) < 1e-8
            line = randn(rng, ComplexF64, 3)
            denominator = sum(charts[i, k] * line[k] for k in 1:3)
        end
        lines[:, i] .= (-charts[i, 4] / denominator) .* line
    end
    lines
end

function eval_monomials(exps, point)
    [point[1]^a * point[2]^b * point[3]^c for (a, b, c) in exps]
end

function coefficients_for_curve_through_vertices(lines, exps, rng::AbstractRNG)
    rows = Vector{Vector{ComplexF64}}()
    for i in 1:5
        for j in (i+1):5
            vertex = cross_product(lines[:, i], lines[:, j])
            push!(rows, eval_monomials(exps, vertex))
        end
    end

    evaluation_matrix = reduce(vcat, transpose.(rows))
    factorization = svd(evaluation_matrix; full = true)
    rank_estimate = count(>(1e-9 * maximum(factorization.S)), factorization.S)
    kernel = factorization.Vt'[:, (rank_estimate+1):end]
    if size(kernel, 2) == 0
        error("the sampled pentagram imposes independent conditions with no degree $(sum(exps[1])) curve through it")
    end

    kernel * randn(rng, ComplexF64, size(kernel, 2))
end

function start_pair(charts, exps, rng::AbstractRNG)
    lines = random_lines_in_charts(charts, rng)
    coefficients = coefficients_for_curve_through_vertices(lines, exps, rng)
    vec(lines), coefficients
end

function build_system(d::Int; charts = affine_charts())
    exps = monomial_exponents(d)
    ncoeffs = length(exps)

    @var ell[1:3, 1:5]
    @var coeffs[1:ncoeffs]

    equations = Expression[]
    for i in 1:5
        for j in (i+1):5
            vertex = cross_product(ell[:, i], ell[:, j])
            push!(equations, curve_value(coeffs, exps, vertex))
        end
    end

    for i in 1:5
        push!(equations, charts[i, 4] + sum(charts[i, k] * ell[k, i] for k in 1:3))
    end

    variables = vec(ell)
    parameters = coeffs
    System(equations; variables, parameters), charts, exps
end

function relabeling_action(charts)
    permutations = SymmetricGroup(5)
    function action(solution)
        map(permutations) do p
            relabeled = similar(solution)
            for new_i in 1:5
                old_i = p[new_i]
                old_block = @view solution[(3old_i-2):(3old_i)]
                denominator = sum(charts[new_i, k] * old_block[k] for k in 1:3)
                scale = -charts[new_i, 4] / denominator
                relabeled[(3new_i-2):(3new_i)] .= scale .* old_block
            end
            relabeled
        end
    end
    action
end

function short_status(result)
    if is_success(result)
        return "success"
    elseif is_heuristic_stop(result)
        return "heuristic_stop"
    else
        return string(result.returncode)
    end
end

function checkpoint_writer(opts, charts, exps)
    isnothing(opts.checkpoint_dir) && return nothing

    mkpath(opts.checkpoint_dir)
    started_at = time()
    last_checkpoint_at = Ref(0.0)
    checkpoint_index = Ref(0)

    function write_checkpoint(results; force = false)
        elapsed = time() - started_at
        if !force && elapsed - last_checkpoint_at[] < opts.checkpoint_every_seconds
            return false
        end

        checkpoint_index[] += 1
        last_checkpoint_at[] = elapsed
        path = joinpath(
            opts.checkpoint_dir,
            @sprintf("checkpoint_%05d_%06d_results.jls", checkpoint_index[], length(results)),
        )
        tmp_path = path * ".tmp"
        payload = Dict{Symbol,Any}(
            :created_at => string(now()),
            :elapsed_seconds => elapsed,
            :degree => opts.degree,
            :quotient_labels => opts.quotient_labels,
            :seed => opts.seed,
            :nresults => length(results),
            :charts => charts,
            :exponents => exps,
            :results => copy(results),
        )
        open(tmp_path, "w") do io
            serialize(io, payload)
        end
        mv(tmp_path, path; force = true)
        @printf("\ncheckpoint saved:         %s (%d results, %.1fs)\n", path, length(results), elapsed)
        return true
    end

    function callback(results)
        write_checkpoint(results)
        if !isnothing(opts.stop_after) && time() - started_at >= opts.stop_after
            write_checkpoint(results; force = true)
            return true
        end
        return false
    end

    return callback
end

function save_result_payload(path, opts, result, charts, exps)
    mkpath(dirname(path))
    tmp_path = path * ".tmp"
    payload = Dict{Symbol,Any}(
        :created_at => string(now()),
        :degree => opts.degree,
        :quotient_labels => opts.quotient_labels,
        :seed => opts.seed,
        :nsolutions => nsolutions(result),
        :nresults => nresults(result),
        :parameters => parameters(result),
        :base_parameters => parameters(result),
        :base_solutions => solutions(result),
        :charts => charts,
        :exponents => exps,
        :result => result,
    )
    open(tmp_path, "w") do io
        serialize(io, payload)
    end
    mv(tmp_path, path; force = true)
end

function load_start_checkpoint(path, fallback_p0)
    payload = open(deserialize, path)
    if haskey(payload, :result)
        result = payload[:result]
        starts = get(payload, :base_solutions, solutions(result))
        base_parameters = get(payload, :base_parameters, parameters(result))
        return starts, base_parameters, get(payload, :nsolutions, length(starts))
    elseif haskey(payload, :results)
        starts = [solution(result) for result in payload[:results]]
        p0 = get(payload, :parameters, fallback_p0)
        return starts, p0, length(starts)
    else
        error("checkpoint '$path' does not contain :result or :results")
    end
end

function run(opts::Options)
    Random.seed!(opts.seed)

    system, charts, exps = build_system(opts.degree)

    println("Pentagram monodromy experiment")
    println("==============================")
    println("degree:                   ", opts.degree)
    println("variables:                ", nvariables(system))
    println("parameters:               ", nparameters(system), " curve coefficients")
    println("equations:                ", length(expressions(system)))
    println("monomials:                ", length(exps))
    println("counting modulo labels:   ", opts.quotient_labels)
    println("target:                   ", isnothing(opts.target_solutions_count) ? "unset" : opts.target_solutions_count)
    println("max loops no progress:    ", opts.max_loops_no_progress)
    println("timeout:                  ", isnothing(opts.timeout) ? "unset" : string(opts.timeout, " seconds"))
    println("seed:                     ", opts.seed)
    println("checkpoint dir:           ", isnothing(opts.checkpoint_dir) ? "unset" : opts.checkpoint_dir)
    println("target step:              ", isnothing(opts.target_step) ? "unset" : opts.target_step)
    println("start checkpoint:         ", isnothing(opts.start_checkpoint) ? "unset" : opts.start_checkpoint)
    println("stop after:               ", isnothing(opts.stop_after) ? "unset" : string(opts.stop_after, " seconds"))
    println()

    if opts.degree < 4
        error("a general pentagram is not contained in a curve of degree $(opts.degree); use degree >= 4")
    end

    x0, p0 = start_pair(charts, exps, Random.default_rng())
    start_residual = norm(evaluate(system, x0, p0), Inf)
    @printf("start residual:           %.3e\n\n", start_residual)

    kwargs = Dict{Symbol,Any}(
        :show_progress => true,
        :threading => opts.threading,
        :compile => opts.compile,
        :max_loops_no_progress => opts.max_loops_no_progress,
    )
    if !isnothing(opts.target_solutions_count)
        kwargs[:target_solutions_count] = opts.target_solutions_count
    end
    if !isnothing(opts.timeout)
        kwargs[:timeout] = opts.timeout
    end
    if opts.certified_duplicates
        kwargs[:duplicate_check] = :certified
    end
    if opts.quotient_labels
        kwargs[:group_action] = relabeling_action(charts)
        kwargs[:equivalence_classes] = true
    end
    callback = isnothing(opts.target_step) ? checkpoint_writer(opts, charts, exps) : nothing
    if !isnothing(callback)
        kwargs[:loop_finished_callback] = callback
    end

    if !isnothing(opts.start_checkpoint)
        starts, p0, start_count = load_start_checkpoint(opts.start_checkpoint, p0)
        @printf("loaded checkpoint starts: %d\n\n", start_count)
    else
        starts = x0
    end

    run_started_at = time()
    result = nothing

    if isnothing(opts.target_step)
        result = monodromy_solve(system, starts, p0; kwargs...)
    else
        isnothing(opts.checkpoint_dir) && error("--target-step requires --checkpoint-dir")
        mkpath(opts.checkpoint_dir)
        current_count = starts isa AbstractVector{<:Number} ? 1 : length(starts)
        batch = 0
        while true
            batch += 1
            target = current_count + opts.target_step
            kwargs[:target_solutions_count] = target
            @printf("\nMilestone batch %d: target %d solutions/classes\n", batch, target)
            result = monodromy_solve(system, starts, p0; kwargs...)

            current_count = nsolutions(result)
            checkpoint_path = joinpath(
                opts.checkpoint_dir,
                @sprintf("batch_%05d_%08d_solutions.jls", batch, current_count),
            )
            save_result_payload(checkpoint_path, opts, result, charts, exps)
            @printf("milestone checkpoint:     %s\n", checkpoint_path)

            if current_count < target || is_heuristic_stop(result)
                break
            end
            if !isnothing(opts.stop_after) && time() - run_started_at >= opts.stop_after
                break
            end

            starts = solutions(result)
            p0 = parameters(result)
        end
    end

    println()
    println("Result")
    println("------")
    println("status:                   ", short_status(result))
    println("solutions reported:       ", nsolutions(result))
    println("raw path results:         ", nresults(result))
    println("base parameter length:    ", length(parameters(result)))

    if opts.verify && opts.quotient_labels
        println()
        println("Skipping completeness verification: trace-test verification applies to the full")
        println("ordered fiber, while this run is quotienting by S_5. Re-run with --ordered --verify.")
    elseif opts.verify
        println()
        println("Running trace-test completeness verification...")
        complete = verify_solution_completeness(
            system,
            solutions(result),
            parameters(result);
            show_progress = true,
            monodromy_options = (
                threading = opts.threading,
                compile = opts.compile,
                max_loops_no_progress = opts.max_loops_no_progress,
            ),
        )
        println("verified complete:        ", complete)
    end

    if !isnothing(opts.save_path)
        save_result_payload(opts.save_path, opts, result, charts, exps)
        println("saved result:             ", opts.save_path)
    end

    result
end

if abspath(PROGRAM_FILE) == @__FILE__
    opts = parse_options(ARGS)
    run(opts)
end
