#!/usr/bin/env julia

include(joinpath(@__DIR__, "scroll_projection_ramification.jl"))

using .ScrollProjectionRamification
using HomotopyContinuation
using LinearAlgebra
using Random
using Serialization

const SPR = ScrollProjectionRamification
const HC = HomotopyContinuation

function usage()
    return """
    Usage:
      julia --project=. -t auto classify_O22_real_ramification.jl --start-checkpoint CHECKPOINT [options]

    This samples real bidegree (1,4) ramification curves for E = O(2)+O(2),
    tracks the two known complex PR preimages from CHECKPOINT to each sampled
    real target, and records whether the endpoint fiber has 0 or 2 real points.

    Options:
      --samples N             Number of random real targets to try. Default: 50.
      --seed N                Random seed. Default: 20260610.
      --out PATH              Serialized output path. Default: checkpoints/O22_real_classification.jls.
      --smooth-tol TOL        Reject targets with small quartic resultant. Default: 1e-8.
      --real-atol TOL         Numerical realness tolerance. Default: 1e-6.
      --gamma-retries N       Retry random complex gamma this many times. Default: 5.
      --help                  Print this help.
    """
end

function parse_args(args)
    opts = Dict{Symbol,Any}(
        :start_checkpoint => nothing,
        :samples => 50,
        :seed => UInt64(20260610),
        :out => joinpath(pwd(), "checkpoints", "O22_real_classification.jls"),
        :smooth_tol => 1e-8,
        :real_atol => 1e-6,
        :gamma_retries => 5,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println(usage())
            exit(0)
        end
        startswith(arg, "--") || error("Unrecognized argument '$arg'.")
        i == length(args) && error("Missing value for $arg.")
        val = args[i + 1]
        if arg == "--start-checkpoint"
            opts[:start_checkpoint] = val
        elseif arg == "--samples"
            opts[:samples] = parse(Int, val)
        elseif arg == "--seed"
            opts[:seed] = parse(UInt64, val)
        elseif arg == "--out"
            opts[:out] = val
        elseif arg == "--smooth-tol"
            opts[:smooth_tol] = parse(Float64, val)
        elseif arg == "--real-atol"
            opts[:real_atol] = parse(Float64, val)
        elseif arg == "--gamma-retries"
            opts[:gamma_retries] = parse(Int, val)
        else
            error("Unrecognized option '$arg'.")
        end
        i += 2
    end

    opts[:start_checkpoint] === nothing && error("--start-checkpoint is required.")
    return opts
end

function sylvester_resultant_abs(a_asc, b_asc)
    n = length(a_asc) - 1
    m = length(b_asc) - 1
    a = reverse(a_asc)
    b = reverse(b_asc)
    S = zeros(Float64, n + m, n + m)
    for row in 1:m
        S[row, row:(row+n)] .= a
    end
    for row in 1:n
        S[m + row, row:(row+m)] .= b
    end
    return abs(det(S))
end

function eval_quartic_homogeneous(c_asc, theta)
    S = cos(theta)
    T = sin(theta)
    return sum(c_asc[m + 1] * S^m * T^(4 - m) for m in 0:4)
end

function unwrap_delta(a, b)
    d = b - a
    while d <= -pi
        d += 2pi
    end
    while d > pi
        d -= 2pi
    end
    return d
end

function topological_degree(target; samples = 4000)
    alpha = target[1:5]
    beta = target[6:10]
    previous = atan(-eval_quartic_homogeneous(beta, 0.0), eval_quartic_homogeneous(alpha, 0.0))
    total = 0.0
    for k in 1:samples
        theta = pi * k / samples
        current = atan(-eval_quartic_homogeneous(beta, theta), eval_quartic_homogeneous(alpha, theta))
        total += unwrap_delta(previous, current)
        previous = current
    end
    return round(Int, total / pi)
end

function affine_parameters(target, target_index)
    abs(target[target_index]) > 0 || error("Target coordinate $target_index is zero.")
    return [target[i] / target[target_index] for i in 1:length(target) if i != target_index]
end

function random_real_target(target_index)
    target = randn(10)
    while abs(target[target_index]) < 0.1
        target = randn(10)
    end
    return target
end

function classify_target(sys, starts, pstart, ptarget, opts)
    last_error = nothing
    for _ in 1:opts[:gamma_retries]
        gamma = cis(2pi * rand())
        try
            result = solve(
                sys,
                sys,
                starts;
                start_parameters = pstart,
                target_parameters = ptarget,
                gamma = gamma,
                show_progress = false,
                threading = true,
            )
            path_results = HC.results(result)
            success_results = filter(HC.is_success, path_results)
            sols = [HC.solution(r) for r in success_results]
            return (
                ok = length(sols) == 2,
                nreal = count(s -> SPR.solution_is_real(s; atol = opts[:real_atol], rtol = 0.0), sols),
                nsuccess = length(sols),
                result = result,
                gamma = gamma,
                error = nothing,
            )
        catch err
            last_error = sprint(showerror, err)
        end
    end
    return (ok = false, nreal = -1, nsuccess = 0, result = nothing, gamma = nothing, error = last_error)
end

function main(args = ARGS)
    opts = parse_args(args)
    Random.seed!(opts[:seed])

    base_state = SPR.read_checkpoint(opts[:start_checkpoint])
    problem = SPR.make_problem([2, 2]; pivot_columns = Vector{Int}(base_state["chart"]["pivot_columns"]))
    target_index = Int(base_state["target"]["target_index"])
    sys, _ = SPR.build_fiber_system(problem, target_index)
    starts = base_state["monodromy"]["solutions"]
    length(starts) == 2 || error("Start checkpoint must contain the two known solutions.")
    pstart = base_state["target"]["affine_parameters"]

    records = Any[]
    summary = Dict{Int,Dict{Int,Int}}()
    attempts = 0
    while length(records) < opts[:samples]
        attempts += 1
        target = random_real_target(target_index)
        resultant = sylvester_resultant_abs(target[1:5], target[6:10])
        resultant > opts[:smooth_tol] || continue

        ptarget = affine_parameters(target, target_index)
        classification = classify_target(sys, starts, pstart, ptarget, opts)
        classification.ok || continue

        tdeg = topological_degree(target)
        get!(summary, tdeg, Dict{Int,Int}())
        summary[tdeg][classification.nreal] = get(summary[tdeg], classification.nreal, 0) + 1

        record = Dict{String,Any}(
            "target_coefficients" => target,
            "target_index" => target_index,
            "affine_parameters" => ptarget,
            "quartic_resultant_abs" => resultant,
            "topological_degree" => tdeg,
            "nreal_preimages" => classification.nreal,
            "nsuccess" => classification.nsuccess,
            "gamma" => classification.gamma,
        )
        push!(records, record)
        println(
            "sample ", length(records),
            ": topdeg=", tdeg,
            " nreal=", classification.nreal,
            " resultant_abs=", resultant,
        )
    end

    state = Dict{String,Any}(
        "script" => "classify_O22_real_ramification.jl",
        "seed" => opts[:seed],
        "start_checkpoint" => opts[:start_checkpoint],
        "attempts" => attempts,
        "samples" => length(records),
        "summary" => summary,
        "records" => records,
    )

    mkpath(dirname(abspath(opts[:out])))
    open(opts[:out], "w") do io
        serialize(io, state)
    end

    println("summary:")
    for tdeg in sort(collect(keys(summary)))
        println("  topological degree ", tdeg, " => ", summary[tdeg])
    end
    println("wrote ", abspath(opts[:out]))
end

main()
