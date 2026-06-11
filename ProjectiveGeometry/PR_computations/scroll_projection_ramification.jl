#!/usr/bin/env julia

module ScrollProjectionRamification

using Dates
using HomotopyContinuation
using LinearAlgebra
using Printf
using Random
using Serialization
using SHA

const HC = HomotopyContinuation
const SCRIPT_VERSION = "0.1.0"

function usage()
    return """
    Usage:
      julia -t auto scroll_projection_ramification.jl --degrees 2,2 [options]
      julia -t auto scroll_projection_ramification.jl --resume checkpoint.jls [options]

    Required unless --resume is used:
      --degrees a1,a2,...,ar       Splitting type E = O(a1) + ... + O(ar), all ai > 0.

    Main options:
      --mode complex|real          complex: random complex basepoint. real: random real basepoint.
      --checkpoint PATH            Write checkpoint to PATH. Default is under ./checkpoints/.
      --resume PATH                Resume from a previous checkpoint with the same basepoint.
      --merge-checkpoints PATHS    Comma-separated compatible checkpoints; start from union of solutions.
      --seed N                     Random seed. Stored in checkpoints.
      --timeout SECONDS            HomotopyContinuation monodromy timeout.
      --target-solutions-count N   Stop once N solutions have been found.
      --max-loops-no-progress N    Monodromy no-progress stopping heuristic.
      --duplicate-check heuristic|certified
      --parameter-sampler complex|real
                                   complex is the default even in real mode; real is experimental.
      --real-atol TOL              Absolute tolerance for numerical realness.
      --real-rtol TOL              Relative tolerance for numerical realness.
      --pivot-columns i,j,...      One-based H0(E) basis columns used for the Grassmannian chart.
      --target-index i             One-based target coefficient used as affine target coordinate.
      --basepoint-tries N          Attempts to find a full-rank basepoint.
      --checkpoint-every-loops N   Checkpoint every N monodromy-loop callbacks.
      --compile true|false|mixed   Passed to HomotopyContinuation.
      --dry-run                    Build the system, choose/check the basepoint, save checkpoint, stop.
      --verify                     Run verify_solution_completeness after monodromy.
      --no-threading               Disable HomotopyContinuation path-tracking threads.
      --no-progress                Disable HomotopyContinuation progress display.
    """
end

parse_int_list(s::AbstractString) = [parse(Int, strip(x)) for x in split(s, ",") if !isempty(strip(x))]

function parse_bool_or_symbol(s::AbstractString)
    t = lowercase(strip(s))
    if t == "true"
        return true
    elseif t == "false"
        return false
    elseif t == "mixed"
        return :mixed
    else
        error("Expected true, false, or mixed; got '$s'.")
    end
end

function parse_duplicate_check(s::AbstractString)
    t = Symbol(lowercase(strip(s)))
    t in (:heuristic, :certified) || error("--duplicate-check must be heuristic or certified.")
    return t
end

function parse_mode(s::AbstractString)
    t = Symbol(lowercase(strip(s)))
    t in (:complex, :real) || error("--mode must be complex or real.")
    return t
end

function parse_sampler(s::AbstractString)
    t = Symbol(lowercase(strip(s)))
    t in (:complex, :real) || error("--parameter-sampler must be complex or real.")
    return t
end

function parse_args(args)
    opts = Dict{Symbol,Any}(
        :degrees => nothing,
        :mode => nothing,
        :checkpoint => nothing,
        :resume => nothing,
        :merge_checkpoints => String[],
        :seed => nothing,
        :timeout => nothing,
        :target_solutions_count => nothing,
        :max_loops_no_progress => 5,
        :duplicate_check => :heuristic,
        :parameter_sampler => :complex,
        :real_atol => 1e-6,
        :real_rtol => 0.0,
        :pivot_columns => nothing,
        :target_index => nothing,
        :basepoint_tries => 20,
        :checkpoint_every_loops => 1,
        :compile => :mixed,
        :dry_run => false,
        :verify => false,
        :threading => true,
        :show_progress => true,
        :rank_atol => 1e-8,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println(usage())
            exit(0)
        elseif arg in ("--dry-run",)
            opts[:dry_run] = true
            i += 1
            continue
        elseif arg in ("--verify",)
            opts[:verify] = true
            i += 1
            continue
        elseif arg in ("--no-threading",)
            opts[:threading] = false
            i += 1
            continue
        elseif arg in ("--no-progress",)
            opts[:show_progress] = false
            i += 1
            continue
        end

        key::String = arg
        value::String = ""
        if startswith(arg, "--") && occursin("=", arg)
            parts = split(arg, "=", limit = 2)
            key, value = parts[1], parts[2]
            i += 1
        elseif startswith(arg, "--")
            i == length(args) && error("Missing value for $arg.")
            value = args[i + 1]
            i += 2
        else
            error("Unrecognized argument '$arg'.")
        end

        if key == "--degrees"
            opts[:degrees] = parse_int_list(value)
        elseif key == "--mode"
            opts[:mode] = parse_mode(value)
        elseif key == "--checkpoint"
            opts[:checkpoint] = value
        elseif key == "--resume"
            opts[:resume] = value
        elseif key == "--merge-checkpoints"
            opts[:merge_checkpoints] = [strip(x) for x in split(value, ",") if !isempty(strip(x))]
        elseif key == "--seed"
            opts[:seed] = parse(UInt64, value)
        elseif key == "--timeout"
            opts[:timeout] = parse(Float64, value)
        elseif key == "--target-solutions-count"
            opts[:target_solutions_count] = parse(Int, value)
        elseif key == "--max-loops-no-progress"
            opts[:max_loops_no_progress] = parse(Int, value)
        elseif key == "--duplicate-check"
            opts[:duplicate_check] = parse_duplicate_check(value)
        elseif key == "--parameter-sampler"
            opts[:parameter_sampler] = parse_sampler(value)
        elseif key == "--real-atol"
            opts[:real_atol] = parse(Float64, value)
        elseif key == "--real-rtol"
            opts[:real_rtol] = parse(Float64, value)
        elseif key == "--pivot-columns"
            opts[:pivot_columns] = parse_int_list(value)
        elseif key == "--target-index"
            opts[:target_index] = parse(Int, value)
        elseif key == "--basepoint-tries"
            opts[:basepoint_tries] = parse(Int, value)
        elseif key == "--checkpoint-every-loops"
            opts[:checkpoint_every_loops] = parse(Int, value)
        elseif key == "--compile"
            opts[:compile] = parse_bool_or_symbol(value)
        elseif key == "--rank-atol"
            opts[:rank_atol] = parse(Float64, value)
        else
            error("Unrecognized option '$key'.")
        end
    end
    return opts
end

function validate_degrees(degrees)
    isempty(degrees) && error("The splitting type cannot be empty.")
    all(>(0), degrees) || error("All degrees ai in E = sum O(ai) must be positive.")
    return degrees
end

function basis_sections(degrees)
    basis = Tuple{Int,Int}[]
    labels = String[]
    for (j, a) in enumerate(degrees)
        for m in 0:a
            push!(basis, (j, m))
            push!(labels, "summand $j, t^$m")
        end
    end
    return basis, labels
end

function default_checkpoint_path(degrees, mode)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    degstr = join(degrees, "_")
    return joinpath(pwd(), "checkpoints", "scroll_PR_O$(degstr)_$(mode)_$(stamp).jls")
end

zero_like(x) = x - x
one_like(x) = zero_like(x) + 1

function det_expansion(A)
    n, m = size(A)
    n == m || error("det_expansion requires a square matrix.")
    n == 0 && error("0 by 0 determinant is not used in this script.")

    z = zero_like(A[1, 1])
    o = one_like(A[1, 1])
    dp = Dict{Int,Any}(0 => o)
    for i in 1:n
        ndp = Dict{Int,Any}()
        for (mask, val) in dp
            for j in 1:n
                bit = 1 << (j - 1)
                (mask & bit) == 0 || continue
                inversions = 0
                for q in (j + 1):n
                    inversions += ((mask & (1 << (q - 1))) != 0) ? 1 : 0
                end
                signed_val = isodd(inversions) ? -val : val
                newmask = mask | bit
                ndp[newmask] = get(ndp, newmask, z) + signed_val * A[i, j]
            end
        end
        dp = ndp
    end
    return dp[(1 << n) - 1]
end

function grassmann_chart_matrix(xvars, k, n, pivot_columns)
    length(pivot_columns) == k || error("Expected $k pivot columns.")
    length(unique(pivot_columns)) == k || error("Pivot columns must be distinct.")
    all(c -> 1 <= c <= n, pivot_columns) || error("Pivot columns must lie in 1:$n.")
    length(xvars) == k * (n - k) || error("Wrong number of Grassmannian chart variables.")

    z = zero_like(xvars[1])
    G = fill(z, k, n)
    for row in 1:k
        G[row, pivot_columns[row]] = one_like(z)
    end

    nonpivots = [c for c in 1:n if !(c in pivot_columns)]
    pos = 1
    for col in nonpivots
        for row in 1:k
            G[row, col] = xvars[pos]
            pos += 1
        end
    end
    return G, nonpivots
end

function numeric_grassmann_matrix(x0, k, n, pivot_columns)
    T = promote_type(eltype(x0), ComplexF64)
    G = zeros(T, k, n)
    for row in 1:k
        G[row, pivot_columns[row]] = one(T)
    end
    nonpivots = [c for c in 1:n if !(c in pivot_columns)]
    pos = 1
    for col in nonpivots
        for row in 1:k
            G[row, col] = x0[pos]
            pos += 1
        end
    end
    return G
end

function evaluation_matrix(G, basis, tvar, r)
    k, _ = size(G)
    z = zero_like(G[1, 1] + tvar)
    M = fill(z, k, r)
    for col in 1:length(basis)
        j, power = basis[col]
        tp = power == 0 ? one_like(z) : tvar^power
        for row in 1:k
            M[row, j] = M[row, j] + G[row, col] * tp
        end
    end
    return M
end

function dense_univariate_coefficients(f, tvar, maxdeg, zero_expr)
    fexp = HC.expand(f)
    out = [zero_expr for _ in 0:maxdeg]
    iszero(fexp) && return out

    exps, coeffs = HC.exponents_coefficients(fexp, [tvar]; expanded = true, unpack_coeffs = false)
    for col in 1:size(exps, 2)
        deg = Int(exps[1, col])
        coeff = zero_expr + coeffs[col]
        if deg <= maxdeg
            out[deg + 1] = out[deg + 1] + coeff
        elseif !iszero(coeff)
            error("A t^$deg coefficient survived, but the expected maximum degree is $maxdeg.")
        end
    end
    return out
end

function ramification_coefficients(degrees, xvars, pivot_columns)
    r = length(degrees)
    k = r + 1
    d = sum(degrees)
    basis, basis_labels = basis_sections(degrees)
    n = length(basis)

    tvar = HC.variables(:t, 1:1)[1]
    G, nonpivots = grassmann_chart_matrix(xvars, k, n, pivot_columns)
    M = evaluation_matrix(G, basis, tvar, r)

    cofactors = Vector{Any}(undef, k)
    for row in 1:k
        rows = [i for i in 1:k if i != row]
        minor = M[rows, :]
        sign = isodd(row - 1) ? -1 : 1
        cofactors[row] = sign * det_expansion(minor)
    end

    z = zero_like(xvars[1])
    coeffs = typeof(z)[]
    target_labels = String[]
    for j in 1:r
        local_section = z
        for row in 1:k
            local_section = local_section + cofactors[row] * HC.differentiate(M[row, j], tvar)
        end
        maxdeg = degrees[j] + d - 2
        local_coeffs = dense_univariate_coefficients(local_section, tvar, maxdeg, z)
        append!(coeffs, local_coeffs)
        for m in 0:maxdeg
            push!(target_labels, "summand $j, t^$m")
        end
    end

    return coeffs, target_labels, basis, basis_labels, nonpivots
end

function make_problem(degrees; pivot_columns = nothing)
    validate_degrees(degrees)
    r = length(degrees)
    d = sum(degrees)
    k = r + 1
    n = d + r
    domain_dim = k * (n - k)
    target_dim = sum(a + d - 1 for a in degrees)
    target_dim == domain_dim + 1 ||
        error("Dimension check failed: target vector dimension $target_dim, domain dimension $domain_dim.")

    pivots = isnothing(pivot_columns) ? collect(1:k) : collect(pivot_columns)
    length(pivots) == k || error("Need exactly r+1 = $k pivot columns.")
    all(c -> 1 <= c <= n, pivots) || error("Pivot columns must be between 1 and $n.")

    xvars = HC.variables(:x, 1:domain_dim)
    pvars = HC.variables(:p, 1:domain_dim)
    Rcoeffs, target_labels, basis, basis_labels, nonpivots =
        ramification_coefficients(degrees, xvars, pivots)
    length(Rcoeffs) == target_dim || error("Internal target coefficient count mismatch.")

    Rsys = System(Rcoeffs; variables = xvars)
    return (
        degrees = collect(degrees),
        rank = r,
        total_degree = d,
        k = k,
        h0 = n,
        domain_dim = domain_dim,
        target_dim = target_dim,
        pivot_columns = pivots,
        nonpivot_columns = nonpivots,
        basis = basis,
        basis_labels = basis_labels,
        target_labels = target_labels,
        xvars = xvars,
        pvars = pvars,
        Rcoeffs = Rcoeffs,
        Rsys = Rsys,
    )
end

function build_fiber_system(problem, target_index)
    1 <= target_index <= problem.target_dim || error("Target index must lie in 1:$(problem.target_dim).")
    indices = [i for i in 1:problem.target_dim if i != target_index]
    length(indices) == problem.domain_dim || error("Affine target index count mismatch.")
    F = [problem.Rcoeffs[idx] - problem.pvars[j] * problem.Rcoeffs[target_index]
         for (j, idx) in enumerate(indices)]
    sys = System(F; variables = problem.xvars, parameters = problem.pvars)
    return sys, indices
end

function evaluate_target(problem, x0)
    return vec(HC.evaluate(problem.Rsys, x0))
end

function affine_parameters(target, target_index, affine_indices)
    denom = target[target_index]
    abs(denom) > 0 || error("Chosen target coordinate is zero.")
    return [target[idx] / denom for idx in affine_indices]
end

function sample_chart_point(n, mode)
    if mode == :real
        return randn(Float64, n)
    elseif mode == :complex
        return randn(ComplexF64, n)
    else
        error("Unknown mode $mode.")
    end
end

function jacobian_rank(sys, x0, p0; atol = 1e-8)
    J = HC.jacobian(sys, x0, p0)
    return rank(Matrix{ComplexF64}(J); atol = atol), norm(HC.evaluate(sys, x0, p0), Inf)
end

function choose_basepoint(problem, opts)
    best = nothing
    for attempt in 1:opts[:basepoint_tries]
        x0 = sample_chart_point(problem.domain_dim, opts[:mode])
        target = evaluate_target(problem, x0)
        target_index = something(opts[:target_index], argmax(abs.(target)))
        if abs(target[target_index]) == 0
            continue
        end
        sys, affine_indices = build_fiber_system(problem, target_index)
        p0 = affine_parameters(target, target_index, affine_indices)
        rk, residual = jacobian_rank(sys, x0, p0; atol = opts[:rank_atol])
        candidate = (
            x0 = x0,
            target = target,
            target_index = target_index,
            affine_indices = affine_indices,
            p0 = p0,
            sys = sys,
            rank = rk,
            residual = residual,
            attempt = attempt,
        )
        best = candidate
        if rk == problem.domain_dim
            return candidate
        end
        @warn "Basepoint attempt $attempt has Jacobian rank $rk < $(problem.domain_dim); trying again."
    end
    @warn "No full-rank basepoint was found. The PR map may fail to be generically finite for this splitting type, or the chart/basepoint may be unlucky."
    return best
end

function solution_is_real(sol; atol = 1e-6, rtol = 0.0)
    isempty(sol) && return true
    imnorm = maximum(abs, imag.(sol))
    imnorm < atol && return true
    iszero(rtol) && return false
    return imnorm < rtol * norm(sol, 1)
end

function complex_hash(v)
    io = IOBuffer()
    for z in vec(v)
        @printf(io, "%.17e,%.17e;", real(z), imag(z))
    end
    return bytes2hex(sha256(take!(io)))
end

function compatibility(problem, basepoint, checkpoint_mode)
    G0 = numeric_grassmann_matrix(basepoint.x0, problem.k, problem.h0, problem.pivot_columns)
    target_hash = complex_hash(basepoint.target)
    basepoint_hash = complex_hash(basepoint.x0)
    chart_hash = bytes2hex(sha256(join(problem.pivot_columns, ",")))
    signature_text = join([
        SCRIPT_VERSION,
        join(problem.degrees, ","),
        checkpoint_mode,
        join(problem.pivot_columns, ","),
        string(basepoint.target_index),
        basepoint_hash,
        target_hash,
    ], "|")
    return Dict{String,Any}(
        "signature" => bytes2hex(sha256(signature_text)),
        "basepoint_hash" => basepoint_hash,
        "target_hash" => target_hash,
        "chart_hash" => chart_hash,
        "grassmannian_matrix_hash" => complex_hash(G0),
    )
end

function path_result_summary(r, opts)
    return Dict{String,Any}(
        "return_code" => string(r.return_code),
        "is_success" => HC.is_success(r),
        "is_finite" => HC.is_finite(r),
        "is_singular" => HC.is_singular(r),
        "is_real" => HC.is_real(r; atol = opts[:real_atol], rtol = opts[:real_rtol]),
        "accuracy" => HC.accuracy(r),
        "residual" => HC.residual(r),
        "multiplicity" => HC.multiplicity(r),
        "solution" => HC.solution(r),
    )
end

function extract_result_data(result, partial_results, opts)
    if result === nothing
        raw_results = partial_results
        sols = [HC.solution(r) for r in raw_results if HC.is_success(r)]
        returncode = "in_progress"
        nsolutions = length(sols)
        nresults = length(raw_results)
        result_seed = nothing
        trace_value = nothing
    else
        raw_results = HC.results(result)
        sols = HC.solutions(result)
        returncode = string(result.returncode)
        nsolutions = HC.nsolutions(result)
        nresults = HC.nresults(result)
        result_seed = HC.seed(result)
        trace_value = result.trace
    end

    real_flags = [solution_is_real(s; atol = opts[:real_atol], rtol = opts[:real_rtol]) for s in sols]
    return Dict{String,Any}(
        "return_code" => returncode,
        "nsolutions" => nsolutions,
        "nresults" => nresults,
        "seed" => result_seed,
        "trace" => trace_value,
        "solutions" => sols,
        "solution_real_flags" => real_flags,
        "nreal_solutions" => count(identity, real_flags),
        "path_results" => [path_result_summary(r, opts) for r in raw_results],
    )
end

function record_known_solutions!(state, sols, opts)
    real_flags = [solution_is_real(s; atol = opts[:real_atol], rtol = opts[:real_rtol]) for s in sols]
    state["monodromy"]["solutions"] = sols
    state["monodromy"]["solution_real_flags"] = real_flags
    state["monodromy"]["nsolutions"] = length(sols)
    state["monodromy"]["nreal_solutions"] = count(identity, real_flags)
    return state
end

function make_state(problem, basepoint, opts; result = nothing, partial_results = Any[], status = "in_progress")
    G0 = numeric_grassmann_matrix(basepoint.x0, problem.k, problem.h0, problem.pivot_columns)
    mode = string(opts[:mode])
    return Dict{String,Any}(
        "script_version" => SCRIPT_VERSION,
        "status" => status,
        "written_at" => string(now()),
        "problem" => Dict{String,Any}(
            "degrees" => problem.degrees,
            "rank" => problem.rank,
            "total_degree" => problem.total_degree,
            "h0_dimension" => problem.h0,
            "domain_dimension" => problem.domain_dim,
            "target_vector_dimension" => problem.target_dim,
            "target_projective_dimension" => problem.target_dim - 1,
        ),
        "chart" => Dict{String,Any}(
            "pivot_columns" => problem.pivot_columns,
            "nonpivot_columns" => problem.nonpivot_columns,
            "basis_labels" => problem.basis_labels,
        ),
        "basepoint" => Dict{String,Any}(
            "mode" => mode,
            "chart_coordinates" => basepoint.x0,
            "grassmannian_matrix" => G0,
            "jacobian_rank" => basepoint.rank,
            "fiber_equation_residual_at_basepoint" => basepoint.residual,
            "attempt" => basepoint.attempt,
        ),
        "target" => Dict{String,Any}(
            "target_index" => basepoint.target_index,
            "target_label" => problem.target_labels[basepoint.target_index],
            "target_labels" => problem.target_labels,
            "coefficients" => basepoint.target,
            "affine_indices" => basepoint.affine_indices,
            "affine_parameters" => basepoint.p0,
        ),
        "monodromy_options" => Dict{String,Any}(
            "mode" => mode,
            "parameter_sampler" => string(opts[:parameter_sampler]),
            "timeout" => opts[:timeout],
            "target_solutions_count" => opts[:target_solutions_count],
            "max_loops_no_progress" => opts[:max_loops_no_progress],
            "duplicate_check" => string(opts[:duplicate_check]),
            "compile" => string(opts[:compile]),
            "threading" => opts[:threading],
            "show_progress" => opts[:show_progress],
            "seed" => opts[:seed],
        ),
        "reality" => Dict{String,Any}(
            "atol" => opts[:real_atol],
            "rtol" => opts[:real_rtol],
            "note" => "Reality is numerical: max(abs(imag(solution))) < atol, or the analogous rtol test.",
        ),
        "compatibility" => compatibility(problem, basepoint, mode),
        "monodromy" => extract_result_data(result, partial_results, opts),
    )
end

function manifest_path(path)
    return path * ".manifest.txt"
end

function write_manifest(path, state)
    open(manifest_path(path), "w") do io
        println(io, "script_version = ", state["script_version"])
        println(io, "status = ", state["status"])
        println(io, "written_at = ", state["written_at"])
        println(io, "degrees = ", state["problem"]["degrees"])
        println(io, "rank = ", state["problem"]["rank"])
        println(io, "total_degree = ", state["problem"]["total_degree"])
        println(io, "domain_dimension = ", state["problem"]["domain_dimension"])
        println(io, "target_projective_dimension = ", state["problem"]["target_projective_dimension"])
        println(io, "pivot_columns = ", state["chart"]["pivot_columns"])
        println(io, "target_index = ", state["target"]["target_index"])
        println(io, "target_label = \"", state["target"]["target_label"], "\"")
        println(io, "mode = \"", state["basepoint"]["mode"], "\"")
        println(io, "jacobian_rank = ", state["basepoint"]["jacobian_rank"])
        println(io, "signature = \"", state["compatibility"]["signature"], "\"")
        println(io, "basepoint_hash = \"", state["compatibility"]["basepoint_hash"], "\"")
        println(io, "target_hash = \"", state["compatibility"]["target_hash"], "\"")
        println(io, "return_code = \"", state["monodromy"]["return_code"], "\"")
        println(io, "nsolutions = ", state["monodromy"]["nsolutions"])
        println(io, "nreal_solutions = ", state["monodromy"]["nreal_solutions"])
    end
end

function write_checkpoint(path, state)
    abs_path = abspath(path)
    mkpath(dirname(abs_path))
    tmp = abs_path * ".tmp"
    open(tmp, "w") do io
        serialize(io, state)
    end
    mv(tmp, abs_path; force = true)
    write_manifest(abs_path, state)
    return abs_path
end

function read_checkpoint(path)
    open(path, "r") do io
        return deserialize(io)
    end
end

function load_basepoint_from_state(problem, state, opts)
    degrees = Vector{Int}(state["problem"]["degrees"])
    degrees == problem.degrees || error("Checkpoint degrees $(degrees) do not match requested degrees $(problem.degrees).")
    pivots = Vector{Int}(state["chart"]["pivot_columns"])
    pivots == problem.pivot_columns || error("Checkpoint pivot columns $(pivots) do not match requested pivot columns $(problem.pivot_columns).")

    target_index = Int(state["target"]["target_index"])
    sys, affine_indices = build_fiber_system(problem, target_index)
    x0 = state["basepoint"]["chart_coordinates"]
    target = state["target"]["coefficients"]
    p0 = state["target"]["affine_parameters"]
    rk, residual = jacobian_rank(sys, x0, p0; atol = opts[:rank_atol])
    return (
        x0 = x0,
        target = target,
        target_index = target_index,
        affine_indices = affine_indices,
        p0 = p0,
        sys = sys,
        rank = rk,
        residual = residual,
        attempt = Int(get(state["basepoint"], "attempt", 0)),
    )
end

function checkpoint_start_solutions(state)
    sols = state["monodromy"]["solutions"]
    if isempty(sols)
        return [state["basepoint"]["chart_coordinates"]]
    else
        return sols
    end
end

function unique_solution_vectors(solutions; atol = 1e-8)
    unique = Vector{Any}()
    for sol in solutions
        duplicate = false
        for known in unique
            if length(sol) == length(known) && norm(sol - known, Inf) < atol
                duplicate = true
                break
            end
        end
        duplicate || push!(unique, sol)
    end
    return unique
end

function compatible_signature(state)
    return state["compatibility"]["signature"]
end

function merge_checkpoint_solutions(paths, active_state; atol = 1e-8)
    isempty(paths) && return checkpoint_start_solutions(active_state)
    signature = compatible_signature(active_state)
    sols = Any[]
    append!(sols, checkpoint_start_solutions(active_state))
    for path in paths
        state = read_checkpoint(path)
        compatible_signature(state) == signature ||
            error("Checkpoint '$path' is not basepoint-compatible with the active checkpoint.")
        append!(sols, checkpoint_start_solutions(state))
    end
    return unique_solution_vectors(sols; atol = atol)
end

function parameter_sampler(kind)
    if kind == :complex
        return p -> randn(ComplexF64, length(p))
    elseif kind == :real
        return p -> randn(Float64, length(p))
    else
        error("Unknown parameter sampler $kind.")
    end
end

function print_setup_summary(problem, basepoint, opts, checkpoint)
    println("Scroll projection-ramification setup")
    println("  E = ", join(["O($(a))" for a in problem.degrees], " + "))
    println("  rank(E) = ", problem.rank, ", deg(E) = ", problem.total_degree)
    println("  dim Gr(r+1,H0(E)) = ", problem.domain_dim)
    println("  dim |K_PE + (r+1)H| = ", problem.target_dim - 1)
    println("  Grassmannian chart pivot columns = ", problem.pivot_columns)
    println("  target affine coordinate index = ", basepoint.target_index, " (", problem.target_labels[basepoint.target_index], ")")
    println("  basepoint mode = ", opts[:mode])
    println("  Jacobian rank at basepoint = ", basepoint.rank, " / ", problem.domain_dim)
    println("  checkpoint = ", abspath(checkpoint))
    if basepoint.rank < problem.domain_dim
        println("  WARNING: rank is deficient; the fiber may be positive-dimensional in this chart.")
    end
end

function monodromy_kwargs(opts, checkpoint_path, problem, basepoint)
    callback_counter = Ref(0)
    callback = function (args...)
        callback_counter[] += 1
        if callback_counter[] % opts[:checkpoint_every_loops] == 0
            partial_results = isempty(args) ? Any[] : args[1]
            state = make_state(problem, basepoint, opts; partial_results = partial_results, status = "in_progress")
            write_checkpoint(checkpoint_path, state)
        end
        return false
    end

    kws = Dict{Symbol,Any}(
        :seed => UInt32(mod(opts[:seed], UInt64(typemax(UInt32)))),
        :show_progress => opts[:show_progress],
        :threading => opts[:threading],
        :compile => opts[:compile],
        :max_loops_no_progress => opts[:max_loops_no_progress],
        :duplicate_check => opts[:duplicate_check],
        :parameter_sampler => parameter_sampler(opts[:parameter_sampler]),
        :loop_finished_callback => callback,
    )
    opts[:timeout] === nothing || (kws[:timeout] = opts[:timeout])
    opts[:target_solutions_count] === nothing ||
        (kws[:target_solutions_count] = opts[:target_solutions_count])
    return kws
end

function run(opts)
    if opts[:resume] === nothing && !isempty(opts[:merge_checkpoints])
        opts[:resume] = first(opts[:merge_checkpoints])
        opts[:merge_checkpoints] = opts[:merge_checkpoints][2:end]
    end

    if opts[:resume] === nothing
        opts[:degrees] === nothing && error("--degrees is required unless --resume is used.")
        opts[:mode] === nothing && (opts[:mode] = :complex)
    else
        state = read_checkpoint(opts[:resume])
        opts[:degrees] === nothing && (opts[:degrees] = Vector{Int}(state["problem"]["degrees"]))
        opts[:mode] === nothing && (opts[:mode] = Symbol(state["basepoint"]["mode"]))
        opts[:pivot_columns] === nothing && (opts[:pivot_columns] = Vector{Int}(state["chart"]["pivot_columns"]))
    end

    opts[:seed] === nothing && (opts[:seed] = UInt64(time_ns()))
    Random.seed!(opts[:seed])
    degrees = validate_degrees(Vector{Int}(opts[:degrees]))
    checkpoint_path = something(opts[:checkpoint], default_checkpoint_path(degrees, opts[:mode]))

    problem = make_problem(degrees; pivot_columns = opts[:pivot_columns])
    if opts[:resume] === nothing
        basepoint = choose_basepoint(problem, opts)
        start_solutions = [basepoint.x0]
    else
        resume_state = read_checkpoint(opts[:resume])
        basepoint = load_basepoint_from_state(problem, resume_state, opts)
        start_solutions = merge_checkpoint_solutions(opts[:merge_checkpoints], resume_state; atol = opts[:real_atol])
    end

    print_setup_summary(problem, basepoint, opts, checkpoint_path)
    println("  start solutions supplied to monodromy = ", length(start_solutions))
    initial_state = make_state(problem, basepoint, opts; status = "initialized")
    record_known_solutions!(initial_state, start_solutions, opts)
    write_checkpoint(checkpoint_path, initial_state)

    if opts[:dry_run]
        println("Dry run requested; wrote initialized checkpoint and stopped.")
        return nothing
    end

    kws = monodromy_kwargs(opts, checkpoint_path, problem, basepoint)
    result = HC.monodromy_solve(basepoint.sys, start_solutions, basepoint.p0; kws...)
    final_state = make_state(problem, basepoint, opts; result = result, status = "finished")
    write_checkpoint(checkpoint_path, final_state)

    println()
    println("Monodromy finished")
    println("  return code = ", final_state["monodromy"]["return_code"])
    println("  solutions = ", final_state["monodromy"]["nsolutions"])
    println("  numerically real solutions = ", final_state["monodromy"]["nreal_solutions"],
            " (atol=", opts[:real_atol], ", rtol=", opts[:real_rtol], ")")
    println("  checkpoint = ", abspath(checkpoint_path))
    println("  manifest = ", manifest_path(abspath(checkpoint_path)))

    if opts[:verify]
        println()
        println("Running verify_solution_completeness...")
        ok = HC.verify_solution_completeness(basepoint.sys, result)
        println("  verify_solution_completeness returned: ", ok)
    end
    return result
end

function main(args = ARGS)
    try
        opts = parse_args(args)
        run(opts)
    catch err
        println(stderr, "ERROR: ", sprint(showerror, err))
        println(stderr)
        println(stderr, usage())
        rethrow()
    end
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    ScrollProjectionRamification.main(ARGS)
end
