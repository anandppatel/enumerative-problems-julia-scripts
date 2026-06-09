#!/usr/bin/env julia

using Dates
using HomotopyContinuation
using LinearAlgebra
using Printf
using Random
using Serialization

const CHECKPOINT_VERSION = 1
const N_QUADRATIC = 6
const N_CUBIC = 10
const N_SEXTIC = 28
const DOMAIN_DIM = N_QUADRATIC + N_CUBIC
const ROOT_ACTION_ORDER = 6

struct RunConfig
    seed::UInt32
    checkpoint::String
    resume::Bool
    use_group_action::Bool
    monodromy_seed::Union{Nothing,UInt32}
    target_orbits::Union{Nothing,Int}
    timeout::Union{Nothing,Float64}
    max_loops_no_progress::Int
    checkpoint_interval::Float64
    show_progress::Bool
    threading::Bool
    duplicate_check::Symbol
    compile
    rank_check::Bool
end

function usage()
    println("""
    Usage:
      julia ternary_sextic_two_powers_monodromy.jl [options]

    Computes the degree of the locus of ternary sextics expressible as

        F = A^3 + B^2,

    where A is a ternary quadratic and B is a ternary cubic.  The script
    intersects the affine cone with a random affine 16-plane and uses
    HomotopyContinuation.jl monodromy to count the fiber.  By default it
    quotients by the six root symmetries A -> zeta_3 A and B -> +/- B, so
    the reported orbit count is the expected projective degree.

    Options:
      --checkpoint PATH          .jls checkpoint path
      --resume                   resume from checkpoint if it exists
      --no-group-action          do not quotient by root-of-unity actions
      --monodromy-seed N         random seed for monodromy loops
      --target-orbits N          stop once N orbit representatives are found
      --timeout SECONDS          stop after this many seconds
      --max-loops-no-progress N  default: 40
      --checkpoint-interval SEC  default: 30
      --seed N                   default: 20260523
      --threading true|false     default: true
      --show-progress true|false default: true
      --duplicate-check heuristic|certified  default: heuristic
      --compile true|false       default: true
      --rank-check true|false    default: true
    """)
end

function parse_bool(s::AbstractString)::Bool
    t = lowercase(s)
    t in ("true", "yes", "1", "on") && return true
    t in ("false", "no", "0", "off") && return false
    error("Expected boolean value, got '$s'.")
end

function parse_cli(argv)::RunConfig
    opts = Dict{String,String}()
    flags = Set{String}()
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg in ("-h", "--help")
            usage()
            exit(0)
        elseif arg in ("--resume", "--no-group-action")
            push!(flags, arg)
            i += 1
        elseif startswith(arg, "--")
            i == length(argv) && error("Missing value after $arg.")
            opts[arg] = argv[i + 1]
            i += 2
        else
            error("Unexpected argument: $arg")
        end
    end

    default_checkpoint = joinpath(@__DIR__, "ternary_sextic_two_powers_monodromy_checkpoint.jls")

    duplicate_check = Symbol(get(opts, "--duplicate-check", "heuristic"))
    duplicate_check in (:heuristic, :certified) ||
        error("--duplicate-check must be heuristic or certified.")

    RunConfig(
        UInt32(parse(UInt64, get(opts, "--seed", "20260523")) % (UInt64(typemax(UInt32)) + 1)),
        get(opts, "--checkpoint", default_checkpoint),
        "--resume" in flags,
        !("--no-group-action" in flags),
        haskey(opts, "--monodromy-seed") ?
            UInt32(parse(UInt64, opts["--monodromy-seed"]) % (UInt64(typemax(UInt32)) + 1)) :
            nothing,
        haskey(opts, "--target-orbits") ? parse(Int, opts["--target-orbits"]) : nothing,
        haskey(opts, "--timeout") ? parse(Float64, opts["--timeout"]) : nothing,
        parse(Int, get(opts, "--max-loops-no-progress", "40")),
        parse(Float64, get(opts, "--checkpoint-interval", "30")),
        parse_bool(get(opts, "--show-progress", "true")),
        parse_bool(get(opts, "--threading", "true")),
        duplicate_check,
        parse_bool(get(opts, "--compile", "true")),
        parse_bool(get(opts, "--rank-check", "true")),
    )
end

function ternary_monomials(degree::Int)
    exps = NTuple{3,Int}[]
    for i in degree:-1:0
        for j in (degree - i):-1:0
            push!(exps, (i, j, degree - i - j))
        end
    end
    exps
end

const QUADRATIC_MONOMIALS = ternary_monomials(2)
const CUBIC_MONOMIALS = ternary_monomials(3)
const SEXTIC_MONOMIALS = ternary_monomials(6)

function add_exp(a::NTuple{3,Int}, b::NTuple{3,Int})
    (a[1] + b[1], a[2] + b[2], a[3] + b[3])
end

function coefficient_dict(coeffs, monomials)
    Dict(monomials[i] => coeffs[i] for i in eachindex(monomials))
end

function multiply_polys(p::Dict, q::Dict)
    out = Dict{NTuple{3,Int},Any}()
    for (e1, c1) in p, (e2, c2) in q
        e = add_exp(e1, e2)
        value = c1 * c2
        out[e] = haskey(out, e) ? out[e] + value : value
    end
    out
end

function power_poly(coeffs, monomials, exponent::Int)
    exponent < 0 && error("Polynomial exponent must be nonnegative.")
    one_value = one(coeffs[1])
    result = Dict{NTuple{3,Int},Any}((0, 0, 0) => one_value)
    base = coefficient_dict(coeffs, monomials)
    for _ in 1:exponent
        result = multiply_polys(result, base)
    end
    result
end

function coefficient_vector(poly::Dict, target_monomials, zero_value)
    [haskey(poly, exp) ? poly[exp] : zero_value for exp in target_monomials]
end

function image_coefficients(a_coeffs, b_coeffs)
    a3 = power_poly(a_coeffs, QUADRATIC_MONOMIALS, 3)
    b2 = power_poly(b_coeffs, CUBIC_MONOMIALS, 2)
    zero_value = zero(a_coeffs[1])
    coefficient_vector(a3, SEXTIC_MONOMIALS, zero_value) .+
        coefficient_vector(b2, SEXTIC_MONOMIALS, zero_value)
end

function random_complex_vector(rng, n::Int)
    randn(rng, ComplexF64, n)
end

function random_complex_matrix(rng, rows::Int, cols::Int)
    randn(rng, ComplexF64, rows, cols)
end

function build_system(projection_matrix, projection_offset)
    @var x[1:DOMAIN_DIM]
    @var y[1:DOMAIN_DIM]

    a_coeffs = x[1:N_QUADRATIC]
    b_coeffs = x[(N_QUADRATIC + 1):DOMAIN_DIM]
    image = image_coefficients(a_coeffs, b_coeffs)

    equations = [
        sum(projection_matrix[i, j] * image[j] for j in 1:N_SEXTIC) +
        projection_offset[i] - y[i]
        for i in 1:DOMAIN_DIM
    ]

    System(equations; variables = x, parameters = y)
end

function projected_parameter(projection_matrix, projection_offset, solution)
    a_coeffs = solution[1:N_QUADRATIC]
    b_coeffs = solution[(N_QUADRATIC + 1):DOMAIN_DIM]
    image = ComplexF64.(image_coefficients(a_coeffs, b_coeffs))
    projection_matrix * image + projection_offset
end

function coefficient_vector_from_dict(poly::Dict)
    ComplexF64.(coefficient_vector(poly, SEXTIC_MONOMIALS, 0.0 + 0.0im))
end

function coefficient_map_jacobian(solution)
    a_coeffs = solution[1:N_QUADRATIC]
    b_coeffs = solution[(N_QUADRATIC + 1):DOMAIN_DIM]
    a2 = power_poly(a_coeffs, QUADRATIC_MONOMIALS, 2)
    b1 = coefficient_dict(b_coeffs, CUBIC_MONOMIALS)

    jac = Matrix{ComplexF64}(undef, N_SEXTIC, DOMAIN_DIM)

    for i in 1:N_QUADRATIC
        basis = Dict{NTuple{3,Int},Any}(QUADRATIC_MONOMIALS[i] => 3.0 + 0.0im)
        jac[:, i] = coefficient_vector_from_dict(multiply_polys(a2, basis))
    end

    for j in 1:N_CUBIC
        basis = Dict{NTuple{3,Int},Any}(CUBIC_MONOMIALS[j] => 2.0 + 0.0im)
        jac[:, N_QUADRATIC + j] = coefficient_vector_from_dict(multiply_polys(b1, basis))
    end

    jac
end

function rank_summary(solution)
    singular_values = svdvals(coefficient_map_jacobian(solution))
    tolerance = maximum(size(coefficient_map_jacobian(solution))) * eps(Float64) * maximum(singular_values)
    rank = count(>(tolerance), singular_values)
    (; rank, tolerance, smallest = minimum(singular_values), largest = maximum(singular_values))
end

function new_instance(config::RunConfig)
    rng = MersenneTwister(config.seed)
    projection_matrix = random_complex_matrix(rng, DOMAIN_DIM, N_SEXTIC)
    projection_offset = random_complex_vector(rng, DOMAIN_DIM)
    start_solution = random_complex_vector(rng, DOMAIN_DIM)
    start_parameter = projected_parameter(projection_matrix, projection_offset, start_solution)

    Dict{Symbol,Any}(
        :version => CHECKPOINT_VERSION,
        :problem => :ternary_sextic_two_powers,
        :created_at => string(now()),
        :updated_at => string(now()),
        :seed => config.seed,
        :projection_matrix => projection_matrix,
        :projection_offset => projection_offset,
        :start_parameter => start_parameter,
        :solutions => [start_solution],
        :n_orbit_representatives => 1,
        :raw_solution_multiple => ROOT_ACTION_ORDER,
        :return_code => :start_pair,
        :tracked_loops => 0,
    )
end

function save_checkpoint(path::AbstractString, data::Dict{Symbol,Any})
    data[:updated_at] = string(now())
    tmp = path * ".tmp"
    serialize(tmp, data)
    mv(tmp, path; force = true)
    nothing
end

function load_or_create_checkpoint(config::RunConfig)
    if config.resume && isfile(config.checkpoint)
        data = deserialize(config.checkpoint)
        data[:version] == CHECKPOINT_VERSION ||
            error("Checkpoint version $(data[:version]) is not supported.")
        data[:problem] == :ternary_sextic_two_powers ||
            error("Checkpoint problem $(data[:problem]) is not ternary_sextic_two_powers.")
        return data
    end

    data = new_instance(config)
    save_checkpoint(config.checkpoint, data)
    return data
end

function root_group_action()
    function action(solution)
        transformed = Vector{Vector{ComplexF64}}()
        for ia in 0:2, ib in 0:1
            ia == 0 && ib == 0 && continue
            a_scale = cis(2 * pi * ia / 3)
            b_scale = ib == 0 ? 1.0 + 0.0im : -1.0 + 0.0im
            s = Vector{ComplexF64}(solution)
            s[1:N_QUADRATIC] .*= a_scale
            s[(N_QUADRATIC + 1):DOMAIN_DIM] .*= b_scale
            push!(transformed, s)
        end
        Tuple(transformed)
    end
    action
end

function root_group_orbit(solution)
    transformed = Vector{Vector{ComplexF64}}()
    for ia in 0:2, ib in 0:1
        a_scale = cis(2 * pi * ia / 3)
        b_scale = ib == 0 ? 1.0 + 0.0im : -1.0 + 0.0im
        s = Vector{ComplexF64}(solution)
        s[1:N_QUADRATIC] .*= a_scale
        s[(N_QUADRATIC + 1):DOMAIN_DIM] .*= b_scale
        push!(transformed, s)
    end
    transformed
end

function solution_key(solution; digits::Int = 8)
    orbit_keys = Vector{NTuple{2 * DOMAIN_DIM,Int}}()
    scale = 10.0^digits
    for s in root_group_orbit(solution)
        values = Int[]
        sizehint!(values, 2 * DOMAIN_DIM)
        for z in s
            push!(values, round(Int, real(z) * scale))
            push!(values, round(Int, imag(z) * scale))
        end
        push!(orbit_keys, ntuple(i -> values[i], 2 * DOMAIN_DIM))
    end
    minimum(orbit_keys)
end

function orbit_distance(a, b)
    denom = max(norm(a), norm(b), 1.0)
    minimum(norm(Vector{ComplexF64}(a) - s) / denom for s in root_group_orbit(b))
end

function same_root_orbit(a, b; tolerance::Float64 = 1e-7)
    orbit_distance(a, b) <= tolerance
end

function validate_checkpoint!(data::Dict{Symbol,Any})
    data[:version] == CHECKPOINT_VERSION ||
        error("Checkpoint version $(data[:version]) is not supported.")
    data[:problem] == :ternary_sextic_two_powers ||
        error("Checkpoint problem $(data[:problem]) is not ternary_sextic_two_powers.")
    for key in (:projection_matrix, :projection_offset, :start_parameter, :solutions)
        haskey(data, key) || error("Checkpoint is missing key $key.")
    end
    data
end

function checkpoint_compatible(a::Dict{Symbol,Any}, b::Dict{Symbol,Any};
    tolerance::Float64 = 1e-12,
)
    validate_checkpoint!(a)
    validate_checkpoint!(b)
    a[:seed] == b[:seed] || return false
    isapprox(a[:projection_matrix], b[:projection_matrix]; rtol = tolerance, atol = tolerance) || return false
    isapprox(a[:projection_offset], b[:projection_offset]; rtol = tolerance, atol = tolerance) || return false
    isapprox(a[:start_parameter], b[:start_parameter]; rtol = tolerance, atol = tolerance) || return false
    return true
end

function merged_solution_representatives(checkpoints;
    key_digits::Int = 8,
    tolerance::Float64 = 1e-7,
)
    representatives = Vector{Vector{ComplexF64}}()
    keyed = Dict{NTuple{2 * DOMAIN_DIM,Int},Int}()

    for data in checkpoints
        for solution in data[:solutions]
            candidate = Vector{ComplexF64}(solution)
            key = solution_key(candidate; digits = key_digits)
            haskey(keyed, key) && continue

            duplicate = false
            for existing in representatives
                if same_root_orbit(candidate, existing; tolerance)
                    duplicate = true
                    keyed[key] = 0
                    break
                end
            end
            duplicate && continue

            push!(representatives, candidate)
            keyed[key] = length(representatives)
        end
    end

    representatives
end

function merge_checkpoint_data(base::Dict{Symbol,Any}, others::Vector{Dict{Symbol,Any}};
    key_digits::Int = 8,
    tolerance::Float64 = 1e-7,
)
    validate_checkpoint!(base)
    for data in others
        checkpoint_compatible(base, data) ||
            error("Cannot merge checkpoints from different random projections.")
    end

    merged = copy(base)
    checkpoints = vcat([base], others)
    reps = merged_solution_representatives(checkpoints; key_digits, tolerance)
    merged[:solutions] = reps
    merged[:n_orbit_representatives] = length(reps)
    merged[:raw_solution_multiple] = ROOT_ACTION_ORDER
    merged[:raw_solutions_if_free_action] = length(reps) * ROOT_ACTION_ORDER
    merged[:return_code] = :merged
    merged[:merged_at] = string(now())
    merged[:merged_checkpoint_count] = length(checkpoints)
    merged
end

function merge_checkpoint_files(output::AbstractString, inputs::Vector{<:AbstractString};
    key_digits::Int = 8,
    tolerance::Float64 = 1e-7,
)
    isempty(inputs) && error("No input checkpoints to merge.")
    checkpoints = [validate_checkpoint!(deserialize(path)) for path in inputs]
    merged = merge_checkpoint_data(first(checkpoints), collect(checkpoints[2:end]); key_digits, tolerance)
    save_checkpoint(output, merged)
    merged
end

function update_checkpoint_from_result!(data::Dict{Symbol,Any}, result)
    data[:solutions] = solutions(results(result); only_nonsingular = false)
    data[:n_orbit_representatives] = nsolutions(result)
    data[:raw_solution_multiple] = ROOT_ACTION_ORDER
    data[:raw_solutions_if_free_action] = nsolutions(result) * ROOT_ACTION_ORDER
    data[:return_code] = result.returncode
    data[:tracked_loops] = length(result.loops)
    data[:monodromy_seed] = result.seed
    data[:trace] = result.trace
    data
end

function run(config::RunConfig)
    data = load_or_create_checkpoint(config)
    system = build_system(data[:projection_matrix], data[:projection_offset])
    start_parameter = data[:start_parameter]
    start_solutions = Vector{Vector{ComplexF64}}(data[:solutions])

    println("Ternary sextic two-powers monodromy")
    println("  locus: ternary sextics F = A^3 + B^2, deg(A)=2, deg(B)=3")
    println("  source affine dimension: ", DOMAIN_DIM)
    println("  target sextic coefficient dimension: ", N_SEXTIC)
    println("  random projection: S_6 -> A^", DOMAIN_DIM)
    println("  checkpoint: ", config.checkpoint)
    println("  starting representatives: ", length(start_solutions))
    println("  root-action quotient: ", config.use_group_action ? "on (6-fold raw symmetry)" : "off")
    println("  monodromy seed: ", isnothing(config.monodromy_seed) ? "default" : string(config.monodromy_seed))

    if config.rank_check
        summary = rank_summary(start_solutions[1])
        @printf("  coefficient-map Jacobian rank at start: %d/%d\n", summary.rank, DOMAIN_DIM)
        @printf("  Jacobian singular values: largest %.3e, smallest %.3e\n", summary.largest, summary.smallest)
    end

    start_residual = norm(evaluate(system, start_solutions[1], start_parameter), Inf)
    @printf("  start residual: %.3e\n", start_residual)

    last_saved_count = Ref(length(start_solutions))
    last_saved_time = Ref(time())
    callback = function (path_results)
        current_count = length(path_results)
        due = current_count != last_saved_count[] || (time() - last_saved_time[]) >= config.checkpoint_interval
        if due
            data[:solutions] = solutions(path_results; only_nonsingular = false)
            data[:n_orbit_representatives] = current_count
            data[:return_code] = :in_progress
            save_checkpoint(config.checkpoint, data)
            last_saved_count[] = current_count
            last_saved_time[] = time()
            println("\ncheckpoint saved with $(data[:n_orbit_representatives]) representatives")
        end
        return false
    end

    base_kwargs = (
        target_solutions_count = config.target_orbits,
        timeout = config.timeout,
        max_loops_no_progress = config.max_loops_no_progress,
        show_progress = config.show_progress,
        threading = config.threading,
        duplicate_check = config.duplicate_check,
        compile = config.compile,
        loop_finished_callback = callback,
    )
    kwargs = isnothing(config.monodromy_seed) ?
        base_kwargs :
        merge((seed = config.monodromy_seed,), base_kwargs)

    result =
        if config.use_group_action
            monodromy_solve(
                system,
                start_solutions,
                start_parameter;
                group_action = root_group_action(),
                kwargs...,
            )
        else
            monodromy_solve(system, start_solutions, start_parameter; kwargs...)
        end

    update_checkpoint_from_result!(data, result)
    save_checkpoint(config.checkpoint, data)

    println(result)
    println("orbit representatives / degree candidate: ", nsolutions(result))
    if config.use_group_action
        println("raw solutions if the root action is free: ", nsolutions(result) * ROOT_ACTION_ORDER)
    end
    println("checkpoint written: ", config.checkpoint)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        config = parse_cli(ARGS)
        run(config)
    catch err
        println(stderr, "ERROR: ", err)
        println(stderr)
        usage()
        exit(1)
    end
end
