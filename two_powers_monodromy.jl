#!/usr/bin/env julia

using Dates
using HomotopyContinuation
using LinearAlgebra
using Random
using Serialization

const CHECKPOINT_VERSION = 1

struct RunConfig
    m::Int
    n::Int
    a::Int
    b::Int
    seed::UInt32
    checkpoint::String
    resume::Bool
    use_group_action::Bool
    target_orbits::Union{Nothing,Int}
    timeout::Union{Nothing,Float64}
    max_loops_no_progress::Int
    checkpoint_interval::Float64
    show_progress::Bool
    threading::Bool
    duplicate_check::Symbol
    compile
end

function usage()
    println("""
    Usage:
      julia two_powers_monodromy.jl --m 6 --n 9 --a 3 --b 2 [options]

    Verifies counts for the map S_m x S_n -> S_d, (f,g) -> f^a + g^b,
    after composing with a random affine-linear map S_d -> A^(m+n+2).

    Options:
      --checkpoint PATH          .jls checkpoint path
      --resume                   resume from checkpoint if it exists
      --no-group-action          do not quotient by root-of-unity actions
      --target-orbits N          stop once N orbit representatives are found
      --timeout SECONDS          stop after this many seconds
      --max-loops-no-progress N  default: 30
      --checkpoint-interval SEC  default: 30
      --seed N                   default: 20260519
      --threading true|false     default: false
      --show-progress true|false default: true
      --duplicate-check heuristic|certified  default: heuristic
      --compile true|false       default: true
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

    for required in ("--m", "--n", "--a", "--b")
        haskey(opts, required) || error("Missing required option $required.")
    end

    m = parse(Int, opts["--m"])
    n = parse(Int, opts["--n"])
    a = parse(Int, opts["--a"])
    b = parse(Int, opts["--b"])
    all(>(0), (m, n, a, b)) || error("m, n, a, b must be positive.")
    a * m == b * n || error("Expected a*m = b*n, got $(a*m) and $(b*n).")

    default_checkpoint = joinpath(
        @__DIR__,
        "two_powers_m$(m)_n$(n)_a$(a)_b$(b)_monodromy_checkpoint.jls",
    )

    duplicate_check = Symbol(get(opts, "--duplicate-check", "heuristic"))
    duplicate_check in (:heuristic, :certified) || error("--duplicate-check must be heuristic or certified.")

    compile_value = parse_bool(get(opts, "--compile", "true"))

    RunConfig(
        m,
        n,
        a,
        b,
        UInt32(parse(UInt64, get(opts, "--seed", "20260519")) % (UInt64(typemax(UInt32)) + 1)),
        get(opts, "--checkpoint", default_checkpoint),
        "--resume" in flags,
        !("--no-group-action" in flags),
        haskey(opts, "--target-orbits") ? parse(Int, opts["--target-orbits"]) : nothing,
        haskey(opts, "--timeout") ? parse(Float64, opts["--timeout"]) : nothing,
        parse(Int, get(opts, "--max-loops-no-progress", "30")),
        parse(Float64, get(opts, "--checkpoint-interval", "30")),
        parse_bool(get(opts, "--show-progress", "true")),
        parse_bool(get(opts, "--threading", "false")),
        duplicate_check,
        compile_value,
    )
end

function power_coefficients(coeffs, exponent::Int)
    out = Any[one(coeffs[1])]
    for _ in 1:exponent
        next = [zero(coeffs[1]) for _ in 1:(length(out) + length(coeffs) - 1)]
        for i in eachindex(out), j in eachindex(coeffs)
            next[i + j - 1] += out[i] * coeffs[j]
        end
        out = next
    end
    out
end

function build_system(config::RunConfig, projection_matrix, projection_offset)
    m, n, a, b = config.m, config.n, config.a, config.b
    domain_dim = m + n + 2
    d = a * m

    @var x[1:domain_dim]
    @var y[1:domain_dim]

    f = x[1:(m + 1)]
    g = x[(m + 2):domain_dim]
    image_coeffs = power_coefficients(f, a) .+ power_coefficients(g, b)

    equations = [
        sum(projection_matrix[i, j] * image_coeffs[j] for j in 1:(d + 1)) +
        projection_offset[i] - y[i]
        for i in 1:domain_dim
    ]

    System(equations; variables = x, parameters = y)
end

function new_instance(config::RunConfig)
    Random.seed!(config.seed)
    domain_dim = config.m + config.n + 2
    form_dim = config.a * config.m + 1
    projection_matrix = randn(ComplexF64, domain_dim, form_dim)
    projection_offset = randn(ComplexF64, domain_dim)
    system = build_system(config, projection_matrix, projection_offset)
    start_solution, start_parameter = find_start_pair(system)

    Dict{Symbol,Any}(
        :version => CHECKPOINT_VERSION,
        :created_at => string(now()),
        :updated_at => string(now()),
        :m => config.m,
        :n => config.n,
        :a => config.a,
        :b => config.b,
        :seed => config.seed,
        :projection_matrix => projection_matrix,
        :projection_offset => projection_offset,
        :start_parameter => start_parameter,
        :solutions => [start_solution],
        :n_orbit_representatives => 1,
        :raw_solution_multiple => config.a * config.b,
        :return_code => :start_pair,
        :tracked_loops => 0,
    )
end

function load_or_create_checkpoint(config::RunConfig)
    if config.resume && isfile(config.checkpoint)
        data = deserialize(config.checkpoint)
        for key in (:m, :n, :a, :b)
            data[key] == getfield(config, key) ||
                error("Checkpoint has $key=$(data[key]), but command line has $(getfield(config, key)).")
        end
        return data
    end

    data = new_instance(config)
    save_checkpoint(config.checkpoint, data)
    return data
end

function save_checkpoint(path::AbstractString, data::Dict{Symbol,Any})
    data[:updated_at] = string(now())
    tmp = path * ".tmp"
    serialize(tmp, data)
    mv(tmp, path; force = true)
    nothing
end

function root_group_action(config::RunConfig)
    m, n, a, b = config.m, config.n, config.a, config.b
    function action(solution)
        transformed = Vector{Vector{ComplexF64}}()
        for ia in 0:(a - 1), ib in 0:(b - 1)
            ia == 0 && ib == 0 && continue
            sf = cis(2π * ia / a)
            sg = cis(2π * ib / b)
            s = Vector{ComplexF64}(solution)
            s[1:(m + 1)] .*= sf
            s[(m + 2):(m + n + 2)] .*= sg
            push!(transformed, s)
        end
        Tuple(transformed)
    end
    action
end

function update_checkpoint_from_result!(data::Dict{Symbol,Any}, result)
    data[:solutions] = solutions(results(result); only_nonsingular = false)
    data[:n_orbit_representatives] = nsolutions(result)
    data[:raw_solution_multiple] = data[:a] * data[:b]
    data[:raw_solutions_if_free_action] = nsolutions(result) * data[:raw_solution_multiple]
    data[:return_code] = result.returncode
    data[:tracked_loops] = length(result.loops)
    data[:monodromy_seed] = result.seed
    data[:trace] = result.trace
    data
end

function run(config::RunConfig)
    data = load_or_create_checkpoint(config)
    system = build_system(config, data[:projection_matrix], data[:projection_offset])
    start_parameter = data[:start_parameter]
    start_solutions = Vector{Vector{ComplexF64}}(data[:solutions])

    println("Two-powers monodromy verification")
    println("  map: S_$(config.m) x S_$(config.n) -> S_$(config.a * config.m), (f,g) -> f^$(config.a) + g^$(config.b)")
    println("  domain dimension: $(config.m + config.n + 2)")
    println("  checkpoint: $(config.checkpoint)")
    println("  starting representatives: $(length(start_solutions))")
    println("  root-action quotient: $(config.use_group_action ? "on ($(config.a * config.b)-fold raw symmetry)" : "off")")

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

    kwargs = (
        target_solutions_count = config.target_orbits,
        timeout = config.timeout,
        max_loops_no_progress = config.max_loops_no_progress,
        show_progress = config.show_progress,
        threading = config.threading,
        duplicate_check = config.duplicate_check,
        compile = config.compile,
        loop_finished_callback = callback,
    )

    result =
        if config.use_group_action
            monodromy_solve(
                system,
                start_solutions,
                start_parameter;
                group_action = root_group_action(config),
                kwargs...,
            )
        else
            monodromy_solve(system, start_solutions, start_parameter; kwargs...)
        end

    update_checkpoint_from_result!(data, result)
    save_checkpoint(config.checkpoint, data)

    println(result)
    println("orbit representatives: ", nsolutions(result))
    if config.use_group_action
        println("raw solutions if the root action is free: ", nsolutions(result) * config.a * config.b)
    end
    println("checkpoint written: ", config.checkpoint)
    return result
end

try
    config = parse_cli(ARGS)
    run(config)
catch err
    println(stderr, "ERROR: ", err)
    println(stderr)
    usage()
    exit(1)
end
