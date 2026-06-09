#!/usr/bin/env julia

include(joinpath(@__DIR__, "ternary_sextic_two_powers_monodromy.jl"))

using Dates
using Printf
using Serialization

struct OrchestratorConfig
    master_checkpoint::String
    worker_dir::String
    workers::Int
    rounds::Int
    max_rounds_no_progress::Int
    worker_timeout::Union{Nothing,Float64}
    worker_max_loops_no_progress::Int
    checkpoint_interval::Float64
    target_orbits::Union{Nothing,Int}
    system_seed::UInt32
    first_worker_seed::UInt32
    worker_threading::Bool
    show_progress::Bool
    duplicate_check::Symbol
    compile
    rank_check::Bool
    key_digits::Int
    merge_tolerance::Float64
    julia_executable::String
    cleanup_workers::Bool
end

struct WorkerHandle
    id::Int
    checkpoint::String
    log::String
    seed::UInt32
    process::Base.Process
    io::IO
end

function orchestrator_usage()
    println("""
    Usage:
      julia ternary_sextic_monodromy_orchestrator.jl [options]

    Runs independent monodromy workers from a shared master checkpoint, then
    merges their worker-local checkpoints modulo the six root actions
    A -> zeta_3 A and B -> +/- B.

    Options:
      --master PATH                 master .jls checkpoint
      --worker-dir DIR              default: ternary_sextic_monodromy_workers
      --workers N                   default: 4
      --rounds N                    default: 1
      --max-rounds-no-progress N    default: rounds
      --worker-timeout SEC          default: none
      --worker-max-loops-no-progress N  default: 20
      --checkpoint-interval SEC     default: 60
      --target-orbits N             stop after merged checkpoint reaches N
      --seed N                      system/projection seed, default: 20260523
      --first-worker-seed N         monodromy seed for first worker, default: 900001
      --worker-threading true|false default: false
      --show-progress true|false    default: false
      --duplicate-check heuristic|certified  default: heuristic
      --compile true|false          default: true
      --rank-check true|false       default: false
      --merge-key-digits N          default: 8
      --merge-tolerance EPS         default: 1e-7
      --julia PATH                  default: current Julia executable
      --cleanup-workers true|false  default: false
    """)
end

function parse_uint32_option(value::AbstractString)
    UInt32(parse(UInt64, value) % (UInt64(typemax(UInt32)) + 1))
end

function parse_orchestrator_cli(argv)::OrchestratorConfig
    opts = Dict{String,String}()
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg in ("-h", "--help")
            orchestrator_usage()
            exit(0)
        elseif startswith(arg, "--")
            i == length(argv) && error("Missing value after $arg.")
            opts[arg] = argv[i + 1]
            i += 2
        else
            error("Unexpected argument: $arg")
        end
    end

    duplicate_check = Symbol(get(opts, "--duplicate-check", "heuristic"))
    duplicate_check in (:heuristic, :certified) ||
        error("--duplicate-check must be heuristic or certified.")

    rounds = parse(Int, get(opts, "--rounds", "1"))
    default_checkpoint = joinpath(@__DIR__, "ternary_sextic_two_powers_monodromy_checkpoint.jls")
    default_worker_dir = joinpath(@__DIR__, "ternary_sextic_monodromy_workers")

    OrchestratorConfig(
        get(opts, "--master", default_checkpoint),
        get(opts, "--worker-dir", default_worker_dir),
        parse(Int, get(opts, "--workers", "4")),
        rounds,
        parse(Int, get(opts, "--max-rounds-no-progress", string(rounds))),
        haskey(opts, "--worker-timeout") ? parse(Float64, opts["--worker-timeout"]) : nothing,
        parse(Int, get(opts, "--worker-max-loops-no-progress", "20")),
        parse(Float64, get(opts, "--checkpoint-interval", "60")),
        haskey(opts, "--target-orbits") ? parse(Int, opts["--target-orbits"]) : nothing,
        parse_uint32_option(get(opts, "--seed", "20260523")),
        parse_uint32_option(get(opts, "--first-worker-seed", "900001")),
        parse_bool(get(opts, "--worker-threading", "false")),
        parse_bool(get(opts, "--show-progress", "false")),
        duplicate_check,
        parse_bool(get(opts, "--compile", "true")),
        parse_bool(get(opts, "--rank-check", "false")),
        parse(Int, get(opts, "--merge-key-digits", "8")),
        parse(Float64, get(opts, "--merge-tolerance", "1e-7")),
        get(opts, "--julia", joinpath(Sys.BINDIR, Base.julia_exename())),
        parse_bool(get(opts, "--cleanup-workers", "false")),
    )
end

function worker_seed(config::OrchestratorConfig, round::Int, worker_id::Int)
    offset = UInt64((round - 1) * config.workers + (worker_id - 1))
    parse_uint32_option(string(UInt64(config.first_worker_seed) + offset))
end

function ensure_master_checkpoint(config::OrchestratorConfig)
    mkpath(dirname(config.master_checkpoint))
    master_config = RunConfig(
        config.system_seed,
        config.master_checkpoint,
        isfile(config.master_checkpoint),
        true,
        nothing,
        nothing,
        nothing,
        config.worker_max_loops_no_progress,
        config.checkpoint_interval,
        false,
        false,
        config.duplicate_check,
        config.compile,
        false,
    )
    data = load_or_create_checkpoint(master_config)
    validate_checkpoint!(data)
    data
end

function checkpoint_count(path::AbstractString)
    data = validate_checkpoint!(deserialize(path))
    length(data[:solutions])
end

function worker_paths(config::OrchestratorConfig, round::Int, worker_id::Int)
    stem = @sprintf("round-%04d-worker-%03d", round, worker_id)
    checkpoint = joinpath(config.worker_dir, stem * ".jls")
    log = joinpath(config.worker_dir, stem * ".log")
    (; checkpoint, log)
end

function worker_command(config::OrchestratorConfig, checkpoint::AbstractString, seed::UInt32)
    script = joinpath(@__DIR__, "ternary_sextic_two_powers_monodromy.jl")
    args = String[
        config.julia_executable,
        "--startup-file=no",
        script,
        "--checkpoint", checkpoint,
        "--resume",
        "--monodromy-seed", string(seed),
        "--max-loops-no-progress", string(config.worker_max_loops_no_progress),
        "--checkpoint-interval", string(config.checkpoint_interval),
        "--threading", string(config.worker_threading),
        "--show-progress", string(config.show_progress),
        "--duplicate-check", string(config.duplicate_check),
        "--compile", string(config.compile),
        "--rank-check", string(config.rank_check),
    ]
    if !isnothing(config.worker_timeout)
        append!(args, ["--timeout", string(config.worker_timeout)])
    end
    if !isnothing(config.target_orbits)
        append!(args, ["--target-orbits", string(config.target_orbits)])
    end
    Cmd(args)
end

function launch_worker(config::OrchestratorConfig, round::Int, worker_id::Int)
    paths = worker_paths(config, round, worker_id)
    cp(config.master_checkpoint, paths.checkpoint; force = true)
    seed = worker_seed(config, round, worker_id)
    cmd = worker_command(config, paths.checkpoint, seed)
    io = open(paths.log, "w")
    println(io, "started_at = ", now())
    println(io, "seed = ", seed)
    println(io, "command = ", join(cmd.exec, " "))
    flush(io)
    process = Base.run(pipeline(cmd, stdout = io, stderr = io); wait = false)
    WorkerHandle(worker_id, paths.checkpoint, paths.log, seed, process, io)
end

function wait_worker(handle::WorkerHandle)
    exitcode = -1
    try
        wait(handle.process)
        exitcode = handle.process.exitcode
    catch err
        println(handle.io, "wait_error = ", err)
    finally
        println(handle.io, "finished_at = ", now())
        println(handle.io, "exitcode = ", exitcode)
        close(handle.io)
    end
    exitcode == 0
end

function merge_workers!(config::OrchestratorConfig, worker_checkpoints::Vector{String})
    inputs = [config.master_checkpoint; filter(isfile, worker_checkpoints)]
    before = checkpoint_count(config.master_checkpoint)
    merged = merge_checkpoint_files(
        config.master_checkpoint,
        inputs;
        key_digits = config.key_digits,
        tolerance = config.merge_tolerance,
    )
    after = length(merged[:solutions])
    (; before, after, added = after - before)
end

function cleanup_worker_files(config::OrchestratorConfig, handles::Vector{WorkerHandle})
    config.cleanup_workers || return
    for handle in handles
        isfile(handle.checkpoint) && rm(handle.checkpoint; force = true)
    end
end

function orchestrate(config::OrchestratorConfig)
    mkpath(config.worker_dir)
    master = ensure_master_checkpoint(config)
    println("parallel ternary sextic monodromy orchestrator")
    println("  master checkpoint: ", config.master_checkpoint)
    println("  worker dir: ", config.worker_dir)
    println("  starting representatives: ", length(master[:solutions]))
    println("  workers per round: ", config.workers)
    println("  rounds: ", config.rounds)
    println("  worker timeout: ", isnothing(config.worker_timeout) ? "none" : string(config.worker_timeout))
    println("  target orbits: ", isnothing(config.target_orbits) ? "none" : string(config.target_orbits))

    rounds_without_progress = 0

    for round in 1:config.rounds
        start_count = checkpoint_count(config.master_checkpoint)
        println()
        println("round $round starting from $start_count representatives")

        handles = [launch_worker(config, round, id) for id in 1:config.workers]
        for handle in handles
            println("  launched worker $(handle.id), seed $(handle.seed), checkpoint $(handle.checkpoint)")
        end

        statuses = [wait_worker(handle) for handle in handles]
        for (handle, ok) in zip(handles, statuses)
            status = ok ? "ok" : "failed"
            count_text = isfile(handle.checkpoint) ? string(checkpoint_count(handle.checkpoint)) : "missing"
            println("  worker $(handle.id) $status, representatives: $count_text, log: $(handle.log)")
        end

        merge_summary = merge_workers!(config, [handle.checkpoint for handle in handles])
        println(
            "  merged master: ",
            merge_summary.before,
            " -> ",
            merge_summary.after,
            " (added ",
            merge_summary.added,
            ")",
        )

        cleanup_worker_files(config, handles)

        if merge_summary.added == 0
            rounds_without_progress += 1
        else
            rounds_without_progress = 0
        end

        if !isnothing(config.target_orbits) && merge_summary.after >= config.target_orbits
            println("target reached: $(merge_summary.after) >= $(config.target_orbits)")
            break
        end

        if rounds_without_progress >= config.max_rounds_no_progress
            println("stopping after $rounds_without_progress round(s) without merged progress")
            break
        end
    end

    println()
    println("final master representatives: ", checkpoint_count(config.master_checkpoint))
    println("master checkpoint written: ", config.master_checkpoint)
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        config = parse_orchestrator_cli(ARGS)
        orchestrate(config)
    catch err
        println(stderr, "ERROR: ", err)
        println(stderr)
        orchestrator_usage()
        exit(1)
    end
end
