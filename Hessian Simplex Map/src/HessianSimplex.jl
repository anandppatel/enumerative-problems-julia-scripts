module HessianSimplex

using HomotopyContinuation
using LinearAlgebra
using Random
using Dates
using Printf
using TOML

export build_model,
       expected_degree,
       monodromy_degree,
       chart_residuals,
       omitted_product_residuals,
       symmetric_group_actions,
       random_geometric_start_pair,
       write_checkpoint,
       read_checkpoint,
       merge_checkpoints,
       checkpoint_callback

struct HessianSimplexModel
    d::Int
    system::System
    variables::Vector
    parameters::Vector
    input_coefficients::Matrix
    target_chart::Matrix
    gauge_coefficients::Vector{ComplexF64}
    products::Vector
    product_coefficients::Vector{Vector}
    chart_rows::Vector{Int}
    nonchart_rows::Vector{Int}
end

function expected_degree(d::Integer)
    d >= 1 || throw(ArgumentError("d must be positive"))
    N = d * (d + 1)
    numerator = big(d)^N * prod(factorial(big(i)) for i in (d + 1):(2d))
    denominator = factorial(big(d))^(d + 1) * prod(factorial(big(i)) for i in 1:(d - 1); init = big(1))
    return numerator ÷ denominator
end

function diffpow(f, z, n::Integer)
    n == 0 && return f
    g = f
    for _ in 1:n
        g = differentiate(g, z)
    end
    return g
end

function coeff_vector(f, degree::Integer, x, y)
    [
        subs(diffpow(diffpow(f, x, degree - k), y, k), x => 0, y => 0) /
        (factorial(degree - k) * factorial(k))
        for k in 0:degree
    ]
end

function binary_form(coeffs, d::Integer, x, y)
    sum(coeffs[k + 1] * x^(d - k) * y^k for k in 0:d)
end

function hessian_minor(forms, omitted::Integer, d::Integer, x, y)
    cols = [i for i in eachindex(forms) if i != omitted]
    M = [
        diffpow(diffpow(forms[col], x, d - 1 - r), y, r)
        for r in 0:(d - 1), col in cols
    ]
    return det(M)
end

function random_gauge(d::Integer; seed = 101)
    rng = MersenneTwister(seed)
    return randn(rng, ComplexF64, d + 1)
end

"""
    build_model(d; gauge_coefficients = random_gauge(d), chart_rows = 1:d)

Build a square affine chart system for the fiber of the Hessian simplex map.

The target `d`-plane in `S_{2d}` is represented in the Grassmann chart
`[I_d; B]`, where the entries of `B` are the system parameters.  The equations
say that the first `d` products `f_j H_j` lie in this target plane, plus one
generic affine gauge equation for each factor of `(P^d)^(d+1)`.
"""
function build_model(d::Integer; gauge_coefficients = random_gauge(d), chart_rows = collect(1:d))
    d >= 1 || throw(ArgumentError("d must be positive"))
    n = d + 1
    degree = 2d

    length(gauge_coefficients) == n ||
        throw(ArgumentError("expected $(n) gauge coefficients"))
    length(chart_rows) == d ||
        throw(ArgumentError("expected $(d) chart rows"))

    @var x y
    @var a[1:n, 1:n]
    @var b[1:n, 1:d]

    variables = [a[j, k] for j in 1:n for k in 1:n]
    parameters = [b[r, c] for r in 1:n for c in 1:d]

    forms = [binary_form([a[j, k] for k in 1:n], d, x, y) for j in 1:n]
    hessians = [hessian_minor(forms, j, d, x, y) for j in 1:n]
    products = [forms[j] * hessians[j] for j in 1:n]
    product_coefficients = [coeff_vector(products[j], degree, x, y) for j in 1:n]

    nonchart_rows = [i for i in 1:(degree + 1) if !(i in chart_rows)]
    chart_equations = [
        product_coefficients[col][row] -
        sum(b[r, q] * product_coefficients[col][chart_rows[q]] for q in 1:d)
        for col in 1:d
        for (r, row) in enumerate(nonchart_rows)
    ]

    gauge_equations = [
        sum(gauge_coefficients[k] * a[j, k] for k in 1:n) - 1
        for j in 1:n
    ]

    equations = [chart_equations; gauge_equations]
    system = System(equations; variables, parameters)

    return HessianSimplexModel(
        Int(d),
        system,
        variables,
        parameters,
        a,
        b,
        ComplexF64.(gauge_coefficients),
        products,
        product_coefficients,
        collect(chart_rows),
        nonchart_rows,
    )
end

function permute_solution_blocks(d::Integer, solution, perm)
    n = d + 1
    moved = similar(solution)
    for new_block in 1:n
        old_block = perm[new_block]
        moved[((new_block - 1) * n + 1):(new_block * n)] =
            solution[((old_block - 1) * n + 1):(old_block * n)]
    end
    return moved
end

function symmetric_group_actions(d::Integer)
    n = d + 1
    perms = collect(SymmetricGroup(n))
    action = function (solution)
        return tuple((permute_solution_blocks(d, solution, perm) for perm in perms)...)
    end
    return GroupActions(action)
end

function product_coefficient_matrix(model::HessianSimplexModel, solution)
    coeffs = reduce(vcat, model.product_coefficients)
    values = evaluate(
        System(coeffs; variables = model.variables, parameters = model.parameters),
        solution,
        zeros(ComplexF64, length(model.parameters)),
    )
    rows = 2model.d + 1
    cols = model.d + 1
    return reshape(values, rows, cols)
end

function random_gauged_coefficients(model::HessianSimplexModel, rng)
    n = model.d + 1
    A = randn(rng, ComplexF64, n, n)
    for j in 1:n
        scale = sum(model.gauge_coefficients[k] * A[j, k] for k in 1:n)
        abs(scale) > 1e-10 || return nothing
        A[j, :] ./= scale
    end
    return A
end

"""
    random_geometric_start_pair(model; seed = 1001, max_tries = 1000)

Construct a monodromy start pair geometrically.  Pick a random ordered tuple of
binary forms, rescale each form to satisfy the affine gauge, compute the
products `f_j H_j`, and use their span to read off the target Grassmann chart
parameter.
"""
function random_geometric_start_pair(model::HessianSimplexModel; seed = 1001, max_tries = 1000)
    rng = MersenneTwister(seed)
    d = model.d
    n = d + 1
    for _ in 1:max_tries
        A = random_gauged_coefficients(model, rng)
        isnothing(A) && continue
        x = ComplexF64[A[j, k] for j in 1:n for k in 1:n]
        C = product_coefficient_matrix(model, x)
        pivot = C[model.chart_rows, 1:d]
        abs(det(pivot)) > 1e-10 || continue
        B = C[model.nonchart_rows, 1:d] / pivot
        p = ComplexF64[B[r, c] for r in 1:n for c in 1:d]
        residual = evaluate(model.system, x, p)
        maximum(abs.(residual)) < 1e-8 || continue
        return x, p
    end
    return nothing
end

random_geometric_start_pair(d::Integer; kwargs...) =
    random_geometric_start_pair(build_model(d); kwargs...)

function complex_to_pairs(v)
    [[real(z), imag(z)] for z in v]
end

function pairs_to_complex(v)
    ComplexF64[item[1] + im * item[2] for item in v]
end

function write_complex_matrix(path::AbstractString, rows)
    open(path, "w") do io
        for row in rows
            first_entry = true
            for z in row
                if !first_entry
                    print(io, '\t')
                end
                @printf(io, "%.17e\t%.17e", real(z), imag(z))
                first_entry = false
            end
            println(io)
        end
    end
end

function read_complex_matrix(path::AbstractString)
    rows = Vector{Vector{ComplexF64}}()
    isfile(path) || return rows
    for line in eachline(path)
        stripped = strip(line)
        isempty(stripped) && continue
        vals = parse.(Float64, split(stripped))
        iseven(length(vals)) || error("expected real/imag pairs in $path")
        push!(rows, [vals[i] + im * vals[i + 1] for i in 1:2:length(vals)])
    end
    return rows
end

function atomic_write_complex_matrix(path::AbstractString, rows)
    mkpath(dirname(path))
    tmp = path * ".tmp"
    write_complex_matrix(tmp, rows)
    mv(tmp, path; force = true)
end

function solution_vectors(results)
    [Vector{ComplexF64}(solution(r)) for r in results if is_success(r)]
end

function write_checkpoint(
    path::AbstractString,
    model::HessianSimplexModel,
    solutions,
    parameter_values;
    use_symmetry::Bool = false,
    note::AbstractString = "",
)
    mkpath(path)
    metadata = Dict{String, Any}(
        "format" => "hessian-simplex-checkpoint-v1",
        "d" => model.d,
        "nvariables" => length(model.variables),
        "nparameters" => length(model.parameters),
        "chart_rows" => model.chart_rows,
        "nonchart_rows" => model.nonchart_rows,
        "gauge_coefficients" => complex_to_pairs(model.gauge_coefficients),
        "use_symmetry" => use_symmetry,
        "orbit_factor" => factorial(model.d + 1),
        "nsolutions" => length(solutions),
        "updated_at" => string(now()),
        "note" => note,
    )
    metadata_path = joinpath(path, "metadata.toml")
    tmp_metadata_path = metadata_path * ".tmp"
    open(tmp_metadata_path, "w") do io
        TOML.print(io, metadata)
    end
    mv(tmp_metadata_path, metadata_path; force = true)
    atomic_write_complex_matrix(joinpath(path, "parameters.tsv"), [parameter_values])
    atomic_write_complex_matrix(joinpath(path, "solutions.tsv"), solutions)
    return path
end

function read_checkpoint(path::AbstractString)
    metadata = TOML.parsefile(joinpath(path, "metadata.toml"))
    parameters = only(read_complex_matrix(joinpath(path, "parameters.tsv")))
    solutions = read_complex_matrix(joinpath(path, "solutions.tsv"))
    return metadata, solutions, parameters
end

function checkpoint_callback(
    path::AbstractString,
    model::HessianSimplexModel,
    parameter_values;
    use_symmetry::Bool = false,
    every_seconds::Real = 30,
)
    last_write = Ref(0.0)
    last_count = Ref(-1)
    return function (results)
        sols = solution_vectors(results)
        t = time()
        if length(sols) != last_count[] && t - last_write[] >= every_seconds
            write_checkpoint(
                path,
                model,
                sols,
                parameter_values;
                use_symmetry,
                note = "periodic monodromy checkpoint",
            )
            last_write[] = t
            last_count[] = length(sols)
        end
        return false
    end
end

function same_vector(a, b; atol = 1e-10, rtol = 1e-8)
    length(a) == length(b) || return false
    return norm(a - b) <= max(atol, rtol * max(norm(a), norm(b)))
end

function same_orbit(a, b, actions; atol = 1e-10, rtol = 1e-8)
    same_vector(a, b; atol, rtol) && return true
    isnothing(actions) && return false
    for moved in actions(b)
        same_vector(a, moved; atol, rtol) && return true
    end
    return false
end

function deduplicate_solutions(solutions, d::Integer; use_symmetry = false, atol = 1e-10, rtol = 1e-8)
    actions = use_symmetry ? symmetric_group_actions(d) : nothing
    unique = Vector{Vector{ComplexF64}}()
    for sol in solutions
        if !any(existing -> same_orbit(existing, sol, actions; atol, rtol), unique)
            push!(unique, sol)
        end
    end
    return unique
end

function merge_checkpoints(
    output_path::AbstractString,
    input_paths::AbstractVector{<:AbstractString};
    use_symmetry = nothing,
    atol = 1e-10,
    rtol = 1e-8,
)
    isempty(input_paths) && throw(ArgumentError("at least one input checkpoint is required"))
    base_metadata, base_solutions, base_parameters = read_checkpoint(first(input_paths))
    d = base_metadata["d"]
    symmetry = isnothing(use_symmetry) ? get(base_metadata, "use_symmetry", false) : use_symmetry
    all_solutions = copy(base_solutions)

    for path in input_paths[2:end]
        metadata, solutions, parameters = read_checkpoint(path)
        metadata["d"] == d || error("cannot merge checkpoints with different d")
        same_vector(
            pairs_to_complex(base_metadata["gauge_coefficients"]),
            pairs_to_complex(metadata["gauge_coefficients"]);
            atol,
            rtol,
        ) || error("cannot merge checkpoints with different gauge coefficients")
        same_vector(base_parameters, parameters; atol, rtol) ||
            error("cannot merge checkpoints with different base parameters")
        append!(all_solutions, solutions)
    end

    unique = deduplicate_solutions(all_solutions, d; use_symmetry = symmetry, atol, rtol)
    model = build_model(d; gauge_coefficients = pairs_to_complex(base_metadata["gauge_coefficients"]))
    write_checkpoint(
        output_path,
        model,
        unique,
        base_parameters;
        use_symmetry = symmetry,
        note = "merged from $(length(input_paths)) checkpoints",
    )
    return unique
end

function monodromy_degree(
    d::Integer;
    target_solutions_count = nothing,
    timeout = 300,
    seed = 101,
    use_symmetry = false,
    checkpoint_path = nothing,
    checkpoint_interval = 30,
    start_solutions = nothing,
    start_parameters = nothing,
    gauge_coefficients = random_gauge(d; seed),
    start_pair_max_tries = 500,
    kwargs...,
)
    kwargs_dict = Dict(kwargs)
    model = build_model(d; gauge_coefficients)
    if checkpoint_path !== nothing && start_solutions === nothing
        start_pair = random_geometric_start_pair(
            model;
            seed,
            max_tries = start_pair_max_tries,
        )
        isnothing(start_pair) && error(
            "Cannot compute a start pair for checkpointed monodromy. " *
            "Pass resume=<checkpoint> or provide explicit start data.",
        )
        x, p = start_pair
        start_solutions = [x]
        start_parameters = p
    end
    options = Dict{Symbol, Any}(
        :timeout => Float64(timeout),
        :permutations => true,
    )
    if target_solutions_count !== nothing
        target = target_solutions_count isa Integer ? Int(target_solutions_count) : target_solutions_count
        options[:target_solutions_count] = target
    end
    use_symmetry && (options[:group_actions] = symmetric_group_actions(d))
    if checkpoint_path !== nothing && start_parameters !== nothing
        options[:loop_finished_callback] = checkpoint_callback(
            checkpoint_path,
            model,
            start_parameters;
            use_symmetry,
            every_seconds = checkpoint_interval,
        )
    end
    merge!(options, kwargs_dict)
    if start_solutions === nothing
        result = monodromy_solve(model.system; options...)
    else
        result = monodromy_solve(model.system, start_solutions, start_parameters; options...)
    end
    if checkpoint_path !== nothing
        write_checkpoint(
            checkpoint_path,
            model,
            solutions(result),
            parameters(result);
            use_symmetry,
            note = "final monodromy checkpoint",
        )
    end
    return model, result
end

function chart_residuals(model::HessianSimplexModel, solution, parameter_values)
    evaluate(model.system, solution, parameter_values)
end

function omitted_product_residuals(model::HessianSimplexModel, solution, parameter_values)
    d = model.d
    n = d + 1
    residuals = [
        model.product_coefficients[n][row] -
        sum(model.target_chart[r, q] * model.product_coefficients[n][model.chart_rows[q]] for q in 1:d)
        for (r, row) in enumerate(model.nonchart_rows)
    ]
    return evaluate(System(residuals; variables = model.variables, parameters = model.parameters), solution, parameter_values)
end

end
