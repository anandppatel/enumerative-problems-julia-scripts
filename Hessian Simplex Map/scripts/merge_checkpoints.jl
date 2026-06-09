#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "src", "HessianSimplex.jl"))
using .HessianSimplex

length(ARGS) >= 2 || error("usage: merge_checkpoints.jl OUTPUT INPUT1 [INPUT2 ...] [symmetry]")

use_symmetry = "symmetry" in ARGS || "quotient" in ARGS
paths = [arg for arg in ARGS if !(arg in ("symmetry", "quotient"))]
output = first(paths)
inputs = paths[2:end]

unique = merge_checkpoints(output, inputs; use_symmetry = use_symmetry ? true : nothing)

println("merged $(length(inputs)) checkpoint(s)")
println("unique solutions = $(length(unique))")
println("output = $output")
