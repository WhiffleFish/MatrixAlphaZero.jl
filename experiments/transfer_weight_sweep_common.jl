# Shared driver for the learned-prior transfer-weight sweeps.
#
# A sweep cell is one ordered matchup between SM-AZ with regret transfer at
# prior_scale = wT1 (`full_solver`) and SM-AZ(-RT) (`value_oracle`) at a given
# search budget. Seat 1 puts the transfer solver in the player-one seat. Each
# cell is written to its own CSV under `cells/`, so independent processes can
# split the grid with `--shard i/n`, and a rerun skips finished cells.

using DelimitedFiles
using LinearAlgebra
using Printf
using Statistics

const SWEEP_SOLVERS = ("full_solver", "value_oracle")

parse_float_list(spec) = parse.(Float64, split(spec, ','))
parse_int_list(spec) = parse.(Int, split(spec, ','))

function parse_shard(spec)
    parts = parse.(Int, split(spec, '/'))
    length(parts) == 2 || error("--shard must have the form i/n")
    index, count = parts
    1 <= index <= count || error("--shard must satisfy 1 <= i <= n")
    return index, count
end

# The most expensive cells come first so round-robin sharding balances load.
function sweep_cells(queries, scales)
    cells = [(; queries=q, scale=s, seat) for q in queries for s in scales for seat in 1:2]
    return sort!(cells; by=c -> -c.queries)
end

shard_cells(cells, index, count) = cells[index:count:end]

scale_label(scale) = replace(@sprintf("%g", scale), '.' => 'p')
cell_file(dir, cell) = joinpath(
    dir, "cells", "q$(cell.queries)_scale$(scale_label(cell.scale))_seat$(cell.seat).csv",
)

seat_solvers(seat) = seat == 1 ? SWEEP_SOLVERS : reverse(SWEEP_SOLVERS)

# Every scale shares the initial states and rollout seed of its (budget, seat),
# so differences across w are paired.
cell_seed(seed, cell) = seed + 1000 * cell.seat + cell.queries

csv_field(value::AbstractString) = replace(value, ',' => ';')
csv_field(value) = value

function write_cell(path, row::NamedTuple)
    mkpath(dirname(path))
    header = reshape(collect(String.(keys(row))), 1, :)
    data = reshape(Any[csv_field(v) for v in values(row)], 1, :)
    tmp = path * ".tmp"
    writedlm(tmp, [header; data], ',')
    mv(tmp, path; force=true)
    return path
end

function run_sweep_cells(evaluate, cells, output_dir; tag, source_queries, sweep_seed, extras...)
    pending = filter(cell -> !isfile(cell_file(output_dir, cell)), cells)
    println("[$(tag)] cells=$(length(cells)) done=$(length(cells) - length(pending)) pending=$(length(pending))")
    flush(stdout)
    start = time()
    for (k, cell) in enumerate(pending)
        row_solver, column_solver = seat_solvers(cell.seat)
        row = evaluate(row_solver, column_solver, cell, cell_seed(sweep_seed, cell))
        row = merge(row, (;
            sweep_scale=cell.scale,
            transfer_weight=cell.scale / source_queries,
            source_queries,
            transfer_seat=cell.seat,
            cell_seed=cell_seed(sweep_seed, cell),
            sweep_seed,
        ), (; extras...))
        write_cell(cell_file(output_dir, cell), row)
        per_cell = (time() - start) / k
        @printf(
            "[%s] cell %d/%d q=%d scale=%g seat=%d p1_utility=%.4f elapsed=%.0fs eta=%.1f min\n",
            tag, k, length(pending), cell.queries, cell.scale, cell.seat, row.p1_utility,
            row.elapsed_seconds, per_cell * (length(pending) - k) / 60,
        )
        flush(stdout)
    end
    return output_dir
end

function read_cells(output_dir)
    dir = joinpath(output_dir, "cells")
    isdir(dir) || return NamedTuple[]
    rows = NamedTuple[]
    for file in sort!(filter(f -> endswith(f, ".csv"), readdir(dir)))
        data, header = readdlm(joinpath(dir, file), ','; header=true)
        field(name) = data[1, findfirst(==(name), vec(header))]
        push!(rows, (;
            queries=Int(field("queries")),
            scale=Float64(field("sweep_scale")),
            weight=Float64(field("transfer_weight")),
            seat=Int(field("transfer_seat")),
            runs=Int(field("runs")),
            p1_utility=Float64(field("p1_utility")),
            p1_stderr=Float64(field("p1_stderr")),
            p2_utility=Float64(field("p2_utility")),
            p2_stderr=Float64(field("p2_stderr")),
            elapsed_seconds=Float64(field("elapsed_seconds")),
        ))
    end
    return rows
end

# Seat-balanced utility of the transfer solver against SM-AZ(-RT), as in the
# paper's round robins: (u1(full, value) - u1(value, full)) / 2. Per-seat
# columns give the transfer solver's own utility, and the deltas subtract the
# scale-0 cell of the same seat (the same search as SM-AZ(-RT)).
function summarize_sweep_dir(output_dir)
    rows = read_cells(output_dir)
    isempty(rows) && return nothing
    lookup = Dict((r.queries, r.scale, r.seat) => r for r in rows)
    summary = NamedTuple[]
    for (q, s) in sort!(unique((r.queries, r.scale) for r in rows))
        haskey(lookup, (q, s, 1)) && haskey(lookup, (q, s, 2)) || continue
        a, b = lookup[(q, s, 1)], lookup[(q, s, 2)]
        balanced = (a.p1_utility - b.p1_utility) / 2
        balanced_stderr = hypot(a.p1_stderr, b.p1_stderr) / 2
        seat1, seat1_stderr = a.p1_utility, a.p1_stderr
        seat2, seat2_stderr = b.p2_utility, b.p2_stderr
        ref1 = get(lookup, (q, 0.0, 1), nothing)
        ref2 = get(lookup, (q, 0.0, 2), nothing)
        push!(summary, (;
            queries=q,
            prior_scale=s,
            transfer_weight=a.weight,
            runs_per_seat=a.runs,
            seat_balanced_utility=balanced,
            seat_balanced_stderr=balanced_stderr,
            ci95_lower=balanced - 1.96 * balanced_stderr,
            ci95_upper=balanced + 1.96 * balanced_stderr,
            transfer_as_p1_utility=seat1,
            transfer_as_p1_stderr=seat1_stderr,
            transfer_as_p2_utility=seat2,
            transfer_as_p2_stderr=seat2_stderr,
            p1_delta_vs_scale0=isnothing(ref1) ? NaN : seat1 - ref1.p1_utility,
            p2_delta_vs_scale0=isnothing(ref2) ? NaN : seat2 - ref2.p2_utility,
            elapsed_seconds=a.elapsed_seconds + b.elapsed_seconds,
        ))
    end
    isempty(summary) && return nothing
    path = joinpath(output_dir, "summary.csv")
    header = reshape(collect(String.(keys(first(summary)))), 1, :)
    data = reduce(vcat, [reshape(collect(Any, values(r)), 1, :) for r in summary])
    # Shards finish independently, so each writes through its own temp file.
    tmp = "$(path).$(getpid()).tmp"
    writedlm(tmp, [header; data], ',')
    mv(tmp, path; force=true)
    return path
end
