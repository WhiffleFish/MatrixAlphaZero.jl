using DelimitedFiles
using Printf

const SOLVER_ORDER = ("zero_oracle", "value_oracle", "full_solver")
const BENCHMARK_ORDER = ("uniform_random", "heuristic", "ppo_br")
const SOLVER_LABELS = Dict(
    "zero_oracle" => "Zero oracle",
    "value_oracle" => "Value oracle",
    "full_solver" => "Full solver",
)
const BENCHMARK_LABELS = Dict(
    "uniform_random" => "Uniform random",
    "heuristic" => "Heuristic",
    "ppo_br" => "PPO BR",
)

function usage()
    println("""
    Usage:
      julia --project=experiments experiments/solver_benchmark_to_latex.jl DATA_DIR [options]

    Options:
      --output PATH    Output path (default: DATA_DIR/benchmark_table.tex)
      --problem NAME   Problem name used in the caption (inferred from DATA_DIR)
      --caption TEXT   Override the complete table caption
      --label LABEL    LaTeX label (default: tab:<problem>-solver-benchmarks)
      --digits N       Digits after the decimal point (default: 3)
      --help           Show this message
    """)
end

function parse_args(args)
    (isempty(args) || "--help" in args) && (usage(); exit(isempty(args) ? 1 : 0))
    startswith(first(args), "--") && error("The first argument must be DATA_DIR")
    data_dir = abspath(first(args))
    options = Dict(
        "output" => joinpath(data_dir, "benchmark_table.tex"),
        "problem" => "",
        "caption" => "",
        "label" => "",
        "digits" => "3",
    )
    i = 2
    while i <= length(args)
        key = args[i]
        startswith(key, "--") || error("Expected --key, got $(key)")
        option = key[3:end]
        haskey(options, option) || error("Unknown option $(key)")
        i += 1
        i <= length(args) || error("Missing value for $(key)")
        options[option] = args[i]
        i += 1
    end

    isdir(data_dir) || error("Data directory does not exist: $(data_dir)")
    digits = parse(Int, options["digits"])
    digits >= 0 || error("--digits must be nonnegative")
    problem = isempty(options["problem"]) ? infer_problem(data_dir) : options["problem"]
    slug = lowercase(replace(problem, r"[^A-Za-z0-9]+" => "-"))
    slug = strip(slug, '-')
    caption = isempty(options["caption"]) ?
        "Solver benchmark utilities for $(problem). Entries report mean utility " *
        raw"$\pm$ standard error for the tree-search player." :
        options["caption"]
    label = isempty(options["label"]) ? "tab:$(slug)-solver-benchmarks" : options["label"]
    return (; data_dir, output=abspath(options["output"]), problem, caption, label, digits)
end

function infer_problem(data_dir)
    path = lowercase(data_dir)
    occursin("dubin", path) && return "Dubin"
    occursin("sda", path) && return "SDA"
    parent = basename(dirname(data_dir))
    return isempty(parent) ? "solver benchmark" : parent
end

function read_benchmark_csv(path)
    isfile(path) || error("Missing benchmark file: $(path)")
    data, header = readdlm(path, ',', Any, '\n'; header=true)
    columns = String.(vec(header))
    expected = ["solver"; collect(BENCHMARK_ORDER)]
    columns == expected || error(
        "Unexpected columns in $(path): $(join(columns, ", ")); expected $(join(expected, ", "))",
    )
    size(data, 1) == length(SOLVER_ORDER) || error(
        "Expected $(length(SOLVER_ORDER)) solver rows in $(path), got $(size(data, 1))",
    )
    values = Dict{String,Vector{Float64}}()
    for row in axes(data, 1)
        solver = String(data[row, 1])
        solver in SOLVER_ORDER || error("Unknown solver $(solver) in $(path)")
        haskey(values, solver) && error("Duplicate solver $(solver) in $(path)")
        values[solver] = Float64.(data[row, 2:end])
        all(isfinite, values[solver]) || error("Non-finite value for $(solver) in $(path)")
    end
    Set(keys(values)) == Set(SOLVER_ORDER) || error("Solver rows are incomplete in $(path)")
    return values
end

function format_value(mean, stderr, digits; bold=false)
    format = Printf.Format("%.$(digits)f")
    value = "$(Printf.format(format, mean)) \\pm $(Printf.format(format, stderr))"
    return bold ? "\$\\mathbf{$(value)}\$" : "\$$(value)\$"
end

function player_panel(player, utilities, stderrs, digits)
    best_means = [
        maximum(utilities[solver][column] for solver in SOLVER_ORDER)
        for column in eachindex(BENCHMARK_ORDER)
    ]
    lines = String[
        "\\multicolumn{4}{c}{Tree-search player: Player $(player)} \\\\",
        "\\cmidrule(lr){1-4}",
        "Solver & " * join((BENCHMARK_LABELS[name] for name in BENCHMARK_ORDER), " & ") * " \\\\",
        "\\midrule",
    ]
    for solver in SOLVER_ORDER
        cells = [
            begin
                format_value(
                    utilities[solver][column],
                    stderrs[solver][column],
                    digits,
                    bold=utilities[solver][column] == best_means[column],
                )
            end
            for column in eachindex(BENCHMARK_ORDER)
        ]
        push!(lines, SOLVER_LABELS[solver] * " & " * join(cells, " & ") * " \\\\")
    end
    return lines
end

function make_table(cfg)
    tables = Dict{Int,Tuple{Dict{String,Vector{Float64}},Dict{String,Vector{Float64}}}}()
    for player in (1, 2)
        utilities = read_benchmark_csv(joinpath(cfg.data_dir, "player$(player)_utilities.csv"))
        stderrs = read_benchmark_csv(joinpath(cfg.data_dir, "player$(player)_stderrs.csv"))
        tables[player] = (utilities, stderrs)
    end

    lines = String[
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{$(cfg.caption)}",
        "\\label{$(cfg.label)}",
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
    ]
    append!(lines, player_panel(1, tables[1]..., cfg.digits))
    push!(lines, "\\midrule")
    append!(lines, player_panel(2, tables[2]..., cfg.digits))
    append!(lines, [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return join(lines, '\n') * "\n"
end

function main(args)
    cfg = parse_args(args)
    latex = make_table(cfg)
    mkpath(dirname(cfg.output))
    open(cfg.output, "w") do io
        write(io, latex)
    end
    println("Wrote LaTeX benchmark table: $(cfg.output)")
    return cfg.output
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
