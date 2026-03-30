function parse_commandline(;
        iter            = 10,
        steps_per_iter  = 20_000,
        tree_queries    = 20,
        max_depth       = 50,
        runs            = nothing,
        checkpoint      = nothing,
        every           = nothing,
    )
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--test"
            help        = "run file test procedure (2 procs, 3 iters)"
            action      = :store_true
        "--addprocs"
            help = "add n more processes"
            arg_type    = Int
            default     = 19
        "--iter"
            arg_type    = Int
            default     = iter
        "--steps_per_iter"
            arg_type    = Int
            default     = steps_per_iter
        "--tree_queries"
            arg_type    = Int
            default     = tree_queries
        "--max_depth"
            arg_type    = Int
            default     = max_depth
    end

    if !isnothing(runs)
        @add_arg_table! s begin
            "--runs"
                arg_type = Int
                default = runs
        end
    end

    if !isnothing(checkpoint)
        @add_arg_table! s begin
            "--checkpoint"
                arg_type = Int
                default = checkpoint
        end
    end

    if !isnothing(every)
        @add_arg_table! s begin
            "--every"
                arg_type = Int
                default = every
        end
    end

    parsed_args = parse_args(s)
    if parsed_args["test"]
        parsed_args["addprocs"] = 1
        parsed_args["iter"] = 1
        parsed_args["steps_per_iter"] = 2
        parsed_args["tree_queries"] = 1
        parsed_args["max_depth"] = 10
        if !isnothing(runs)
            parsed_args["runs"] = 2
        end
        if !isnothing(checkpoint)
            parsed_args["checkpoint"] = 1
        end
        if !isnothing(every)
            parsed_args["every"] = 1
        end
    end
    return parsed_args
end
