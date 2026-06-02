function parse_commandline(;
        max_steps       = 10_000_000,
        num_steps       = 2048 * 8,
        update_epochs   = 1,
        num_batches     = 1,
        tree_queries    = 20,
        max_depth       = 50,
        sim_depth      = nothing,
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
        "--max_steps"
            arg_type    = Int
            default     = max_steps
        "--num_steps"
            arg_type    = Int
            default     = num_steps
        "--update_epochs"
            arg_type    = Int
            default     = update_epochs
        "--num_batches"
            arg_type    = Int
            default     = num_batches
        "--tree_queries"
            arg_type    = Int
            default     = tree_queries
        "--max_depth"
            arg_type    = Int
            default     = max_depth
    end

    if !isnothing(sim_depth)
        @add_arg_table! s begin
            "--sim_depth"
                arg_type = Int
                default = sim_depth
        end
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
        parsed_args["max_steps"] = 2
        parsed_args["num_steps"] = 2
        parsed_args["update_epochs"] = 1
        parsed_args["num_batches"] = 1
        parsed_args["tree_queries"] = 1
        parsed_args["max_depth"] = min(max_depth, 10)
        if !isnothing(sim_depth)
            parsed_args["sim_depth"] = min(sim_depth, 10)
        end
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
