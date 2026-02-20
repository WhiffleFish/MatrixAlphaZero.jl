#=
Usage example
`julia new-experiment.jl dubin`
=#

#FIXME: shouldn't assume that script is being run from base directory -- prepend @__DIR__
#FIXME: account for more than 1 experiment run per day e.g. with appended --1, --2, etc.

using Dates

const EXP_DIR_PATTERN = r".*\d{4}-\d{2}-\d{2}"

const DEFAULT_EXCLUSIONS = [
    "models",
    "figures",
    "network_brvs.csv",
    "oracle.jld2",
    "train_info.jld2"
]

function copy_with_exclusions(src::String, dst::String)
    # Define excluded folders/filenames
    excluded_names = DEFAULT_EXCLUSIONS
    
    for (root, dirs, files) in walkdir(src)
        # Filter directories to skip
        filter!(d -> d ∉ excluded_names, dirs)
        
        # Create corresponding destination directory
        rel_path = relpath(root, src)
        dest_root = joinpath(dst, rel_path)
        mkpath(dest_root)
        
        for file in files
            if file ∉ excluded_names
                src_file = joinpath(root, file)
                dest_file = joinpath(dest_root, file)
                # Copy file, skipping if it already exists
                cp(src_file, dest_file, force=false)
            end
        end
    end
end


function new_experiment(name::String)
    experiment_dir = joinpath("experiments", name)
    @assert isdir(experiment_dir)
    lastdir = last(readdir(experiment_dir))
    @assert !isnothing(match(EXP_DIR_PATTERN, lastdir))
    newdir = name * "-" * string(Dates.today())
    copy_with_exclusions(joinpath(experiment_dir, lastdir), joinpath(experiment_dir, newdir))
end


if abspath(PROGRAM_FILE) == @__FILE__ 
    @assert isone(length(ARGS))
    new_experiment(ARGS[1])
end
