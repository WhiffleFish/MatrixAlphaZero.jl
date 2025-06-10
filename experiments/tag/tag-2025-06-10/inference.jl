begin
    using MatrixAlphaZero
    using MarkovGames
    const AZ = MatrixAlphaZero
    using ExperimentTools
    const Tools = ExperimentTools
    using Plots
    using POSGModels.DiscreteTag
end

game = TagMG(reward_model=DiscreteTag.DenseReward(peak=1.0))
oracle = AZ.load_oracle(@__DIR__)

S = states(game)
SV = mapreduce(hcat, S) do s
    MarkovGames.convert_s(Vector{Float32}, s, game)
end
AZ.value(oracle, SV)
