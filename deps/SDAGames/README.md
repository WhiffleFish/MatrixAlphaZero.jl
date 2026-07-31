# SDAGames

Make sure my fork of [SatelliteDynamics.jl](https://github.com/WhiffleFish/SatelliteDynamics.jl) is installed and updated.

# Usage
```julia
using SDAGames
using POMDPs

mdp = SDABMDP()
s = rand(initialstate(mdp))
sp = @gen(:sp)(mdp, s, 1)
```
