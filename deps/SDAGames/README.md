# SDAGames

# Usage
```julia
using SDAGames
using POMDPs

mdp = SDABMDP()
s = rand(initialstate(mdp))
sp = @gen(:sp)(mdp, s, 1)
```
