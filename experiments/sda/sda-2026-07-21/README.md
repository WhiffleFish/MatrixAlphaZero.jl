# SDA 2026-07-21: correlated LEO regime

This experiment narrows the SDA task to one declared in-distribution orbital
regime instead of averaging over the full July 20 altitude/phase mixture.

## Initial-state distribution

- target altitude: 600--1200 km
- observer altitude relative to target: -300--300 km
- absolute observer/target phase separation: 10--120 degrees
- separation sign: uniform
- target phase: uniform around the orbit
- circular, coplanar orbits

The training distribution, fixed periodic-evaluation bank, and primary final
benchmarks must all use `core_initialstate_distribution(game)` from
`initial_state.jl`. The old broad distribution can be reported separately as
an out-of-distribution stress test, but it is not part of the primary table.

## Changes from `sda-2026-07-20`

- network width increases from 32 to 64 for each independent value, regret,
  and average-strategy network;
- regret-matching backups change from `:sample` to `:mean`;
- training still uses no regret, strategy, count, or value prior mass;
- deployed tree search uses 100 queries, search epsilon 0.1, action epsilon
  0.0, and compares prior scale 100 against prior scale 0;
- primary evaluation is restricted to the correlated LEO distribution above.

Width 64 is a measured capacity increase, not a claim that capacity was the
only failure mode. It is small compared with the cost of the 500-query search,
while giving each dense network roughly four times the hidden-layer parameter
count. The no-transfer comparison remains necessary to isolate whether learned
priors actually help.

## Launch

From the repository root:

```sh
julia --project=experiments experiments/sda/sda-2026-07-21/train.jl --addprocs 16
```
