@kwdef struct PPOBestResponseConfig
    total_timesteps::Int = 500_000
    num_steps::Int = 128
    num_envs::Int = 4
    num_minibatches::Int = 4
    update_epochs::Int = 4
    lr::Float32 = 2.5f-4
    gamma::Float32 = 0.99f0
    gae_lambda::Float32 = 0.95f0
    clip_coef::Float32 = 0.2f0
    ent_coeff::Float32 = 0.01f0
    v_coef::Float32 = 0.5f0
    normalize_advantages::Bool = true
    clip_value_loss::Bool = true
    anneal_lr::Bool = true
    max_steps::Union{Nothing,Int} = 50
    seed::Int = 0
    name::String = "ppo_best_response"
    log_dir::String = "logs"
end

struct ZeroSearchOracle
    na::NTuple{2,Int}
end

ZeroSearchOracle(game::MG) = ZeroSearchOracle(Tuple(length.(actions(game))))

_uniform_pair(oracle::ZeroSearchOracle) = (
    fill(1.0f0 / oracle.na[1], oracle.na[1]),
    fill(1.0f0 / oracle.na[2], oracle.na[2]),
)

AZ.state_value(::ZeroSearchOracle, game, s) = 0.0
AZ.batch_state_value(::ZeroSearchOracle, game, states) = fill(0.0, length(states))
AZ.value(::ZeroSearchOracle, x::AbstractVector) = Float32[0.0]
AZ.value(::ZeroSearchOracle, x::AbstractMatrix) = zeros(Float32, 1, size(x, 2))
AZ.state_policy(oracle::ZeroSearchOracle, game, s) = _uniform_pair(oracle)
AZ.batch_state_policy(oracle::ZeroSearchOracle, game, states) = (
    fill(1.0f0 / oracle.na[1], oracle.na[1], length(states)),
    fill(1.0f0 / oracle.na[2], oracle.na[2], length(states)),
)
AZ.state_strategy(oracle::ZeroSearchOracle, game, s) = _uniform_pair(oracle)
AZ.batch_state_strategy(oracle::ZeroSearchOracle, game, states) = (
    fill(1.0f0 / oracle.na[1], oracle.na[1], length(states)),
    fill(1.0f0 / oracle.na[2], oracle.na[2], length(states)),
)
AZ.state_regret(oracle::ZeroSearchOracle, game, s) =
    (zeros(Float32, oracle.na[1]), zeros(Float32, oracle.na[2]))
AZ.batch_state_regret(oracle::ZeroSearchOracle, game, states) =
    (zeros(Float32, oracle.na[1], length(states)), zeros(Float32, oracle.na[2], length(states)))

struct OracleStrategyPolicy{G,O} <: Policy
    game::G
    oracle::O
end

function MarkovGames.behavior(policy::OracleStrategyPolicy, s)
    game = policy.game
    A1, A2 = actions(game)
    x, y = AZ.state_strategy(policy.oracle, game, s)
    return ProductDistribution(
        SparseCat(A1, AZ.normalized_or_uniform(Float64.(x))),
        SparseCat(A2, AZ.normalized_or_uniform(Float64.(y))),
    )
end

struct ProjectedPlayerPolicy{P} <: Policy
    policy::P
    player::Int
end

MarkovGames.behavior(policy::ProjectedPlayerPolicy, s) =
    behavior(policy.policy, s)[policy.player]

struct ActorPlayerPolicy{G,A} <: Policy
    game::G
    player::Int
    actor::A
end

function actor_probs(actor, game, s)
    obs = MarkovGames.convert_s(Vector{Float32}, s, game)
    return vec(Flux.softmax(actor(obs)))
end

function MarkovGames.behavior(policy::ActorPlayerPolicy, s)
    probs = Float64.(actor_probs(policy.actor, policy.game, s))
    return SparseCat(actions(policy.game)[policy.player], probs)
end

struct PPOBestResponseMDP{G,P,S,A,D} <: POMDPs.MDP{S,A}
    game::G
    fixed_policy::P
    br_player::Int
    action_list::Vector{A}
    initialstate_dist::D

    function PPOBestResponseMDP(
            game::G,
            fixed_policy::P,
            br_player::Int;
            initialstate_dist = initialstate(game),
        ) where {G<:MG,P}
        br_player in (1, 2) || throw(ArgumentError("br_player must be 1 or 2"))
        S = statetype(game)
        action_list = collect(actions(game)[br_player])
        isempty(action_list) && throw(ArgumentError("BR player has no actions"))
        A = eltype(action_list)
        D = typeof(initialstate_dist)
        return new{G,P,S,A,D}(game, fixed_policy, br_player, action_list, initialstate_dist)
    end
end

POMDPs.actions(mdp::PPOBestResponseMDP) = mdp.action_list
POMDPs.actionindex(mdp::PPOBestResponseMDP, a) = findfirst(==(a), mdp.action_list)
POMDPs.discount(mdp::PPOBestResponseMDP) = discount(mdp.game)
POMDPs.initialstate(mdp::PPOBestResponseMDP) = mdp.initialstate_dist
POMDPs.isterminal(mdp::PPOBestResponseMDP, s) = isterminal(mdp.game, s)
POMDPs.convert_s(::Type{Vector{Float32}}, s, mdp::PPOBestResponseMDP) =
    MarkovGames.convert_s(Vector{Float32}, s, mdp.game)

player_reward(player::Int, r::Number) = isone(player) ? Float64(r) : -Float64(r)
player_reward(player::Int, r) = Float64(r[player])

function POMDPs.gen(mdp::PPOBestResponseMDP, s, br_action, rng::AbstractRNG=Random.default_rng())
    fixed_player = MarkovGames.other_player(mdp.br_player)
    fixed_action = rand(rng, behavior(mdp.fixed_policy, s)[fixed_player])
    joint_action = isone(mdp.br_player) ? (br_action, fixed_action) : (fixed_action, br_action)
    sp, r = @gen(:sp, :r)(mdp.game, s, joint_action, rng)
    return (; sp, r = player_reward(mdp.br_player, r))
end

function _cleanrl_ppo_config(config::PPOBestResponseConfig)
    return CleanRL.PPOConfig(;
        total_timesteps = config.total_timesteps,
        num_steps = config.num_steps,
        num_envs = config.num_envs,
        num_minibatches = config.num_minibatches,
        update_epochs = config.update_epochs,
        lr = config.lr,
        gamma = config.gamma,
        gae_lambda = config.gae_lambda,
        clip_coef = config.clip_coef,
        ent_coeff = config.ent_coeff,
        v_coef = config.v_coef,
        normalize_advantages = config.normalize_advantages,
        clip_value_loss = config.clip_value_loss,
        anneal_lr = config.anneal_lr,
        name = config.name,
        log_dir = config.log_dir,
    )
end

function _ppo_env_factory(mdp::PPOBestResponseMDP, config::PPOBestResponseConfig)
    next_seed = Ref(config.seed)
    return function ()
        next_seed[] += 1
        local_mdp = deepcopy(mdp)
        return CleanRL.MDPEnv(
            local_mdp;
            rng = Random.MersenneTwister(next_seed[]),
            max_steps = config.max_steps,
            state_encoder = s -> POMDPs.convert_s(Vector{Float32}, s, local_mdp),
        )
    end
end

function train_ppo_best_response(
        game::MG,
        fixed_policy::Policy,
        br_player::Int;
        config::PPOBestResponseConfig = PPOBestResponseConfig(),
        initialstate_dist = initialstate(game),
    )
    mdp = PPOBestResponseMDP(game, fixed_policy, br_player; initialstate_dist)
    Random.seed!(config.seed)
    actor, critic = CleanRL.ppo(_ppo_env_factory(mdp, config), _cleanrl_ppo_config(config))
    return (; actor, critic, mdp)
end

function ppo_best_response_joint_policy(
        game::MG,
        fixed_policy::Policy,
        actor,
        br_player::Int,
    )
    actor_policy = ActorPlayerPolicy(game, br_player, actor)
    fixed = ProjectedPlayerPolicy(fixed_policy, MarkovGames.other_player(br_player))
    return isone(br_player) ? JointPolicy(actor_policy, fixed) : JointPolicy(fixed, actor_policy)
end
