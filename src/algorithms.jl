using Random
using Distributions
using LinearAlgebra
using Printf: @printf


function markovian_score_climbing_with_ibis_marginal_dynamics(
    nb_iter::Int,
    opt_state::NamedTuple,
    batch_size::Int,
    nb_steps::Int,
    nb_trajectories::Int,
    nb_particles::Int,
    init_state::Vector{Float64},
    learner::IBISClosedLoop,
    evaluator::IBISClosedLoop,
    param_prior::MultivariateDistribution,
    param_proposal::T,
    nb_ibis_moves::Int,
    action_penalty::Float64,
    slew_rate_penalty::Float64,
    tempering::Float64,
    reference::IBISReference,
    nb_csmc_moves::Int,
    verbose::Bool = false,
) where {T<:Function}

    all_returns = []

    # Evaluate the untrained policy.
    Flux.reset!(evaluator.ctl)
    state_struct, param_struct = smc_with_ibis_marginal_dynamics(
        nb_steps,
        nb_trajectories,
        nb_particles,
        init_state,
        evaluator,
        param_prior,
        param_proposal,
        nb_ibis_moves,
        action_penalty,
        slew_rate_penalty,
        0.0,
    )
    push!(all_returns, mean(state_struct.cumulative_return))

    if verbose
        @printf(
            "iter: %i, return: %0.4f, entropy: %0.4f\n",
            0,
            mean(state_struct.cumulative_return),
            policy_entropy(learner.ctl)
        )
    end

    for i = 1:nb_iter
        # Sampling step.
        for _ in 1:nb_csmc_moves
            Flux.reset!(learner.ctl)
            state_struct, param_struct = csmc_with_ibis_marginal_dynamics(
                nb_steps,
                nb_trajectories,
                nb_particles,
                init_state,
                learner,
                param_prior,
                param_proposal,
                nb_ibis_moves,
                action_penalty,
                slew_rate_penalty,
                tempering,
                reference
            )
            idx = rand(Categorical(state_struct.weights[:, end]))
            reference = IBISReference(
                state_struct.trajectories[:, :, idx],
                param_struct.particles[:, :, :, idx],
                param_struct.weights[:, :, idx],
                param_struct.log_weights[:, :, idx],
                param_struct.log_likelihoods[:, :, idx]
            )
        end

        idx = rand(Categorical(state_struct.weights[:, end]), nb_trajectories)
        samples = state_struct.trajectories[:, :, idx]

        # maximization step
        batcher = Flux.DataLoader(
            samples,
            batchsize=batch_size,
            shuffle=true
        )

        for _samples in batcher
            _, learner = maximization!(
                opt_state,
                _samples,
                learner
            )
        end

        # evaluation step
        Flux.reset!(evaluator.ctl)
        state_struct, _ = smc_with_ibis_marginal_dynamics(
            nb_steps,
            nb_trajectories,
            nb_particles,
            init_state,
            evaluator,
            param_prior,
            param_proposal,
            nb_ibis_moves,
            action_penalty,
            slew_rate_penalty,
            0.0,
        )
        push!(all_returns, mean(state_struct.cumulative_return))

        if verbose
            @printf(
                "iter: %i, return: %0.4f, entropy: %0.4f\n",
                i,
                mean(state_struct.cumulative_return),
                policy_entropy(learner.ctl)
            )
        end
    end
    return learner, all_returns
end


function score_climbing_with_rao_blackwell_marginal_dynamics(
    nb_iter::Int,
    opt_state::NamedTuple,
    batch_size::Int,
    nb_steps::Int,
    nb_trajectories::Int,
    init_state::Vector{Float64},
    learner::RaoBlackwellClosedLoop,
    evaluator::RaoBlackwellClosedLoop,
    param_prior::Gaussian,
    action_penalty::Float64,
    slew_rate_penalty::Float64,
    tempering::Float64,
    verbose::Bool = false,
)
    # sampling step
    Flux.reset!(learner.ctl)
    state_struct, _ = smc_with_rao_blackwell_marginal_dynamics(
        nb_steps,
        nb_trajectories,
        init_state,
        learner,
        param_prior,
        action_penalty,
        slew_rate_penalty,
        tempering,
    )
    idx = rand(Categorical(state_struct.weights), nb_trajectories)
    samples = state_struct.trajectories[:, :, idx]

    # evaluation step
    Flux.reset!(evaluator.ctl)
    state_struct, _ = smc_with_rao_blackwell_marginal_dynamics(
        nb_steps,
        nb_trajectories,
        init_state,
        evaluator,
        param_prior,
        action_penalty,
        slew_rate_penalty,
        0.0,
    )

    if verbose
        @printf(
            "iter: %i, return: %0.4f, entropy: %0.4f\n",
            0,
            state_struct.cumulative_return' * state_struct.weights,
            policy_entropy(learner.ctl)
        )
    end

    for i = 1:nb_iter
        # maximization step
        batcher = Flux.DataLoader(
            samples,
            batchsize=batch_size,
            shuffle=true
        )

        for _samples in batcher
            _, learner = maximization!(
                opt_state,
                _samples,
                learner
            )
        end

        # sampling step
        Flux.reset!(learner.ctl)
        state_struct, _ = smc_with_rao_blackwell_marginal_dynamics(
            nb_steps,
            nb_trajectories,
            init_state,
            learner,
            param_prior,
            action_penalty,
            slew_rate_penalty,
            tempering,
        )
        idx = rand(Categorical(state_struct.weights), nb_trajectories)
        samples = state_struct.trajectories[:, :, idx]

        # evaluation step
        Flux.reset!(evaluator.ctl)
        state_struct, _ = smc_with_rao_blackwell_marginal_dynamics(
            nb_steps,
            nb_trajectories,
            init_state,
            evaluator,
            param_prior,
            action_penalty,
            slew_rate_penalty,
            0.0,
        )

        if verbose
            @printf(
                "iter: %i, return: %0.4f, entropy: %0.4f\n",
                i,
                state_struct.cumulative_return' * state_struct.weights,
                policy_entropy(learner.ctl)
            )
        end
    end
    return learner, samples
end


function markovian_score_climbing_with_rao_blackwell_marginal_dynamics(
    nb_iter::Int,
    opt_state::NamedTuple,
    batch_size::Int,
    nb_steps::Int,
    nb_trajectories::Int,
    init_state::Vector{Float64},
    learner::RaoBlackwellClosedLoop,
    evaluator::RaoBlackwellClosedLoop,
    param_prior::Gaussian,
    action_penalty::Float64,
    slew_rate_penalty::Float64,
    tempering::Float64,
    reference::RaoBlackwellReference,
    nb_csmc_moves::Int,
    verbose::Bool = false,
)
    all_returns = []

    # sampling step
    Flux.reset!(learner.ctl)
    state_struct, param_struct = csmc_with_rao_blackwell_marginal_dynamics(
        nb_steps,
        nb_trajectories,
        init_state,
        learner,
        param_prior,
        action_penalty,
        slew_rate_penalty,
        tempering,
        reference
    )
    idx = rand(Categorical(state_struct.weights))
    reference = RaoBlackwellReference(
        state_struct.trajectories[:, :, idx],
        param_struct.distributions[:, idx]
    )

    for _ in 1:nb_csmc_moves - 1
        Flux.reset!(learner.ctl)
        state_struct, param_struct = csmc_with_rao_blackwell_marginal_dynamics(
            nb_steps,
            nb_trajectories,
            init_state,
            learner,
            param_prior,
            action_penalty,
            slew_rate_penalty,
            tempering,
            reference
        )
        idx = rand(Categorical(state_struct.weights))
        reference = RaoBlackwellReference(
            state_struct.trajectories[:, :, idx],
            param_struct.distributions[:, idx]
        )
    end
    idx = rand(Categorical(state_struct.weights), nb_trajectories)
    samples = state_struct.trajectories[:, :, idx]

    # evaluation step
    Flux.reset!(evaluator.ctl)
    state_struct, _ = smc_with_rao_blackwell_marginal_dynamics(
        nb_steps,
        nb_trajectories,
        init_state,
        evaluator,
        param_prior,
        action_penalty,
        slew_rate_penalty,
        0.0,
    )
    push!(all_returns, state_struct.cumulative_return' * state_struct.weights)

    if verbose
        @printf(
            "iter: %i, return: %0.4f, entropy: %0.4f\n",
            0,
            state_struct.cumulative_return' * state_struct.weights,
            policy_entropy(learner.ctl)
        )
    end

    for i = 1:nb_iter
        # maximization step
        batcher = Flux.DataLoader(
            samples,
            batchsize=batch_size,
            shuffle=true
        )

        for _samples in batcher
            _, learner = maximization!(
                opt_state,
                _samples,
                learner
            )
        end

        # sampling step
        Flux.reset!(learner.ctl)
        state_struct, param_struct = csmc_with_rao_blackwell_marginal_dynamics(
            nb_steps,
            nb_trajectories,
            init_state,
            learner,
            param_prior,
            action_penalty,
            slew_rate_penalty,
            tempering,
            reference
        )
        idx = rand(Categorical(state_struct.weights))
        reference = RaoBlackwellReference(
            state_struct.trajectories[:, :, idx],
            param_struct.distributions[:, idx]
        )

        for _ in 1:nb_csmc_moves - 1
            Flux.reset!(learner.ctl)
            state_struct, param_struct = csmc_with_rao_blackwell_marginal_dynamics(
                nb_steps,
                nb_trajectories,
                init_state,
                learner,
                param_prior,
                action_penalty,
                slew_rate_penalty,
                tempering,
                reference
            )
            idx = rand(Categorical(state_struct.weights))
            reference = RaoBlackwellReference(
                state_struct.trajectories[:, :, idx],
                param_struct.distributions[:, idx]
            )
        end
        idx = rand(Categorical(state_struct.weights), nb_trajectories)
        samples = state_struct.trajectories[:, :, idx]

        # evaluation step
        Flux.reset!(evaluator.ctl)
        state_struct, _ = smc_with_rao_blackwell_marginal_dynamics(
            nb_steps,
            nb_trajectories,
            init_state,
            evaluator,
            param_prior,
            action_penalty,
            slew_rate_penalty,
            0.0,
        )
        push!(all_returns, state_struct.cumulative_return' * state_struct.weights)

        if verbose
            @printf(
                "iter: %i, return: %0.4f, entropy: %0.4f\n",
                i,
                state_struct.cumulative_return' * state_struct.weights,
                policy_entropy(learner.ctl)
            )
        end
    end
    return learner, all_returns
end
