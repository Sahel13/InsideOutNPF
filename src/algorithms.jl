using Random
using Distributions
using LinearAlgebra
using Printf: @printf


function state_transition_and_potential(
    z::AbstractVector{Float64},
    zn::AbstractVector{Float64},
    ps::AbstractMatrix{Float64},
    dynamics::IBISDynamics,
    scratch::AbstractMatrix{Float64},
    eta::Float64
)
    """
    Computes the transition density of the state and the
    potential function for a single trajectory.
    """
    # TODO: Slew rate penatly is hardcoded at the moment.
    xdim = dynamics.xdim
    log_pdfs = ibis_conditional_dynamics_logpdf(
        dynamics, ps, z[1:xdim], zn[xdim+1:end], zn[1:xdim], scratch
    )
    info_gain = ibis_info_gain_increment(
        dynamics,
        ps,
        zeros(size(ps, 2)),
        z[1:xdim],
        zn[xdim+1:end],
        zn[1:xdim],
        scratch
    )
    slew_rate_penalty = 1e-1
    u = zn[xdim+1:end]
    up = z[xdim+1:end]
    reward = info_gain - slew_rate_penalty * dot(u - up, u - up)
    return (logsumexp(log_pdfs) - log(size(ps, 2))) + eta * reward
end


function design_logpdfs(
    state_struct::StateStruct,
    future_trajectory::Matrix{Float64},
    closed_loop::IBISClosedLoop,
    time_idx::Int,
    indices::Union{Nothing, Vector{Int}} = nothing
)
    """
    Computes the logpdfs of all future designs given the current trajectories.
    """
    if indices === nothing
        indices = 1:state_struct.nb_trajectories
    end

    # Construct the trajectories.
    trajectories = Array{Float64}(undef, state_struct.state_dim, state_struct.nb_steps + 1, length(indices))
    for (i, traj_idx) in enumerate(indices)
        # Get $z_{0:t}^n$ by tracing genealogies.
        trajectories[:, 1:time_idx, i] = genealogy_tracer(
            state_struct.unresampled_trajectories,
            state_struct.resampled_idx,
            time_idx,
            traj_idx
        )
        # $\bar{z}_{t+1:T}$ is the same for all trajectories.
        trajectories[:, time_idx + 1:end, i] = future_trajectory
    end

    # Feed the states $z_{0:t-1}^n$ to the policy.
    Flux.reset!(closed_loop.ctl)
    for t = 1:(time_idx - 1)
        _ = initialize_encoder(closed_loop.ctl, trajectories[:, t, :])
    end

    # Compute the densities $\pi(\bar{\xi}_s \mid z_{0:t}^n, \bar{z}_{t+1:s-1})$
    # for all $s = t+1, \dots, T$.
    xdim = closed_loop.dyn.xdim
    logpdfs = zeros(length(indices))
    for t = time_idx:state_struct.nb_steps
        zs = trajectories[:, t, :]
        us = trajectories[xdim+1:end, t+1, :]

        logpdfs += policy_logpdf(
            closed_loop.ctl, 
            zs,
            us
        )
    end
    return logpdfs
end


function theta_transition_density(
    prev_ps::AbstractMatrix{Float64},
    ps::AbstractMatrix{Float64},
    dynamics::IBISDynamics,
    z::AbstractVector{Float64},
    zn::AbstractVector{Float64},
    nb_particles::Int,
    scratch::AbstractMatrix{Float64}
)
    """
    Computes the transition probability of the theta particles for a single trajectory.
    The transition density is marginalized over the resampling indices.
    """
    # TODO: This currently only works for the case where the prior is a Gaussian.
    jittering_kernel = MvNormal(
        zeros(size(ps, 1)),
        nb_particles^(-3/2) * I
    )
    # Compute the weights for the particles, $W_{\theta, t}^nm$.
    # Since we use multinomial resampling, these are
    # also the probabilities of the resampling indices.
    xdim = dynamics.xdim
    log_weights = ibis_conditional_dynamics_logpdf(
        dynamics, prev_ps, z[1:xdim], zn[xdim+1:end], zn[1:xdim], scratch
    )
    log_weights .-= logsumexp(log_weights)

    out = 0
    log_pdfs = Vector{Float64}(undef, size(ps, 2))
    # Iterate over all current particles.
    for theta in eachcol(ps)
        # Compute the weighted sum of the transition probabilities from
        # all previous particles to the current particle. This corresponds
        # to marginalizing over the ancestors $B_t^m$.
        logpdf!(log_pdfs, jittering_kernel, theta .- prev_ps)
        out += logsumexp(log_pdfs .+ log_weights)
    end
    return out
end


function backward_sampling(
    state_struct::StateStruct,
    param_struct::IBISParamStruct,
    closed_loop::IBISClosedLoop,
    eta::Float64,
    idx::Union{Int, Nothing} = nothing
)
    """
    Standard backward sampling from Godsill et al. (2004).
    Returns a single trajectory.
    """
    trajectory = Matrix{Float64}(undef, state_struct.state_dim, state_struct.nb_steps + 1)
    traj_indices = Vector{Int}(undef, state_struct.nb_steps + 1)

    # Sample a particle at the final time step.
    if idx === nothing
        idx = rand(Categorical(state_struct.weights[:, end]))
    end
    traj_indices[end] = idx
    trajectory[:, end] = state_struct.unresampled_trajectories[:, end, idx]

    # Work our way backwards.
    for t = state_struct.nb_steps:-1:1
        # Compute the probability of all future designs.
        design_densities = design_logpdfs(
            state_struct,
            trajectory[:, t + 1:end],
            closed_loop,
            t
        )
        # Compute the potential function and the state transition density.
        dynamics = closed_loop.dyn
        zs = state_struct.unresampled_trajectories[:, t, :]
        zn = trajectory[:, t + 1]
        x_prob_and_pot = map(enumerate(eachcol(zs))) do (n, z)
            ps = param_struct.raw_particles[:, t, :, n]
            state_transition_and_potential(
                z, zn, ps, dynamics, view(param_struct.scratch, :, :, n), eta
            )
        end
        # Compute the transition probability of the theta particles.
        theta_prob = map(enumerate(eachcol(zs))) do (n, z)
            theta_transition_density(
                param_struct.raw_particles[:, t, :, n],
                param_struct.raw_particles[:, t + 1, :, idx],
                dynamics,
                z,
                zn,
                param_struct.nb_particles,
                view(param_struct.scratch, :, :, n)
            )
        end

        # Use the filtering densities to reweight the probabilities and get the smoothing weights.
        reweighting_ratio = x_prob_and_pot .+ design_densities .+ theta_prob
        smoothing_weights = softmax(reweighting_ratio .+ log.(state_struct.weights[:, t]))
        # Sample a particle.
        idx = rand(Categorical(smoothing_weights))
        traj_indices[t] = idx
        trajectory[:, t] = state_struct.unresampled_trajectories[:, t, idx]
    end
    return trajectory, traj_indices
end


function backward_sampling_mcmc(
    state_struct::StateStruct,
    param_struct::IBISParamStruct,
    closed_loop::IBISClosedLoop,
    eta::Float64,
    idx::Union{Int, Nothing} = nothing
)
    """
    MCMC-based backward sampling from Bunch and Godsill (2013).
    This is the recommended smoothing algorithm by Dau and Chopin (2023).
    Returns a single trajectory.
    Reference for implementation:
    https://github.com/nchopin/particles/blob/841cf363b3f1dee0faa77f6a0349ace3477917ab/particles/smoothing.py#L313
    """
    trajectory = Matrix{Float64}(undef, state_struct.state_dim, state_struct.nb_steps + 1)
    traj_indices = Vector{Int}(undef, state_struct.nb_steps + 1)

    # Sample a particle at the final time step.
    if idx === nothing
        idx = rand(Categorical(state_struct.weights[:, end]))
    end
    traj_indices[end] = idx
    trajectory[:, end] = state_struct.unresampled_trajectories[:, end, idx]

    # Work our way backwards.
    for t = state_struct.nb_steps:-1:1
        ancestor_idx = state_struct.resampled_idx[idx, t]
        proposed_idx = rand(Categorical(state_struct.weights[:, t]))
        indices = [ancestor_idx, proposed_idx]

        # Compute the probability of all future designs.
        design_densities = design_logpdfs(
            state_struct,
            trajectory[:, t + 1:end],
            closed_loop,
            t,
            indices
        )
        # Compute the potential function and the state transition density.
        dynamics = closed_loop.dyn
        zs = state_struct.unresampled_trajectories[:, t, indices]
        zn = trajectory[:, t + 1]
        x_prob_and_pot = map(indices, eachcol(zs)) do anc, z
            ps = param_struct.raw_particles[:, t, :, anc]
            state_transition_and_potential(
                z, zn, ps, dynamics, view(param_struct.scratch, :, :, anc), eta
            )
        end
        ##  Compute the transition probability of the theta particles.
        theta_prob = map(indices, eachcol(zs)) do anc, z
            theta_transition_density(
                param_struct.raw_particles[:, t, :, anc],
                param_struct.raw_particles[:, t + 1, :, idx],
                dynamics,
                z,
                zn,
                param_struct.nb_particles,
                view(param_struct.scratch, :, :, anc)
            )
        end

        reweighting_ratio = x_prob_and_pot .+ design_densities .+ theta_prob
        lpr_acc = reweighting_ratio[2] - reweighting_ratio[1]
        lu = log(rand())
        idx = lpr_acc > lu ? proposed_idx : ancestor_idx

        traj_indices[t] = idx
        trajectory[:, t] = state_struct.unresampled_trajectories[:, t, idx]
    end
    return trajectory, traj_indices
end


function backward_sampling_batched(
    state_struct::StateStruct,
    param_struct::IBISParamStruct,
    closed_loop::IBISClosedLoop,
    eta::Float64,
    num_trajs::Int
)
    """
    Backward sampling function that returns `num_trajs` trajectories.
    Current default is to use MCMC-based backward sampling.
    """
    indices = rand(Categorical(state_struct.weights[:, end]), num_trajs)
    trajectories = Array{Float64}(undef, state_struct.state_dim, state_struct.nb_steps + 1, num_trajs)
    Threads.@threads for traj_idx = 1:num_trajs
        trajectories[:, :, traj_idx], _ = backward_sampling_mcmc(
            state_struct,
            param_struct,
            closed_loop,
            eta,
            indices[traj_idx]
        )
        println("Trajectory $traj_idx sampled")
    end
    return trajectories
end


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
    action_penalty::Float64,
    slew_rate_penalty::Float64,
    tempering::Float64,
    reference::IBISReference,
    nb_csmc_moves::Int,
    backward_sample::Bool = false,
    param_proposal::Union{T, Nothing} = nothing,
    nb_ibis_moves::Union{Int, Nothing} = nothing,
    verbose::Bool = false,
    nb_trajectories_eval::Int = 256,
    nb_particles_eval::Int = 256,
    nb_backward_samples::Int = 32
) where {T<:Function}

    if !backward_sample
        if param_proposal === nothing || nb_ibis_moves === nothing
            throw(ArgumentError("param_proposal and nb_ibis_moves must be provided."))
        end
    end

    all_returns = []

    # Evaluate the untrained policy.
    # Evaluation is always done using IBIS.
    Flux.reset!(evaluator.ctl)
    state_struct, param_struct = smc_with_ibis_marginal_dynamics(
        nb_steps,
        nb_trajectories_eval,
        nb_particles_eval,
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
                reference,
                !backward_sample
            )
            if backward_sample
                trajectory, traj_indices = backward_sampling_mcmc(
                    state_struct,
                    param_struct,
                    learner,
                    tempering
                )
                theta_particles = genealogy_tracer(
                    param_struct,
                    traj_indices,
                    nb_steps
                )
                placeholder = Matrix{Float64}(undef, state_struct.state_dim, nb_steps + 1)
                reference = IBISReference(
                    trajectory,
                    theta_particles,
                    # These are not used.
                    placeholder,
                    placeholder,
                    placeholder
                )
            else
                idx = rand(Categorical(state_struct.weights[:, end]))
                reference = IBISReference(
                    state_struct.trajectories[:, :, idx],
                    param_struct.particles[:, :, :, idx],
                    param_struct.weights[:, :, idx],
                    param_struct.log_weights[:, :, idx],
                    param_struct.log_likelihoods[:, :, idx]
                )
            end
        end

        if backward_sample
            trajectories = backward_sampling_batched(
                state_struct,
                param_struct,
                learner,
                tempering,
                nb_backward_samples
            )
            samples = trajectories
        else
            idx = rand(Categorical(state_struct.weights[:, end]), nb_backward_samples)
            samples = state_struct.trajectories[:, :, idx]
        end

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

        # Evaluation step.
        Flux.reset!(evaluator.ctl)
        state_struct, _ = smc_with_ibis_marginal_dynamics(
            nb_steps,
            nb_trajectories_eval,
            nb_particles_eval,
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
