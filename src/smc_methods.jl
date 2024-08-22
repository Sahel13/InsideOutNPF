using Random


function smc_step_with_ibis_marginal_dynamics!(
    time_idx::Int,
    closedloop::IBISClosedLoop,
    action_penalty::Float64,
    slew_rate_penalty::Float64,
    tempering::Float64,
    state_struct::StateStruct,
    param_struct::IBISParamStruct
)
    if effective_sample_size(state_struct.weights) < 0.75 * state_struct.nb_trajectories
        # Get resampled indices
        systematic_resampling!(state_struct, time_idx)
        resampling_indices = state_struct.resampled_idx[:, time_idx]

        # Resample trajectories
        state_struct.weights .= 1 / state_struct.nb_trajectories
        state_struct.log_weights .= 0.0
        state_struct.log_weights_increment .= 0.0
        state_struct.trajectories .= @view state_struct.trajectories[:, :, resampling_indices]
        state_struct.cumulative_return .= @view state_struct.cumulative_return[resampling_indices]

        # Resample particles
        param_struct.particles .= @view param_struct.particles[:, :, :, resampling_indices]
        param_struct.weights .= @view param_struct.weights[:, :, resampling_indices]
        param_struct.log_weights .= @view param_struct.log_weights[:, :, resampling_indices]
        param_struct.log_likelihoods .= @view param_struct.log_likelihoods[:, :, resampling_indices]

        # Resample hidden states
        if time_idx > 1
            if closedloop.ctl isa StatefulStochasticPolicy
                for layer in closedloop.ctl.encoder_fn
                    if layer isa Flux.Recur
                        if layer.cell isa Flux.GRUCell
                            layer.state .= layer.state[:, resampling_indices]
                        elseif layer.cell isa Flux.LSTMCell
                            layer.state[1] .= layer.state[1][:, resampling_indices]
                            layer.state[2] .= layer.state[2][:, resampling_indices]
                        end
                    end
                end
            end
        end
    else
        state_struct.resampled_idx[:, time_idx] = 1:state_struct.nb_trajectories
    end

    # Propagate
    ibis_marginal_closedloop_sample!(
        closedloop,
        view(param_struct.particles, :, time_idx, :, :),
        view(param_struct.weights, time_idx, :, :),
        view(state_struct.trajectories, :, time_idx, :),
        view(state_struct.trajectories, :, time_idx+1, :)
    )
    state_struct.unresampled_trajectories[:, time_idx+1, :] .= state_struct.trajectories[:, time_idx+1, :]

    # Weights
    @views @inbounds for n = 1:state_struct.nb_trajectories
        u = state_struct.trajectories[closedloop.dyn.xdim+1:end, time_idx+1, n]
        up = state_struct.trajectories[closedloop.dyn.xdim+1:end, time_idx, n]

        info_gain = ibis_info_gain_increment(
            closedloop.dyn,
            param_struct.particles[:, time_idx, :, n],
            param_struct.log_weights[time_idx, :, n],
            state_struct.trajectories[begin:closedloop.dyn.xdim, time_idx, n],
            state_struct.trajectories[closedloop.dyn.xdim+1:end, time_idx+1, n],
            state_struct.trajectories[begin:closedloop.dyn.xdim, time_idx+1, n],
            param_struct.scratch[:, :, n]
        )
        reward = info_gain - action_penalty * dot(u, u) - slew_rate_penalty * dot(u - up, u - up)
        state_struct.log_weights[n] += tempering * reward
        state_struct.log_weights_increment[n] = tempering * reward
        state_struct.cumulative_return[n] += info_gain
    end
    state_struct.log_evidence += logsumexp(state_struct.log_weights_increment) - log(state_struct.nb_trajectories)

    # Normalize the weights
    normalize_weights!(state_struct)
end


function csmc_step_with_ibis_marginal_dynamics!(
    time_idx::Int,
    closedloop::IBISClosedLoop,
    action_penalty::Float64,
    slew_rate_penalty::Float64,
    tempering::Float64,
    reference::IBISReference,
    state_struct::StateStruct,
    param_struct::IBISParamStruct,
)
    if effective_sample_size(state_struct.weights) < 0.75 * state_struct.nb_trajectories
        # Get resampled indices
        multinomial_resampling!(state_struct, time_idx)
        state_struct.resampled_idx[1, time_idx] = 1
        resampling_indices = state_struct.resampled_idx[:, time_idx]

        # Resample trajectories
        state_struct.weights .= 1 / state_struct.nb_trajectories
        state_struct.log_weights .= 0.0
        state_struct.log_weights_increment .= 0.0
        state_struct.trajectories .= @view state_struct.trajectories[:, :, resampling_indices]
        state_struct.cumulative_return .= @view state_struct.cumulative_return[resampling_indices]

        # Resample particles
        param_struct.particles .= @view param_struct.particles[:, :, :, resampling_indices]
        param_struct.weights .= @view param_struct.weights[:, :, resampling_indices]
        param_struct.log_weights .= @view param_struct.log_weights[:, :, resampling_indices]
        param_struct.log_likelihoods .= @view param_struct.log_likelihoods[:, :, resampling_indices]

        # Resample hidden states
        if time_idx > 1
            if closedloop.ctl isa StatefulStochasticPolicy
                for layer in closedloop.ctl.encoder_fn
                    if layer isa Flux.Recur
                        if layer.cell isa Flux.GRUCell
                            layer.state .= layer.state[:, resampling_indices]
                        elseif layer.cell isa Flux.LSTMCell
                            layer.state[1] .= layer.state[1][:, resampling_indices]
                            layer.state[2] .= layer.state[2][:, resampling_indices]
                        end
                    end
                end
            end
        end
    else
        state_struct.resampled_idx[:, time_idx] = 1:state_struct.nb_trajectories
    end

    # Propagate
    ibis_marginal_closedloop_sample!(
        closedloop,
        view(param_struct.particles, :, time_idx, :, :),
        view(param_struct.weights, time_idx, :, :),
        view(state_struct.trajectories, :, time_idx, :),
        view(state_struct.trajectories, :, time_idx+1, :)
    )
    state_struct.trajectories[:, time_idx+1, 1] .= reference.trajectory[:, time_idx+1]
    state_struct.unresampled_trajectories[:, time_idx+1, :] .= state_struct.trajectories[:, time_idx+1, :]

    @views @inbounds for n = 1:state_struct.nb_trajectories
        xdim = closedloop.dyn.xdim
        u = state_struct.trajectories[xdim+1:end, time_idx+1, n]
        up = state_struct.trajectories[xdim+1:end, time_idx, n]

        info_gain = ibis_info_gain_increment(
            closedloop.dyn,
            param_struct.particles[:, time_idx, :, n],
            param_struct.log_weights[time_idx, :, n],
            state_struct.trajectories[begin:xdim, time_idx, n],
            state_struct.trajectories[xdim+1:end, time_idx+1, n],
            state_struct.trajectories[begin:xdim, time_idx+1, n],
            param_struct.scratch[:, :, n]
        )
        reward = info_gain - action_penalty * dot(u, u) - slew_rate_penalty * dot(u - up, u - up)
        state_struct.log_weights[n] += tempering * reward
        state_struct.log_weights_increment[n] = tempering * reward
        state_struct.cumulative_return[n] += info_gain
    end
    state_struct.log_evidence += logsumexp(state_struct.log_weights_increment) - log(state_struct.nb_trajectories)

    # Normalize the weights
    normalize_weights!(state_struct)
end


function smc_step_with_rao_blackwell_marginal_dynamics!(
    time_idx::Int,
    closedloop::RaoBlackwellClosedLoop,
    action_penalty::Float64,
    slew_rate_penalty::Float64,
    tempering::Float64,
    state_struct::StateStruct,
    param_struct::RaoBlackwellParamStruct
)
    if effective_sample_size(state_struct.weights) < 0.75 * state_struct.nb_trajectories
        # Get resampled indices
        systematic_resampling!(state_struct)

        # Resample trajectories
        state_struct.weights .= 1 / state_struct.nb_trajectories
        state_struct.log_weights .= 0.0
        state_struct.log_weights_increment .= 0.0
        state_struct.trajectories .= state_struct.trajectories[:, :, state_struct.resampled_idx]
        state_struct.cumulative_return .= state_struct.cumulative_return[state_struct.resampled_idx]

        # Resample posteriors
        param_struct.distributions .= param_struct.distributions[:, state_struct.resampled_idx]

        # Resample hidden states
        if time_idx > 1
            if closedloop.ctl isa StatefulStochasticPolicy
                for layer in closedloop.ctl.encoder_fn
                    if layer isa Flux.Recur
                        if layer.cell isa Flux.GRUCell
                            layer.state .= layer.state[:, state_struct.resampled_idx]
                        elseif layer.cell isa Flux.LSTMCell
                            layer.state[1] .= layer.state[1][:, state_struct.resampled_idx]
                            layer.state[2] .= layer.state[2][:, state_struct.resampled_idx]
                        end
                    end
                end
            end
        end
    end

    # Propagate
    state_struct.trajectories[:, time_idx+1, :] = rao_blackwell_marginal_closedloop_sample(
        closedloop,
        param_struct.distributions[time_idx, :],
        state_struct.trajectories[:, time_idx, :]
    )

    # Weights
    @views @inbounds for n = 1:state_struct.nb_trajectories
        q = param_struct.distributions[time_idx, n]
        x = state_struct.trajectories[1:closedloop.dyn.xdim, time_idx, n]
        u = state_struct.trajectories[closedloop.dyn.xdim+1:end, time_idx+1, n]
        xn = state_struct.trajectories[1:closedloop.dyn.xdim, time_idx+1, n]
        up = state_struct.trajectories[closedloop.dyn.xdim+1:end, time_idx, n]

        info_gain = rao_blackwell_info_gain_increment(closedloop.dyn, q, x, u, xn)
        reward = info_gain - action_penalty * dot(u, u) - slew_rate_penalty * dot(u - up, u - up)
        state_struct.log_weights[n] += tempering * reward
        state_struct.log_weights_increment[n] = tempering * reward
        state_struct.cumulative_return[n] += info_gain
    end
    state_struct.log_evidence += logsumexp(state_struct.log_weights_increment) - log(state_struct.nb_trajectories)

    # Normalize the weights
    normalize_weights!(state_struct)
    # println(effective_sample_size(state_struct.weights))
end


function csmc_step_with_rao_blackwell_marginal_dynamics!(
    time_idx::Int,
    closedloop::RaoBlackwellClosedLoop,
    action_penalty::Float64,
    slew_rate_penalty::Float64,
    tempering::Float64,
    reference::RaoBlackwellReference,
    state_struct::StateStruct,
    param_struct::RaoBlackwellParamStruct
)
    if effective_sample_size(state_struct.weights) < 0.75 * state_struct.nb_trajectories
        # Get resampled indices
        multinomial_resampling!(state_struct)
        state_struct.resampled_idx[1] = 1

        # Resample trajectories
        state_struct.weights .= 1 / state_struct.nb_trajectories
        state_struct.log_weights .= 0.0
        state_struct.log_weights_increment .= 0.0
        state_struct.trajectories .= state_struct.trajectories[:, :, state_struct.resampled_idx]
        state_struct.cumulative_return .= state_struct.cumulative_return[state_struct.resampled_idx]

        # Resample posteriors
        param_struct.distributions .= param_struct.distributions[:, state_struct.resampled_idx]

        # Resample hidden states
        if time_idx > 1
            if closedloop.ctl isa StatefulStochasticPolicy
                for layer in closedloop.ctl.encoder_fn
                    if layer isa Flux.Recur
                        if layer.cell isa Flux.GRUCell
                            layer.state .= layer.state[:, state_struct.resampled_idx]
                        elseif layer.cell isa Flux.LSTMCell
                            layer.state[1] .= layer.state[1][:, state_struct.resampled_idx]
                            layer.state[2] .= layer.state[2][:, state_struct.resampled_idx]
                        end
                    end
                end
            end
        end
    end

    # Propagate
    state_struct.trajectories[:, time_idx+1, :] = rao_blackwell_marginal_closedloop_sample(
        closedloop,
        param_struct.distributions[time_idx, :],
        state_struct.trajectories[:, time_idx, :]
    )
    state_struct.trajectories[:, time_idx+1, 1] .= reference.trajectory[:, time_idx+1]

    # Weights
    @views @inbounds for n = 1:state_struct.nb_trajectories
        q = param_struct.distributions[time_idx, n]
        x = state_struct.trajectories[1:closedloop.dyn.xdim, time_idx, n]
        u = state_struct.trajectories[closedloop.dyn.xdim+1:end, time_idx+1, n]
        xn = state_struct.trajectories[1:closedloop.dyn.xdim, time_idx+1, n]
        up = state_struct.trajectories[closedloop.dyn.xdim+1:end, time_idx, n]

        info_gain = rao_blackwell_info_gain_increment(closedloop.dyn, q, x, u, xn)
        reward = info_gain - action_penalty * dot(u, u) - slew_rate_penalty * dot(u - up, u - up)
        state_struct.log_weights[n] += tempering * reward
        state_struct.log_weights_increment[n] = tempering * reward
        state_struct.cumulative_return[n] += info_gain
    end
    state_struct.log_evidence += logsumexp(state_struct.log_weights_increment) - log(state_struct.nb_trajectories)

    # Normalize the weights
    normalize_weights!(state_struct)
end


function myopic_smc_step_with_ibis_marginal_dynamics!(
    time_idx::Int,
    adaptive_loop::IBISAdaptiveLoop,
    state_struct::StateStruct,
    param_struct::IBISParamStruct
)
    # Propagate
    state_struct.trajectories[:, time_idx+1, :] = ibis_marginal_adaptive_loop_sample(
        adaptive_loop,
        view(param_struct.particles, :, time_idx, :, :),
        view(param_struct.weights, time_idx, :, :),
        view(param_struct.log_weights, time_idx, :, :),
        view(state_struct.trajectories, :, time_idx, :)
    )

    # Weights
    @views @inbounds for n = 1:state_struct.nb_trajectories
        info_gain = ibis_info_gain_increment(
            adaptive_loop.dyn,
            param_struct.particles[:, time_idx, :, n],
            param_struct.log_weights[time_idx, :, n],
            state_struct.trajectories[1:adaptive_loop.dyn.xdim, time_idx, n],
            state_struct.trajectories[adaptive_loop.dyn.xdim+1:end, time_idx+1, n],
            state_struct.trajectories[1:adaptive_loop.dyn.xdim, time_idx+1, n],
            param_struct.scratch[:, :, n]
        )
        state_struct.cumulative_return[n] += info_gain
    end
end


function myopic_smc_step_with_rao_blackwell_marginal_dynamics!(
    time_idx::Int,
    adaptive_loop::RaoBlackwellAdaptiveLoop,
    state_struct::StateStruct,
    param_struct::RaoBlackwellParamStruct
)
    # Propagate
    state_struct.trajectories[:, time_idx+1, :] = rao_blackwell_marginal_adaptive_loop_sample(
        adaptive_loop,
        param_struct.distributions[time_idx, :],
        state_struct.trajectories[:, time_idx, :]
    )

    # Weights
    @views @inbounds for n = 1:state_struct.nb_trajectories
        q = param_struct.distributions[time_idx, n]
        x = state_struct.trajectories[1:adaptive_loop.dyn.xdim, time_idx, n]
        u = state_struct.trajectories[adaptive_loop.dyn.xdim+1:end, time_idx+1, n]
        xn = state_struct.trajectories[1:adaptive_loop.dyn.xdim, time_idx+1, n]

        info_gain = rao_blackwell_info_gain_increment(adaptive_loop.dyn, q, x, u, xn)
        state_struct.cumulative_return[n] += info_gain
    end
end