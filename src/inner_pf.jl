using Random


function inner_pf_step!(
    time_idx::Int,
    trajectory::AbstractMatrix{Float64},
    dynamics::IBISDynamics,
    param_prior::MultivariateDistribution,
    param_struct::IBISParamStruct,
)

    # 1. Reweight
    reweight_params!(
        time_idx,
        trajectory,
        dynamics,
        param_struct,
    )

    # 2. Resample.
    resample_params!(
        time_idx,
        param_struct
    )

    # 3. Jitter.
    nb_particles = param_struct.nb_particles
    sqrt_covar = (1 / nb_particles) * I
    prev_particles = @view param_struct.particles[:, time_idx + 1, :]
    param_struct.particles[:, time_idx+1, :] .= prev_particles + sqrt_covar * randn(size(prev_particles))
end