using Random
using Distributions


function inner_pf_step!(
    param_struct::IBISParamStruct,
    dynamics::IBISDynamics,
    trajectory::AbstractMatrix{Float64},
    param_prior::MultivariateDistribution,
    time_idx::Int
)

    # 1. Reweight.
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
    # Variance is M^{-3/2}.
    nb_particles = param_struct.nb_particles
    sqrt_covar = nb_particles^(-3/4)
    prev_particles = @view param_struct.particles[:, time_idx + 1, :]
    if typeof(param_prior) <: MvNormal
        param_struct.particles[:, time_idx+1, :] = prev_particles + sqrt_covar * randn(size(prev_particles))
    else
        log_particles = log.(prev_particles) + sqrt_covar * randn(size(prev_particles))
        param_struct.particles[:, time_idx+1, :] = exp.(log_particles)
    end
end
