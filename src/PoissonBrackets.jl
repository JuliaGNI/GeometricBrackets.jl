module PoissonBrackets

using CompactBasisFunctions
using LinearAlgebra
using QuadratureRules
using Random
using SimpleSolvers
using SimpleSplines
using SparseArrays
# Loaded for its SimpleSolvers extension: `SparspakLU` is the linear solver the mixed
# formulation uses. See `Integrator`'s `formulation` keyword.
import Sparspak

# extended rather than defined here, so that the packages of the ecosystem share one
# generic function per accessor
import GeometricBase: integrate, value
import LinearAlgebra: issymmetric
import SimpleSplines: basis, basis_integrals, basis_values, degree, domainlength,
                      evaluate, l2_projection, mass_factorization, mass_matrix,
                      mass_operator, mass_solve!, mixed_matrix, nbasis,
                      ncells, nodes, order, quadrature_nodes, quadrature_weights,
                      pole, pole_triangle, pseudo_cartesian

# Re-exported rather than defined: the LAPACK-backed factorisation lives in SimpleSolvers
# as of 0.12.2, but it is this package's default `linear_solver_method`, so it has to be
# nameable without qualification by anyone who has only done `using PoissonBrackets`.
using SimpleSolvers: LapackLU
export LapackLU

export DiscreteSpace, SplineSpace, LagrangeSpace
export space
export basis, nbasis, degree, order, nodes, ncells, domainlength,
       mass_matrix, mass_factorization, inverse_mass_matrix,
       basis_values, basis_integrals,
       quadrature_nodes, quadrature_weights, project, project!, evaluate,
       derivative_matrix, stiffness_matrix, mixed_matrix, weighted_matrix,
       mass_operator, mass_solve!, nodal_derivative_matrix

include("spaces.jl")

export TensorSplineSpace, tensor_weighted_matrix, domainvolume

include("tensorspaces.jl")

export PolarSplineSpace, pole, pole_triangle, pseudo_cartesian

include("polarspaces.jl")

export PulledBack, measure, metric, jacobian_residual

include("pullback.jl")

export rref, exact_rank, kernel

include("exact.jl")

export DiscreteBracket, ConstantBracket, AffineBracket, GaugedBracket, MiuraBracket
export kernel_operator
export poisson_matrix, poisson_apply, poisson_tensor, isantisymmetric,
       jacobi_residual, structure_constant_residual

include("brackets.jl")

export MetricBracket, DoubleBracket, ProjectorBracket
export metric_matrix, metric_apply, metric_derivative, metric_operator,
       hamiltonian_field, project_orthogonal,
       issymmetric, ispositive_semidefinite, degeneracy_residual

include("metricbrackets.jl")

export CollisionBracket

include("collisionbrackets.jl")

export lie_poisson_matrix, lie_poisson_derivative,
       so3, se3, so_n, random_antisymmetric_c, sine_algebra, sine_coefficient,
       witt_truncation, poly_truncation, torus_truncation, graded_witt,
       closes_on, leak, galerkin_c

include("algebras.jl")

export TorusGrid, spectral_grid, finite_difference_grid, FD8_COEFFICIENTS
export ∂x, ∂y, sample, canonical_bracket

include("torus.jl")

export gardner_x, gardner_y, symmetric_x, symmetric_y
export gardner_2bracket, gardner_2bracket_density,
       gardner_4bracket, gardner_4bracket_density,
       symmetric_2bracket, symmetric_2bracket_density,
       symmetric_4bracket, symmetric_4bracket_density,
       weighted_2bracket, weighted_2bracket_density,
       weighted_4bracket, weighted_4bracket_density,
       lie_poisson_2bracket
export antisymmetry_residuals, plucker_residual

include("fourbrackets.jl")

export kulkarni_nomizu, metriplectic_bracket

include("metriplectic.jl")

export dirac_blocks, schur_complement, dirac_tensor, dirac_R, dirac_projector,
       dschur, reduced_bracket, jacobiator, project_jacobiator,
       restrict_c, is_antisymmetric_c, is_ideal, maximal_second_class

include("dirac.jl")

export shifted_legendre, lagrange_coefficients, dg_local_algebra, coarse_in_dg,
       assemble_dg_hierarchical

include("hierarchical.jl")

export DiscreteHamiltonian, hamiltonian, gradient, hessian
export MassCasimir, QuadraticHamiltonian

include("hamiltonians.jl")

export AbstractFlow, HamiltonianFlow, vectorfield, vectorfield!, jacobian

include("flows.jl")

export MetriplecticFlow, entropy, entropy_gradient, entropy_hessian

include("metriplecticflows.jl")

export IntegratorMethod, ExplicitEuler, RungeKutta4, ImplicitMidpoint,
       AverageVectorField, DiscreteGradient, Gonzalez, GonzalezMass,
       ProjectionMethod
export Integrator, integrate_step!, tangent_map, default_f_abstol

include("integrators.jl")
include("mixed.jl")

export KdVSystem, kdv_bracket_1, kdv_bracket_2, kdv_miura_bracket,
       KdVHamiltonian1, KdVHamiltonian2, KdVHamiltonian3,
       MiuraSystem, MiuraHamiltonian, miura_map, miura_invert, miura_bracket,
       miura_casimir, miura_derivative, miura_moment_matrix, hill_lambda0,
       miura_lambda,
       soliton, two_soliton, solitons, cosine,
       three_solitons, five_solitons, miura_initial_v

include("equations/kdv.jl")

export BurgersSystem, burgers_bracket, BurgersHamiltonian, burgers_casimir

include("equations/burgers.jl")

export CamassaHolmSystem, camassa_holm_bracket_1, camassa_holm_bracket_2,
       CamassaHolmHamiltonian1, CamassaHolmHamiltonian2, HelmholtzMap,
       velocity, momentum, peakon

include("equations/camassaholm.jl")

export invariants, invariant_names, integrate, Trajectory, deviation,
       drift, absolute_drift, growth, poisson_defect, entropy_production
export energyplot, stateplot, sweepplot, convergenceplot,
       INTEGRATOR_COLORS, FLOW_STYLES, ERROR_FLOOR

include("diagnostics.jl")

end
