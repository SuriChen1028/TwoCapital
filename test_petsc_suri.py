#Required packages
import os
import sys
sys.path.append('./src')
from supportfunctions import *
sys.stdout.flush()
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import petsclinearsystem
from scipy.sparse import spdiags
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from datetime import datetime

reporterror = True
# Linear solver choices
# Chosse among petsc, petsc4py, eigen, both
# petsc: matrix assembled in C
# petsc4py: matrix assembled in Python
# eigen: matrix assembled in C++
# both: petsc+eigen
# both=eigen+petsc
#
linearsolver = 'petsc'

# Damage function choices
damageSpec = 'Weighted'  # Choose among "High"(Weitzman), 'Low'(Nordhaus) and 'Weighted' (arithmeticAverage of the two)

if damageSpec == 'High':
    weight = 0.0
elif damageSpec == 'Low':
    weight = 1.0
else:
    weight = 0.5

ξp =  1 / 4000  # Ambiguity Averse Paramter
# We stored solutions for ξp =  1 / 4000 to which we referred as "Ambiguity Averse" and ξp = 1000 as “Ambiguity Neutral” in the paper
# Sensible choices are from 0.0002 to 4000, while for parameters input over 0.01 the final results won't alter as much

if ξp < 1:
    aversespec = "Averse"
else:
    aversespec = 'Neutral'

smart_guess = False
model = 'petsc'
current_time = datetime.now()
filename = 'BBH_' + model + '_' + damageSpec + '_' + aversespec + '_' + "{:.3f}-{:.3f}-{:.3f}".format(current_time.day, current_time.hour, current_time.minute)

McD = np.loadtxt('./data/TCRE_MacDougallEtAl2017_update.txt')
par_lambda_McD = McD / 1000

β𝘧 = np.mean(par_lambda_McD)  # Climate sensitivity parameter, MacDougall (2017)
σᵦ = np.var(par_lambda_McD, ddof = 1)  # varaiance of climate sensitivity parameters
λ = 1.0 / σᵦ

quadrature = 'legendre'
n = 30
a = β𝘧 - 5 * np.sqrt(σᵦ)
b = β𝘧 + 5 * np.sqrt(σᵦ)

(xs,ws) = quad_points_legendre(n)
xs = (b-a) * 0.5  * xs + (a + b) * 0.5
s = np.prod((b-a) * 0.5)

start_time = time.time()

# Parameters as defined in the paper
δ = 0.01
κ = 0.032
σ𝘨 = 0.02
σ𝘬 = 0.0161
σ𝘳 = 0.0339
α = 0.115000000000000
ϕ0 = 0.0600
ϕ1 = 16.666666666666668
μk = -0.034977443912449
ψ0 = 0.112733407891680
ψ1 = 0.142857142857143

# parameters for damage function settings
power = 2
γ1 = 0.00017675
γ2 = 2. * 0.0022
γ2_plus = 2. * 0.0197
γ̄2_plus = weight * 0 + (1 - weight) * γ2_plus

σ1 = 0
σ2 = 0
ρ12 = 0
F̄ = 2
crit = 2
F0 = 1

xi_d = -1 * (1 - κ)

# See Remark 2.1.1 regarding the choice of ε and η
# False Trasient Time step
ε = 0.3
# Cobweb learning rate
η = 0.05


# Specifying Tolerance level
tol = 1e-8

# Grids Specification

# Coarse Grids
# R_min = 0
# R_max = 9
# F_min = 0
# F_max = 4000
# K_min = 0
# K_max = 18
# nR = 4
# nF = 4
# nK = 4
# R = np.linspace(R_min, R_max, nR)
# F = np.linspace(F_min, F_max, nF)
# K = np.linspace(K_min, K_max, nK)

# hR = R[1] - R[0]
# hF = F[1] - F[0]
# hK = K[1] - K[0]

# Dense Grids

R_min = 0
R_max = 9
F_min = 0
F_max = 4000
K_min = 0
K_max = 18

hR = 0.05
hF = 25.0 # make sure it is float instead of int
hK = 0.15

R = np.arange(R_min, R_max + hR, hR)
nR = len(R)
F = np.arange(F_min, F_max + hF, hF)
nF = len(F)
K = np.arange(K_min, K_max + hK, hK)
nK = len(K)

# Discretization of the state space for numerical PDE solution.
# See Remark 2.1.2
(R_mat, F_mat, K_mat) = np.meshgrid(R,F,K, indexing = 'ij')
stateSpace = np.hstack([R_mat.reshape(-1,1,order = 'F'),F_mat.reshape(-1,1,order = 'F'),K_mat.reshape(-1,1,order = 'F')])

# For PETSc
R_mat_1d = R_mat.ravel(order = 'F')
F_mat_1d = F_mat.ravel(order = 'F')
K_mat_1d = K_mat.ravel(order = 'F')
lowerLims = np.array([R_min, F_min, K_min],dtype=np.float64)
upperLims = np.array([R_max, F_max, K_max],dtype=np.float64)

# Inputs for function quad_int
# Integrating across parameter distribution
quadrature = 'legendre'
n = 30
a = β𝘧 - 5 * np.sqrt(σᵦ)
b = β𝘧 + 5 * np.sqrt(σᵦ)

v0 = κ * R_mat + (1-κ) * K_mat - β𝘧 * F_mat

FC_Err = 1
episode = 0

if smart_guess:
    v0 = v0_guess
    q = q_guess
    e_star = e_guess
    episode = 1
    ε = 0.2

max_iter = 20_000
file_iter = open("iter_c_compile.txt", "w")
while FC_Err > tol and episode < max_iter:
    print("-----------------------------------")
    print("---------Episode {}---------------".format(episode))
    print("-----------------------------------")
    start_ep = time.time()
    vold = v0.copy()
    # Applying finite difference scheme to the value function
    v0_dr = finiteDiff(v0,0,1,hR)
    v0_df = finiteDiff(v0,1,1,hF)
    v0_dk = finiteDiff(v0,2,1,hK)

    v0_drr = finiteDiff(v0,0,2,hR)
    v0_drr[v0_dr < 1e-16] = 0
    v0_dr[v0_dr < 1e-16] = 1e-16
    v0_dff = finiteDiff(v0,1,2,hF)
    v0_dkk = finiteDiff(v0,2,2,hK)
    if episode > 2000:
        ε = 0.1
    elif episode > 1000:
        ε = 0.2
    else:
        pass

    if episode == 0:
        # First time into the loop
        B1 = v0_dr - xi_d * (γ1 + γ2 * F_mat * β𝘧 + γ2_plus * (F_mat * β𝘧 - F̄) ** (power - 1) * (F_mat >= (crit / β𝘧))) * β𝘧 * np.exp(R_mat) - v0_df * np.exp(R_mat)
        C1 = - δ * κ
        e = -C1 / B1
        e_hat = e
        Acoeff = np.ones(R_mat.shape)
        Bcoeff = ((δ * (1 - κ) * ϕ1 + ϕ0 * ϕ1 * v0_dk) * δ * (1 - κ) / (v0_dr * ψ0 * 0.5) * np.exp(0.5 * (R_mat - K_mat))) / (δ * (1 - κ) * ϕ1)
        Ccoeff = -α  - 1 / ϕ1
        j = ((-Bcoeff + np.sqrt(Bcoeff ** 2 - 4 * Acoeff * Ccoeff)) / (2 * Acoeff)) ** 2
        i = α - j - (δ * (1 - κ)) / (v0_dr * ψ0 * 0.5) * j ** 0.5 * np.exp(0.5 * (R_mat - K_mat))
        q = δ * (1 - κ) / (α - i - j)
    else:
        e_hat = e_star

        # Step 4 (a) : Cobeweb scheme to update controls i and j; q is an intermediary variable that determines i and j
        Converged = 0
        nums = 0
        while Converged == 0:
            i_star = (ϕ0 * ϕ1 * v0_dk / q - 1) / ϕ1
            j_star = (q * np.exp(ψ1 * (R_mat - K_mat)) / (v0_dr * ψ0 * ψ1)) ** (1 / (ψ1 - 1))
            if α > np.max(i_star + j_star):
                q_star = η * δ * (1 - κ) / (α - i_star - j_star) + (1 - η) * q
            else:
                q_star = 2 * q
            if np.max(abs(q - q_star) / η) <= 1e-5:
                Converged = 1
                q = q_star
                i = i_star
                j = j_star
            else:
                q = q_star
                i = i_star
                j = j_star

            nums += 1
        if episode % 100 == 0:
            print('Cobweb Passed, iterations: {:.3f}, i error: {:10f}, j error: {:10f}'.format(nums, np.max(i - i_star), np.max(j - j_star)))

    i[i <= -1/ϕ1] = - 1/ϕ1 + 1e-8

    a1 = np.zeros(R_mat.shape)
    b1 = xi_d * e_hat * np.exp(R_mat) * γ1
    c1 = 2 * xi_d * e_hat * np.exp(R_mat) * F_mat * γ2
    λ̃1 = λ + c1 / ξp
    β̃1 = β𝘧 - c1 * β𝘧 / (ξp * λ̃1) -  b1 /  (ξp * λ̃1)
    I1 = a1 - 0.5 * np.log(λ) * ξp + 0.5 * np.log(λ̃1) * ξp + 0.5 * λ * β𝘧 ** 2 * ξp - 0.5 * λ̃1 * (β̃1) ** 2 * ξp
    R1 = 1 / ξp * (I1 - (a1 + b1 * β̃1 + c1 / 2 * β̃1 ** 2 + c1 / 2 / λ̃1))
    J1_without_e = xi_d * (γ1 * β̃1 + γ2 * F_mat * (β̃1 ** 2 + 1 / λ̃1)) * np.exp(R_mat)

    π̃1 = weight * np.exp(-1 / ξp * I1)

    # Step (2), solve minimization problem in HJB and calculate drift distortion
    # See remark 2.1.3 for more details
    start_time2 = time.time()
    if episode == 0 or (smart_guess and episode == 1):
        #@nb.jit(nopython = True, parallel = True)
        def scale_2_fnc(x, ndist, e_hat):
            return np.exp(-1 / ξp * x * e_hat) * ndist

        #@nb.jit(nopython = True, parallel = True)
        def q2_tilde_fnc(x, e_hat, scale_2):
            return np.exp(-1 / ξp * x * e_hat) / scale_2

        #@nb.jit(nopython = True, parallel = True)
        def J2_without_e_fnc(x, ndist, e_hat, scale_2):
            return x * q2_tilde_fnc(x, e_hat, scale_2) * ndist

        (xs,ws) = quad_points_legendre(n)
        xs = (b-a) * 0.5  * xs + (a + b) * 0.5
        s = np.prod((b-a) * 0.5)

        normdists = np.zeros(n)
        distort_terms = np.zeros((n, nR, nF, nK))
        for i_iter in range(n):
            normdists[i_iter] = normpdf(xs[i_iter],β𝘧,np.sqrt(σᵦ))
            distort_terms[i_iter] = xi_d * (γ1 * xs[i_iter] + γ2 * xs[i_iter] ** 2 * F_mat + γ2_plus * xs[i_iter] * (xs[i_iter] * F_mat - F̄) ** (power - 1) * ((xs[i_iter] * F_mat - F̄) >= 0)) * np.exp(R_mat)

        dVec = np.array([hR, hF, hK])
        increVec = np.array([1, nR, nR * nF],dtype=np.int32)

        # These are constant
        A = -δ * np.ones(R_mat.shape)
        C_rr = 0.5 * σ𝘳 ** 2 * np.ones(R_mat.shape)
        C_ff = np.zeros(R_mat.shape)
        C_kk = 0.5 * σ𝘬 ** 2 * np.ones(R_mat.shape)
        if linearsolver == 'petsc4py' or linearsolver == 'petsc' or linearsolver == 'both':
            petsc_mat = PETSc.Mat().create()
            petsc_mat.setType('aij')
            petsc_mat.setSizes([nR*nF*nK, nR*nF*nK])
            petsc_mat.setPreallocationNNZ(13)
            petsc_mat.setUp()
            ksp = PETSc.KSP()
            ksp.create(PETSc.COMM_WORLD)
            ksp.setType('bcgs')
            ksp.getPC().setType('ilu')
            ksp.setFromOptions()

            A_1d = A.ravel(order = 'F')
            C_rr_1d = C_rr.ravel(order = 'F')
            C_ff_1d = C_ff.ravel(order = 'F')
            C_kk_1d = C_kk.ravel(order = 'F')

            if linearsolver == 'petsc4py':
                I_LB_R = (stateSpace[:,0] == R_min)
                I_UB_R = (stateSpace[:,0] == R_max)
                I_LB_F = (stateSpace[:,1] == F_min)
                I_UB_F = (stateSpace[:,1] == F_max)
                I_LB_K = (stateSpace[:,2] == K_min)
                I_UB_K = (stateSpace[:,2] == K_max)
                diag_0_base = A_1d[:] + (I_LB_R * C_rr_1d[:] + I_UB_R * C_rr_1d[:] - 2 * (1 - I_LB_R - I_UB_R) * C_rr_1d[:]) / dVec[0] ** 2 + (I_LB_F * C_ff_1d[:] + I_UB_F * C_ff_1d[:] - 2 * (1 - I_LB_F - I_UB_F) * C_ff_1d[:]) / dVec[1] ** 2 + (I_LB_K * C_kk_1d[:] + I_UB_K * C_kk_1d[:] - 2 * (1 - I_LB_K - I_UB_K) * C_kk_1d[:]) / dVec[2] ** 2
                diag_R_base = - 2 * I_LB_R * C_rr_1d[:] / dVec[0] ** 2 + (1 - I_LB_R - I_UB_R) * C_rr_1d[:] / dVec[0] ** 2
                diag_Rm_base = - 2 * I_UB_R * C_rr_1d[:] / dVec[0] ** 2 + (1 - I_LB_R - I_UB_R) * C_rr_1d[:] / dVec[0] ** 2
                diag_F_base = - 2 * I_LB_F * C_ff_1d[:] / dVec[1] ** 2 + (1 - I_LB_F - I_UB_F) * C_ff_1d[:] / dVec[1] ** 2
                diag_Fm_base = - 2 * I_UB_F * C_ff_1d[:] / dVec[1] ** 2 + (1 - I_LB_F - I_UB_F) * C_ff_1d[:] / dVec[1] ** 2
                diag_K_base = - 2 * I_LB_K * C_kk_1d[:] / dVec[2] ** 2 + (1 - I_LB_K - I_UB_K) * C_kk_1d[:] / dVec[2] ** 2
                diag_Km_base = - 2 * I_UB_K * C_kk_1d[:] / dVec[2] ** 2 + (1 - I_LB_K - I_UB_K) * C_kk_1d[:] / dVec[2] ** 2
                diag_RR = I_LB_R * C_rr_1d[:] / dVec[0] ** 2
                diag_RRm = I_UB_R * C_rr_1d[:] / dVec[0] ** 2
                diag_FF = I_LB_F * C_ff_1d[:] / dVec[1] ** 2
                diag_FFm = I_UB_F * C_ff_1d[:] / dVec[1] ** 2
                diag_KK = I_LB_K * C_kk_1d[:] / dVec[2] ** 2
                diag_KKm = I_UB_K * C_kk_1d[:] / dVec[2] ** 2

    scale_2 = np.zeros(F_mat.shape)
    for i_iter in range(n):
        scale_2 += ws[i_iter] * scale_2_fnc(distort_terms[i_iter], normdists[i_iter], e_hat)
    scale_2 = s * scale_2

    I2 = -1 * ξp * np.log(scale_2)

    J2_without_e = np.zeros(F_mat.shape)
    for i_iter in range(n):
        J2_without_e += ws[i_iter] * J2_without_e_fnc(distort_terms[i_iter], normdists[i_iter], e_hat, scale_2)
    J2_without_e = s * J2_without_e
    J2_with_e = J2_without_e * e_hat
    end_time2 = time.time()

    R2 = (I2 - J2_with_e) / ξp
    π̃2 = (1 - weight) * np.exp(-1 / ξp * I2)
    π̃1_norm = π̃1 / (π̃1 + π̃2)
    π̃2_norm = 1 - π̃1_norm

    # step 4 (b) updating e based on first order conditions
    expec_e_sum = (π̃1_norm * J1_without_e + π̃2_norm * J2_without_e)

    B1 = v0_dr - v0_df * np.exp(R_mat) - expec_e_sum
    C1 = -δ * κ
    e = -C1 / B1
    e_star = e

    J1 = J1_without_e * e_star
    J2 = J2_without_e * e_star

    # Step (3) calculating implied entropies
    I_term = -1 * ξp * np.log(π̃1 + π̃2)

    R1 = (I1 - J1) / ξp
    R2 = (I2 - J2) / ξp

    # Step (5) solving for adjusted drift
    drift_distort = (π̃1_norm * J1 + π̃2_norm * J2)

    if weight == 0 or weight == 1:
        RE = π̃1_norm * R1 + π̃2_norm * R2
    else:
        RE = π̃1_norm * R1 + π̃2_norm * R2 + π̃1_norm * np.log(
            π̃1_norm / weight) + π̃2_norm * np.log(π̃2_norm / (1 - weight))

    RE_total = ξp * RE

    # Step (6) and (7) Formulating HJB False Transient parameters
    # See remark 2.1.4 for more details

    B_r = -e_star + ψ0 * (j ** ψ1) * np.exp(ψ1 * (K_mat - R_mat)) - 0.5 * (σ𝘳 ** 2)
    B_f = e_star * np.exp(R_mat)
    B_k = μk + ϕ0 * np.log(1 + i * ϕ1) - 0.5 * (σ𝘬 ** 2)

    D = δ * κ * np.log(e_star) + δ * κ * R_mat + δ * (1 - κ) * (np.log(α - i - j) + K_mat) + drift_distort + RE_total # + I_term

    if linearsolver == 'eigen' or linearsolver == 'both':
        start_eigen = time.time()
        out_eigen = PDESolver(stateSpace, A, B_r, B_f, B_k, C_rr, C_ff, C_kk, D, v0, ε, solverType = 'False Transient')
        out_comp = out_eigen[2].reshape(v0.shape,order = "F")
        print("Eigen solver: {:3f}s".format(time.time() - start_eigen))
        if episode % 1 == 0 and reporterror:
            v = np.array(out_eigen[2])
            res = np.linalg.norm(out_eigen[3].dot(v) - out_eigen[4])
            print("Eigen residual norm: {:g}; iterations: {}".format(res, out_eigen[0]))
            PDE_rhs = A * v0 + B_r * v0_dr + B_f * v0_df + B_k * v0_dk + C_rr * v0_drr + C_kk * v0_dkk + C_ff * v0_dff + D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)))
            print("Episode {:d} (Eigen): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(episode, PDE_Err, FC_Err))

    if linearsolver == 'petsc4py':
        bpoint1 = time.time()
        # ==== original impl ====
        # Transforming the 3-d coefficient matrix to 1-dimensional
        # A = A.reshape(-1,1,order = 'F')
        # B = np.hstack([B_r.reshape(-1,1,order = 'F'),B_f.reshape(-1,1,order = 'F'),B_k.reshape(-1,1,order = 'F')])
        # C = np.hstack([C_rr.reshape(-1,1,order = 'F'), C_ff.reshape(-1,1,order = 'F'), C_kk.reshape(-1,1,order = 'F')])
        # D = D.reshape(-1,1,order = 'F')
        # v0 = v0.reshape(-1,1,order = 'F')
        # B_r = B_r.ravel(order = 'F')
        # B_f = B_f.ravel(order = 'F')
        # B_k = B_k.ravel(order = 'F')
        # D = D.ravel(order = 'F')
        # v0 = v0.ravel(order = 'F')
        # B_plus = np.maximum(B, np.zeros(B.shape))
        # B_minus = np.minimum(B, np.zeros(B.shape))
        # diag_0 = (A[:,0] - 1 / ε
        #         + (I_LB_R * B[:,0] / -dVec[0] + I_UB_R * B[:,0] / dVec[0] - (1 - I_LB_R - I_UB_R) * (B_plus[:,0] - B_minus[:,0]) / dVec[0] + (I_LB_R * C[:,0] + I_UB_R * C[:,0] - 2 * (1 - I_LB_R - I_UB_R) * C[:,0]) / dVec[0] ** 2)
        #         + (I_LB_F * B[:,1] / -dVec[1] + I_UB_F * B[:,1] / dVec[1] - (1 - I_LB_F - I_UB_F) * (B_plus[:,1] - B_minus[:,1]) / dVec[1] + (I_LB_F * C[:,1] + I_UB_F * C[:,1] - 2 * (1 - I_LB_F - I_UB_F) * C[:,1]) / dVec[1] ** 2)
        #         + (I_LB_K * B[:,2] / -dVec[2] + I_UB_K * B[:,2] / dVec[2] - (1 - I_LB_K - I_UB_K) * (B_plus[:,2] - B_minus[:,2]) / dVec[2] + (I_LB_K * C[:,2] + I_UB_K * C[:,2] - 2 * (1 - I_LB_K - I_UB_K) * C[:,2]) / dVec[2] ** 2))
        # diag_R = (I_LB_R * B[:,0] / dVec[0] + (1 - I_LB_R - I_UB_R) * B_plus[:,0] / dVec[0] - 2 * I_LB_R * C[:,0] / dVec[0] ** 2 + (1 - I_LB_R - I_UB_R) * C[:,0] / dVec[0] ** 2)
        # diag_Rm = (I_UB_R * B[:,0] / -dVec[0] - (1 - I_LB_R - I_UB_R) * B_minus[:,0] / dVec[0] - 2 * I_UB_R * C[:,0] / dVec[0] ** 2 + (1 - I_LB_R - I_UB_R) * C[:,0] / dVec[0] ** 2)
        # diag_F = (I_LB_F * B[:,1] / dVec[1] + (1 - I_LB_F - I_UB_F) * B_plus[:,1] / dVec[1] - 2 * I_LB_F * C[:,1] / dVec[1] ** 2 + (1 - I_LB_F - I_UB_F) * C[:,1] / dVec[1] ** 2)
        # diag_Fm = (I_UB_F * B[:,1] / -dVec[1] - (1 - I_LB_F - I_UB_F) * B_minus[:,1] / dVec[1] - 2 * I_UB_F * C[:,1] / dVec[1] ** 2 + (1 - I_LB_F - I_UB_F) * C[:,1] / dVec[1] ** 2)
        # diag_K = (I_LB_K * B[:,2] / dVec[2] + (1 - I_LB_K - I_UB_K) * B_plus[:,2] / dVec[2] - 2 * I_LB_K * C[:,2] / dVec[2] ** 2 + (1 - I_LB_K - I_UB_K) * C[:,2] / dVec[2] ** 2)
        # diag_Km = (I_UB_K * B[:,2] / -dVec[2] - (1 - I_LB_K - I_UB_K) * B_minus[:,2] / dVec[2] - 2 * I_UB_K * C[:,2] / dVec[2] ** 2 + (1 - I_LB_K - I_UB_K) * C[:,2] / dVec[2] ** 2)
        # diag_RR = I_LB_R * C[:,0] / dVec[0] ** 2
        # diag_RRm = I_UB_R * C[:,0] / dVec[0] ** 2
        # diag_FF = I_LB_F * C[:,1] / dVec[1] ** 2
        # diag_FFm = I_UB_F * C[:,1] / dVec[1] ** 2
        # diag_KK = I_LB_K * C[:,2] / dVec[2] ** 2
        # diag_KKm = I_UB_K * C[:,2] / dVec[2] ** 2

        B_r_1d = B_r.ravel(order = 'F')
        B_f_1d = B_f.ravel(order = 'F')
        B_k_1d = B_k.ravel(order = 'F')
        D_1d = D.ravel(order = 'F')
        v0_1d = v0.ravel(order = 'F')
        # profiling
        # bpoint2 = time.time()
        # print("reshape: {:.3f}s".format(bpoint2 - bpoint1))
        diag_0 = diag_0_base - 1 / ε + I_LB_R * B_r_1d[:] / -dVec[0] + I_UB_R * B_r_1d[:] / dVec[0] - (1 - I_LB_R - I_UB_R) * np.abs(B_r_1d[:]) / dVec[0] + I_LB_F * B_f_1d[:] / -dVec[1] + I_UB_F * B_f_1d[:] / dVec[1] - (1 - I_LB_F - I_UB_F) * np.abs(B_f_1d[:]) / dVec[1] + I_LB_K * B_k_1d[:] / -dVec[2] + I_UB_K * B_k_1d[:] / dVec[2] - (1 - I_LB_K - I_UB_K) * np.abs(B_k_1d[:]) / dVec[2]
        diag_R = I_LB_R * B_r_1d[:] / dVec[0] + (1 - I_LB_R - I_UB_R) * B_r_1d.clip(min=0.0) / dVec[0] + diag_R_base
        diag_Rm = I_UB_R * B_r_1d[:] / -dVec[0] - (1 - I_LB_R - I_UB_R) * B_r_1d.clip(max=0.0) / dVec[0] + diag_Rm_base
        diag_F = I_LB_F * B_f_1d[:] / dVec[1] + (1 - I_LB_F - I_UB_F) * B_f_1d.clip(min=0.0) / dVec[1] + diag_F_base
        diag_Fm = I_UB_F * B_f_1d[:] / -dVec[1] - (1 - I_LB_F - I_UB_F) * B_f_1d.clip(max=0.0) / dVec[1] + diag_Fm_base
        diag_K = I_LB_K * B_k_1d[:] / dVec[2] + (1 - I_LB_K - I_UB_K) * B_k_1d.clip(min=0.0) / dVec[2] + diag_K_base
        diag_Km = I_UB_K * B_k_1d[:] / -dVec[2] - (1 - I_LB_K - I_UB_K) * B_k_1d.clip(max=0.0) / dVec[2] + diag_Km_base
        # profiling
        # bpoint3 = time.time()
        # print("prepare: {:.3f}s".format(bpoint3 - bpoint2))

        data = [diag_0, diag_R, diag_Rm, diag_RR, diag_RRm, diag_F, diag_Fm, diag_FF, diag_FFm, diag_K, diag_Km, diag_KK, diag_KKm]
        diags = np.array([0,-increVec[0],increVec[0],-2*increVec[0],2*increVec[0],
                        -increVec[1],increVec[1],-2*increVec[1],2*increVec[1],
                        -increVec[2],increVec[2],-2*increVec[2],2*increVec[2]])
        # The transpose of matrix A_sp is the desired. Create the csc matrix so that it can be used directly as the transpose of the corresponding csr matrix.
        A_sp = spdiags(data, diags, len(diag_0), len(diag_0), format='csc')
        b = -v0_1d/ε - D_1d
        # A_sp = spdiags(data, diags, len(diag_0), len(diag_0))
        # A_sp = csr_matrix(A_sp.T)
        # b = -v0/ε - D
        # profiling
        # bpoint4 = time.time()
        # print("create matrix and rhs: {:.3f}s".format(bpoint4 - bpoint3))
        petsc_mat = PETSc.Mat().createAIJ(size=A_sp.shape, csr=(A_sp.indptr, A_sp.indices, A_sp.data))
        petsc_rhs = PETSc.Vec().createWithArray(b)
        x = petsc_mat.createVecRight()
        # profiling
        # bpoint5 = time.time()
        # print("assemble: {:.3f}s".format(bpoint5 - bpoint4))

        # dump to files
        #x.set(0)
        #viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_A.dat', 'w')
        #petsc_mat.view(viewer)
        #viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_b.dat', 'w')
        #petsc_rhs.view(viewer)

        # create linear solver
        start_ksp = time.time()
        ksp.setOperators(petsc_mat)
        ksp.setTolerances(rtol=1e-12)
        ksp.solve(petsc_rhs, x)
        petsc_mat.destroy()
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(R_mat.shape,order = "F")
        end_ksp = time.time()
        # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
        print("petsc4py total: {:.3f}s".format(end_ksp - bpoint1))
        print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        if episode % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
            PDE_rhs = A * v0 + B_r * v0_dr + B_f * v0_df + B_k * v0_dk + C_rr * v0_drr + C_kk * v0_dkk + C_ff * v0_dff + D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)))
            print("Episode {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(episode, PDE_Err, FC_Err))
            # profling
            # bpoint7 = time.time()
            # print("compute error: {:.3f}s".format(bpoint7 - bpoint6))
        # if linearsolver == 'both':
            # compare
            # csr_mat = csr_mat*(-ε)
            # b = b*(-ε)
            # A_diff =  np.max(np.abs(out_eigen[3] - csr_mat))
            #
            # print("Coefficient matrix difference: {:.3f}".format(A_diff))
            # b_diff = np.max(np.abs(out_eigen[4] - np.squeeze(b)))
            # print("rhs difference: {:.3f}".format(b_diff))

    if linearsolver == 'petsc' or linearsolver == 'both':
        bpoint1 = time.time()
        B_r_1d = B_r.ravel(order = 'F')
        B_f_1d = B_f.ravel(order = 'F')
        B_k_1d = B_k.ravel(order = 'F')
        D_1d = D.ravel(order = 'F')
        v0_1d = v0.ravel(order = 'F')
        petsclinearsystem.formLinearSystem(R_mat_1d, F_mat_1d, K_mat_1d, A_1d, B_r_1d, B_f_1d, B_k_1d, C_rr_1d, C_ff_1d, C_kk_1d, ε, lowerLims, upperLims, dVec, increVec, petsc_mat)
        # profiling
        # bpoint2 = time.time()
        # print("form petsc mat: {:.3f}s".format(bpoint2 - bpoint1))
        b = v0_1d + D_1d*ε
        # petsc4py setting
        # petsc_mat.scale(-1./ε)
        # b = -v0_1d/ε - D_1d
        petsc_rhs = PETSc.Vec().createWithArray(b)
        x = petsc_mat.createVecRight()
        # profiling
        # bpoint3 = time.time()
        # print("form rhs and workvector: {:.3f}s".format(bpoint3 - bpoint2))

        # compare
        # ai, aj, av = petsc_mat.getValuesCSR()
        # A_sp = csr_matrix((av, aj, ai),shape=petsc_mat.size)
        # A_diff =  np.max(np.abs(out_eigen[3] - A_sp))
        # print("Coefficient matrix difference: {:.3f}".format(A_diff))
        # b_diff = np.max(np.abs(out_eigen[4] - np.squeeze(b)))
        # print("rhs difference: {:.3f}".format(b_diff))

        # dump to files
        # x.set(0)
        # viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_A.dat', 'w')
        #petsc_mat.view(viewer)
        #viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_b.dat', 'w')
        #petsc_rhs.view(viewer)

        # create linear solver
        start_ksp = time.time()
        ksp.setOperators(petsc_mat)
        ksp.setTolerances(rtol=1e-12)
        ksp.solve(petsc_rhs, x)
        # petsc_mat.destroy()
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(R_mat.shape,order = "F")
        end_ksp = time.time()
        # profiling
        # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
        num_iter = ksp.getIterationNumber()
        file_iter.write("%s \n" % num_iter)
        print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        if episode % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
            PDE_rhs = A * v0 + B_r * v0_dr + B_f * v0_df + B_k * v0_dk + C_rr * v0_drr + C_kk * v0_dkk + C_ff * v0_dff + D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)))
            print("Episode {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(episode, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_ep))
    # step 9: keep iterating until convergence
    v0 = out_comp
    episode += 1
if reporterror:
    print("===============================================")
    print("Fianal episode {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(episode -1, PDE_Err, FC_Err))
print("--- Total running time: %s seconds ---" % (time.time() - start_time))
exit()

import pickle
# filename = filename
my_shelf = {}
for key in dir():
    if isinstance(globals()[key], (int,float, float, str, bool, np.ndarray,list)):
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    else:
        pass


file = open(filename, 'wb')
pickle.dump(my_shelf, file)
file.close()
