# Optimization of post jump HJB
#Required packages
import os
import sys
sys.path.append('../src')
import csv
from supportfunctions import *
sys.stdout.flush()
# import petsc4py
# petsc4py.init(sys.argv)
# from petsc4py import PETSc
# import petsclinearsystem
from scipy.sparse import spdiags
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from datetime import datetime
from solver import solver_3d
from PostSolver import hjb_post_damage_post_tech
reporterror = True
# Linear solver choices
# Chosse among petsc, petsc4py, eigen, both
# petsc: matrix assembled in C
# petsc4py: matrix assembled in Python
# eigen: matrix assembled in C++
# both: petsc+petsc4py
#
linearsolver = 'eigen'

write_test = False
if write_test:
    f = open("test-log.txt", 'a')


current_time = datetime.now()
filename =  "res" + '-' + "{:d}-{:d}-{:d}".format(current_time.day, current_time.hour, current_time.minute)

if write_test:
    f.write("Script starts: {:d}/{:d}-{:d}:{:d}\n".format(current_time.month, current_time.day, current_time.hour, current_time.minute))
    f.write("Linear solver: " + linearsolver+ "\n" )


start_time = time.time()
# Parameters as defined in the paper
xi_a = 10000.
xi_p = 10000.
xi_b = 10000.
xi_g = 10000.
y_bar = 2.

# Model parameters
delta   = 0.010
alpha   = 0.115
kappa   = 6.667
mu_k    = -0.043
sigma_k = np.sqrt(0.0087**2 + 0.0038**2)
# Technology
theta        = 2 # 3
lambda_bar   = 0.1206
vartheta_bar = 0.0453

gamma_1 = 1.7675/10000
gamma_2 = .0022*2
gamma_3 = 0.0000


theta_ell = pd.read_csv('../data/model144.csv', header=None).to_numpy()[:, 0]/1000.
pi_c_o = np.ones_like(theta_ell)/len(theta_ell)
sigma_y = 1.2 * np.mean(theta_ell)
beta_f = 1.86 / 1000
zeta    = 0.02
psi_0   = 0.05
psi_1   = 1
sigma_g = 0.016
# Tech jump
lambda_bar_first = lambda_bar / 2
vartheta_bar_first = vartheta_bar / 2
lambda_bar_second = 1e-9
vartheta_bar_second = 0.

# Grids Specification
# Coarse Grids
Y_min = 0.
Y_max = 3.
# range of capital
K_min = 4.00
K_max = 8.50
################### arrival rate######################
lam_min = -10.
lam_max = -8.

# hR = 0.05
hK   = 0.10
hY   = 0.10 # make sure it is float instead of int
hlam = 0.20
# R = np.arange(R_min, R_max + hR, hR)
# nR = len(R)
Y = np.arange(Y_min, Y_max + hY, hY)
nY = len(Y)
K = np.arange(K_min, K_max + hK, hK)
nK = len(K)
logI = np.arange(lam_min, lam_max,  hlam)
nlogI = len(logI)


if write_test:
    f.write("Grid dimension: [{}, {}, {}]\n".format(nK, nY, nlogI))

print("Grid dimension: [{}, {}, {}]\n".format(nK, nY, nlogI))
# Discretization of the state space for numerical PDE solution.
######## post jump, 3 states
(K_mat, Y_mat, logI_mat) = np.meshgrid(K, Y, logI, indexing = 'ij')
stateSpace = np.hstack([K_mat.reshape(-1,1,order = 'F'), Y_mat.reshape(-1,1,order = 'F'), logI_mat.reshape(-1, 1, order='F')])

# For PETSc
K_mat_1d =K_mat.ravel(order='F')
Y_mat_1d = Y_mat.ravel(order='F')
logI_mat_1d = logI_mat.ravel(order='F')
lowerLims = np.array([K_min, Y_min, lam_min], dtype=np.float64)
upperLims = np.array([K_max, Y_max, lam_max], dtype=np.float64)


model_args = (delta, alpha, kappa, mu_k, sigma_k, theta_ell, pi_c_o, sigma_y, xi_a, xi_b, gamma_1, gamma_2, gamma_3, y_bar, theta, lambda_bar_first, vartheta_bar_first)


postjump = hjb_post_damage_post_tech(
        K, Y, model_args, v0=None, 
        epsilon=1., fraction=.5,tol=1e-8, max_iter=2000, print_iteration=True)

v_post = postjump["v"]
v0 = np.zeros(K_mat.shape)
for i in range(nlogI):
    v0[:,:,i] = v_post
V_post = v0
# import pickle
# data = pickle.load(open("data/res_13-1-37", "rb"))
# v0 = data["v0"]
############# step up of optimization
FC_Err = 1
epoch = 0
tol = 1e-7
epsilon = 0.001
fraction = 0.0001

# csvfile = open("ResForRatio.csv", "w")
# fieldnames = ["epoch", "iterations", "residual norm", "PDE_Err", "FC_Err"]
# writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
# writer.writeheader()
max_iter = 40000
# file_iter = open("iter_c_compile.txt", "w")

# res = solver_3d(K_mat, R_mat, Y_mat, # FOC_func, Coeff_func,  
        # args=(delta, eta, A_d, A_g, alpha_d, alpha_g, sigma_d, sigma_g, phi_d, phi_g, gamma_1, \
            # gamma_2, y_bar, varphi, varsigma, beta_f ),
        # linearsolver="petsc",
        # reporterror=True,
        # v0=v0, tol=1e-6, max_iter=10000, epsilon=0.1, fraction=0.5,
        # saveRes=True)

# exit()

while FC_Err > tol and epoch < max_iter:
    print("-----------------------------------")
    print("---------Epoch {}---------------".format(epoch))
    print("-----------------------------------")
    start_ep = time.time()
    vold = v0.copy()
    # Applying finite difference scheme to the value function
    ######## first order
    dK = finiteDiff(v0,0,1,hK)
    # dK[dK < 1e-8] = 1e-8
    dY = finiteDiff(v0,2,1,hY)
    dI = finiteDiff(v0,2,1,hlam)
    ######## second order
    ddK = finiteDiff(v0,0,2,hK)
    ddY = finiteDiff(v0,1,2,hY)
    ddI = finiteDiff(v0,2,2,hlam)


    dGamma = gamma_1 + gamma_2 * Y_mat + gamma_3 * (Y_mat - y_bar) * (Y_mat > y_bar)
    ddGamma = gamma_2 + gamma_3 * (Y_mat > y_bar)

    if epoch > 2000:
        epsilon = 0.1
    elif epoch > 1000:
        epsilon = 0.3
    else:
        pass

    # update control
    if epoch == 0:
        i = np.zeros(K_mat.shape)
        e = np.zeros(K_mat.shape)
        x = np.zeros(K_mat.shape)
        temp = alpha - i - alpha * vartheta_bar * (1 - e / (alpha * lambda_bar * np.exp(K_mat)))**theta - x * np.exp(logI_mat - K_mat)
        mc = 1 / temp

    else:
     # updating controls
        # Converged = 0
        # num = 0

        # while Converged == 0 and num < 5000:
            # i_g_1 = (1 - q / (dR * (1 - R_mat) + dK )) / phi_g
            # i_d_1 = (1 - q / (-dR * R_mat + dK)) / phi_d
            # i_d_1[i_d_1 >= A_d] = A_d - 1e-8
            # i_g_1[i_g_1 >= A_g] = A_g - 1e-8

            # if np.max(abs(i_g_1 - i_g)) <= 1e-8 and np.max(abs(i_d_1 - i_d)) <= 1e-8:
                # Converged = 1
                # i_g = i_g_1
                # i_d = i_d_1
            # else:
                # i_g = i_g_1
                # i_d = i_d_1
                # q = delta * (
                    # (A_g * R_mat - i_g * R_mat) + (A_d * (1-R_mat) - i_d * (1-R_mat))) ** (-1) * fraction + (1 - fraction) * q
            # num += 1
            # # print(num)
            # # print(np.max(abs(i_g_1 - i_g)) , np.max(abs(i_d_1 - i_d)))

        # # print(diff)

        mc = dI * psi_1 * psi_0 * np.exp(K_mat - logI_mat)
        temp2 = theta * vartheta_bar / lambda_bar * np.exp(- K_mat)
        F = dY - 1 / delta * dGamma
        G = ddY - 1 / delta * ddGamma
        Omega_1 = mc * temp2 + F * beta_f
        Omega_2 = mc * temp2 / (alpha * lambda_bar * np.exp(K_mat)) - F * sigma_y**2
        e_new =  Omega_1 / Omega_2
        e = e_new * fraction + e_star * (1 - fraction) 
        i_new = (1 - mc / dK) / kappa
        i = i_new * fraction + i_star * (1 - fraction) 
        temp3 = alpha  - i - alpha * vartheta_bar * (1 - e / (alpha * lambda_bar * np.exp(K_mat)))**theta
        x_new = temp3 * np.exp(K_mat - logI_mat) - 1 / (dI * psi_0 * psi_1)
        x = x_new * fraction + x_star * (1 - fraction)


    g = np.exp(1 / xi_g * (v0 - V_post))
    consumption = 1 / mc
    consumption[consumption < 1e-16] = 1e-16
    # Step (2), solve minimization problem in HJB and calculate drift distortion
    # See remark 2.1.3 for more details
    start_time2 = time.time()
    if epoch == 0:
        # dVec = np.array([hK, hR, hY])
        # increVec = np.array([1, nK, nK * nR],dtype=np.int32)
        # These are constant
        A = - delta  * np.ones(K_mat.shape) - np.exp(logI_mat) * g
        C_kk = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_yy = 0.5 * sigma_y**2 * e**2
        C_II = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        # if linearsolver == 'petsc4py' or linearsolver == 'petsc' or linearsolver == 'both':
            # petsc_mat = PETSc.Mat().create()
            # petsc_mat.setType('aij')
            # petsc_mat.setSizes([nK*nR*nY, nK*nR*nY])
            # petsc_mat.setPreallocationNNZ(13)
            # petsc_mat.setUp()
            # ksp = PETSc.KSP()
            # ksp.create(PETSc.COMM_WORLD)
            # ksp.setType('bcgs')
            # ksp.getPC().setType('ilu')
            # ksp.setFromOptions()

            # A_1d = A.ravel(order = 'F')
            # C_dd_1d = C_dd.ravel(order = 'F')
            # C_gg_1d = C_gg.ravel(order = 'F')
            # C_yy_1d = C_yy.ravel(order = 'F')

            # if linearsolver == 'petsc4py':
                # I_LB_d = (stateSpace[:,0] == K_min)
                # I_UB_d = (stateSpace[:,0] == K_max)
                # I_LB_g = (stateSpace[:,1] == R_min)
                # I_UB_g = (stateSpace[:,1] == R_max)
                # I_LB_y = (stateSpace[:,2] == Y_min)
                # I_UB_y = (stateSpace[:,2] == Y_max)
                # diag_0_base = A_1d[:]
                # diag_0_base += (I_LB_d * C_dd_1d[:] + I_UB_d * C_dd_1d[:] - 2 * (1 - I_LB_d - I_UB_d) * C_dd_1d[:]) / dVec[0] ** 2
                # diag_0_base += (I_LB_g * C_gg_1d[:] + I_UB_g * C_gg_1d[:] - 2 * (1 - I_LB_g - I_UB_g) * C_gg_1d[:]) / dVec[1] ** 2
                # diag_0_base += (I_LB_K * C_kk_1d[:] + I_UB_K * C_kk_1d[:] - 2 * (1 - I_LB_K - I_UB_K) * C_kk_1d[:]) / dVec[2] ** 2
                # diag_d_base = - 2 * I_LB_d * C_dd_1d[:] / dVec[0] ** 2 + (1 - I_LB_d - I_UB_d) * C_dd_1d[:] / dVec[0] ** 2
                # diag_dm_base = - 2 * I_UB_d * C_dd_1d[:] / dVec[0] ** 2 + (1 - I_LB_d - I_UB_d) * C_dd_1d[:] / dVec[0] ** 2
                # diag_g_base = - 2 * I_LB_g * C_gg_1d[:] / dVec[1] ** 2 + (1 - I_LB_g - I_UB_g) * C_gg_1d[:] / dVec[1] ** 2
                # diag_gm_base = - 2 * I_UB_g * C_gg_1d[:] / dVec[1] ** 2 + (1 - I_LB_g - I_UB_g) * C_gg_1d[:] / dVec[1] ** 2
                # diag_y_base = - 2 * I_LB_y * C_yy_1d[:] / dVec[2] ** 2 + (1 - I_LB_y - I_UB_y) * C_yy_1d[:] / dVec[2] ** 2
                # diag_ym_base = - 2 * I_UB_y * C_yy_1d[:] / dVec[2] ** 2 + (1 - I_LB_y - I_UB_y) * C_yy_1d[:] / dVec[2] ** 2
                # diag_dd = I_LB_d * C_dd_1d[:] / dVec[0] ** 2
                # diag_ddm = I_UB_d * C_dd_1d[:] / dVec[0] ** 2
                # diag_gg = I_LB_g * C_gg_1d[:] / dVec[1] ** 2
                # diag_ggm = I_UB_g * C_gg_1d[:] / dVec[1] ** 2
                # diag_yy = I_LB_y * C_yy_1d[:] / dVec[2] ** 2
                # diag_yym = I_UB_y * C_yy_1d[:] / dVec[2] ** 2


    # Step (6) and (7) Formulating HJB False Transient parameters
    # See remark 2.1.4 for more details
    B_k = mu_k + i - 0.5 * kappa * i**2 - 0.5 * sigma_k**2
    B_y = beta_f * e
    B_I = - zeta + psi_0 * x**psi_1 - 0.5 * sigma_g**2

    D = np.log(consumption) + K_mat  - 1. / delta * dGamma * beta_f * e  - 0.5 / delta * ddGamma * sigma_y**2 * e**2  + xi_g * np.exp(logI_mat) * (1 - g + g * np.log(g)) + np.exp(logI_mat) * g * V_post

    if linearsolver == 'eigen' or linearsolver == 'both':
        start_eigen = time.time()
        out_eigen = PDESolver(stateSpace, A, B_k, B_y, B_I, C_kk, C_yy, C_II, D, v0, epsilon, solverType = 'False Transient')
        out_comp = out_eigen[2].reshape(v0.shape,order = "F")
        print("Eigen solver: {:3f}s".format(time.time() - start_eigen))
        if epoch % 1 == 0 and reporterror:
            v = np.array(out_eigen[2])
            res = np.linalg.norm(out_eigen[3].dot(v) - out_eigen[4])
            print("Eigen residual norm: {:g}; iterations: {}".format(res, out_eigen[0]))
            PDE_rhs = A * v0 + B_k * dK + B_y * dY + B_I * dI + C_kk * ddK + C_yy * ddY + C_II * ddI + D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)))
            print("Episode {:d} (Eigen): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))

    # if linearsolver == 'petsc4py':
        # bpoint1 = time.time()
        # # ==== original impl ====
        # B_d_1d = B_d.ravel(order = 'F')
        # B_g_1d = B_g.ravel(order = 'F')
        # B_y_1d = B_y.ravel(order = 'F')
        # D_1d = D.ravel(order = 'F')
        # v0_1d = v0.ravel(order = 'F')
        # # profiling
        # # bpoint2 = time.time()
        # # print("reshape: {:.3f}s".format(bpoint2 - bpoint1))
        # diag_0 = diag_0_base - 1 / epsilon + I_LB_R * B_r_1d[:] / -dVec[0] + I_UB_R * B_r_1d[:] / dVec[0] - (1 - I_LB_R - I_UB_R) * np.abs(B_r_1d[:]) / dVec[0] + I_LB_F * B_f_1d[:] / -dVec[1] + I_UB_F * B_f_1d[:] / dVec[1] - (1 - I_LB_F - I_UB_F) * np.abs(B_f_1d[:]) / dVec[1] + I_LB_K * B_k_1d[:] / -dVec[2] + I_UB_K * B_k_1d[:] / dVec[2] - (1 - I_LB_K - I_UB_K) * np.abs(B_k_1d[:]) / dVec[2]
        # diag_R = I_LB_R * B_r_1d[:] / dVec[0] + (1 - I_LB_R - I_UB_R) * B_r_1d.clip(min=0.0) / dVec[0] + diag_R_base
        # diag_Rm = I_UB_R * B_r_1d[:] / -dVec[0] - (1 - I_LB_R - I_UB_R) * B_r_1d.clip(max=0.0) / dVec[0] + diag_Rm_base
        # diag_F = I_LB_F * B_f_1d[:] / dVec[1] + (1 - I_LB_F - I_UB_F) * B_f_1d.clip(min=0.0) / dVec[1] + diag_F_base
        # diag_Fm = I_UB_F * B_f_1d[:] / -dVec[1] - (1 - I_LB_F - I_UB_F) * B_f_1d.clip(max=0.0) / dVec[1] + diag_Fm_base
        # diag_K = I_LB_K * B_k_1d[:] / dVec[2] + (1 - I_LB_K - I_UB_K) * B_k_1d.clip(min=0.0) / dVec[2] + diag_K_base
        # diag_Km = I_UB_K * B_k_1d[:] / -dVec[2] - (1 - I_LB_K - I_UB_K) * B_k_1d.clip(max=0.0) / dVec[2] + diag_Km_base
        # # profiling
        # # bpoint3 = time.time()
        # # print("prepare: {:.3f}s".format(bpoint3 - bpoint2))

        # data = [diag_0, diag_R, diag_Rm, diag_RR, diag_RRm, diag_F, diag_Fm, diag_FF, diag_FFm, diag_K, diag_Km, diag_KK, diag_KKm]
        # diags = np.array([0,-increVec[0],increVec[0],-2*increVec[0],2*increVec[0],
                        # -increVec[1],increVec[1],-2*increVec[1],2*increVec[1],
                        # -increVec[2],increVec[2],-2*increVec[2],2*increVec[2]])
        # # The transpose of matrix A_sp is the desired. Create the csc matrix so that it can be used directly as the transpose of the corresponding csr matrix.
        # A_sp = spdiags(data, diags, len(diag_0), len(diag_0), format='csc')
        # b = -v0_1d/epsilon - D_1d
        # # A_sp = spdiags(data, diags, len(diag_0), len(diag_0))
        # # A_sp = csr_matrix(A_sp.T)
        # # b = -v0/ε - D
        # # profiling
        # # bpoint4 = time.time()
        # # print("create matrix and rhs: {:.3f}s".format(bpoint4 - bpoint3))
        # petsc_mat = PETSc.Mat().createAIJ(size=A_sp.shape, csr=(A_sp.indptr, A_sp.indices, A_sp.data))
        # petsc_rhs = PETSc.Vec().createWithArray(b)
        # x = petsc_mat.createVecRight()
        # # profiling
        # # bpoint5 = time.time()
        # # print("assemble: {:.3f}s".format(bpoint5 - bpoint4))

        # # dump to files
        # #x.set(0)
        # #viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_A.dat', 'w')
        # #petsc_mat.view(viewer)
        # #viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_b.dat', 'w')
        # #petsc_rhs.view(viewer)

        # # create linear solver
        # start_ksp = time.time()
        # ksp.setOperators(petsc_mat)
        # ksp.setTolerances(rtol=1e-14)
        # ksp.solve(petsc_rhs, x)
        # petsc_mat.destroy()
        # petsc_rhs.destroy()
        # x.destroy()
        # out_comp = np.array(ksp.getSolution()).reshape(R_mat.shape,order = "F")
        # end_ksp = time.time()
        # # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
        # print("petsc4py total: {:.3f}s".format(end_ksp - bpoint1))
        # print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        # if epoch % 1 == 0 and reporterror:
            # # Calculating PDE error and False Transient error
            # PDE_rhs = A * v0 + B_d * dK + B_g * dR + B_y * dY + C_dd * ddK + C_gg * ddR + C_yy * ddY + D
            # PDE_Err = np.max(abs(PDE_rhs))
            # FC_Err = np.max(abs((out_comp - v0) / epsilon))
            # print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
            # # profling
            # # bpoint7 = time.time()
            # # print("compute error: {:.3f}s".format(bpoint7 - bpoint6))
        # # if linearsolver == 'both':
            # # compare
            # # csr_mat = csr_mat*(-ε)
            # # b = b*(-ε)
            # # A_diff =  np.max(np.abs(out_eigen[3] - csr_mat))
            # #
            # # print("Coefficient matrix difference: {:.3f}".format(A_diff))
            # # b_diff = np.max(np.abs(out_eigen[4] - np.squeeze(b)))
            # # print("rhs difference: {:.3f}".format(b_diff))

    # if linearsolver == 'petsc' or linearsolver == 'both':
        # bpoint1 = time.time()
        # B_d_1d = B_d.ravel(order = 'F')
        # B_g_1d = B_g.ravel(order = 'F')
        # B_y_1d = B_y.ravel(order = 'F')
        # D_1d = D.ravel(order = 'F')
        # v0_1d = v0.ravel(order = 'F')
        # petsclinearsystem.formLinearSystem(K_mat_1d, R_mat_1d, Y_mat_1d, A_1d, B_d_1d, B_g_1d, B_y_1d, C_dd_1d, C_gg_1d, C_yy_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
        # # profiling
        # # bpoint2 = time.time()
        # # print("form petsc mat: {:.3f}s".format(bpoint2 - bpoint1))
        # b = v0_1d + D_1d*epsilon
        # # petsc4py setting
        # # petsc_mat.scale(-1./ε)
        # # b = -v0_1d/ε - D_1d
        # petsc_rhs = PETSc.Vec().createWithArray(b)
        # x = petsc_mat.createVecRight()
        # # profiling
        # # bpoint3 = time.time()
        # # print("form rhs and workvector: {:.3f}s".format(bpoint3 - bpoint2))


        # # create linear solver
        # start_ksp = time.time()
        # ksp.setOperators(petsc_mat)
        # ksp.setTolerances(rtol=1e-12)
        # ksp.solve(petsc_rhs, x)
        # # petsc_mat.destroy()
        # petsc_rhs.destroy()
        # x.destroy()
        # out_comp = np.array(ksp.getSolution()).reshape(K_mat.shape,order = "F")
        # end_ksp = time.time()
        # # profiling
        # # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
        # num_iter = ksp.getIterationNumber()
        # # file_iter.write("%s \n" % num_iter)
        # print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        # print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        # if epoch % 1 == 0 and reporterror:
            # # Calculating PDE error and False Transient error
            # PDE_rhs = A * v0 + B_d * dK + B_g * dR + B_y * dY + C_dd * ddK + C_gg * ddR + C_yy * ddY + D
            # PDE_Err = np.max(abs(PDE_rhs))
            # FC_Err = np.max(abs((out_comp - v0)/ epsilon))
            # print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
    # print("Epoch time: {:.4f}".format(time.time() - start_ep))
    # # step 9: keep iterating until convergence
    # # rowcontent = {
        # # "epoch": epoch,
        # # "iterations": num_iter,
        # # "residual norm": ksp.getResidualNorm(),
        # # "PDE_Err": PDE_Err,
        # # "FC_Err": FC_Err
    # # }
    # # writer.writerow(rowcontent)
    e_star = e
    i_star = i
    x_star = x
    v0 = out_comp
    epoch += 1
if reporterror:
    print("===============================================")
    print("Fianal epoch {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch -1, PDE_Err, FC_Err))
print("--- Total running time: %s seconds ---" % (time.time() - start_time))

if write_test:
    f.write("Fianal epoch {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}\n" .format(epoch -1, PDE_Err, FC_Err))
    f.write("--- Total running time: %s seconds ---\n" % (time.time() - start_time))

# exit()

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


file = open("./res_data/pre_jump", 'wb')
pickle.dump(my_shelf, file)
file.close()
