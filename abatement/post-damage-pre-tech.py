# Optimization of post jump HJB
#Required packages
import os
import sys
sys.path.append('../src')
import csv
from supportfunctions import *
from supportfunctions import finiteDiff_3D
sys.stdout.flush()
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import petsclinearsystem
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
linearsolver = 'petsc'
now = datetime.now()
current_time = now.strftime("%d-%H:%M")



start_time = time.time()
# Parameters as defined in the paper
xi_a = 10000.
xi_p = 10000.
xi_b = 10000.
xi_g = 100.
y_bar = 2.

# Model parameters
delta   = 0.010
alpha   = 0.115
kappa   = 6.667
mu_k    = -0.043
sigma_k = np.sqrt(0.0087**2 + 0.0038**2)
# Technology
theta        = 3
lambda_bar   = 0.1206
vartheta_bar = 0.0453

gamma_1 = 1.7675/10000
gamma_2 = 0.0022 * 2
gamma_3 = 0.3853 * 2


theta_ell = pd.read_csv('../data/model144.csv', header=None).to_numpy()[:, 0]/1000.
pi_c_o    = np.ones_like(theta_ell)/len(theta_ell)
sigma_y   = 1.2 * np.mean(theta_ell)
beta_f    = 1.86 / 1000
zeta      = 0.00
psi_0     = 0.01
psi_1     = 1/2
sigma_g   = 0.016
# Tech jump
lambda_bar_first = lambda_bar / 2
vartheta_bar_first = vartheta_bar / 2
lambda_bar_second = 1e-9
vartheta_bar_second = 0.

# Grids Specification
# Coarse Grids
K_min = 4.00
K_max = 9.00
hK    = 0.20
K     = np.arange(K_min, K_max + hK, hK)
nK    = len(K)
Y_min = 0.
Y_max = 5.
hY    = 0.20 # make sure it is float instead of int
Y     = np.arange(Y_min, Y_max + hY, hY)
nY    = len(Y)
L_min = - 5.
L_max = - 0.
hL    = 0.2
L     = np.arange(L_min, L_max,  hL)
nL    = len(L)

X1     = K
nX1    = len(X1)
hX1    = X1[1] - X1[0]
X1_min = X1.min()
X1_max = X1.max()
X2     = Y
nX2    = len(X2)
hX2    = X2[1] - X2[0]
X2_min = X2.min()
X2_max = X2.max()
X3     = L
nX3    = len(X3)
hX3    = X3[1] - X3[0]
X3_min = X3.min()
X3_max = X3.max()

filename =  "post_damage_" + str(gamma_3)  + '_{}'.format(current_time)
print("Grid dimension: [{}, {}, {}]\n".format(nX1, nX2, nX3))
print("Grid step: [{}, {}, {}]\n".format(hX1, hX2, hX3))
# Discretization of the state space for numerical PDE solution.
######## post jump, 3 states
(X1_mat, X2_mat, X3_mat) = np.meshgrid(X1, X2, X3, indexing = 'ij')
stateSpace = np.hstack([X1_mat.reshape(-1,1,order = 'F'), X2_mat.reshape(-1,1,order = 'F'), X3_mat.reshape(-1, 1, order='F')])
K_mat = X1_mat
Y_mat = X2_mat
L_mat = X3_mat
# For PETSc
X1_mat_1d = X1_mat.ravel(order='F')
X2_mat_1d = X2_mat.ravel(order='F')
X3_mat_1d = X3_mat.ravel(order='F')
lowerLims = np.array([X1_min, X2_min, X3_min], dtype=np.float64)
upperLims = np.array([X1_max, X2_max, X3_max], dtype=np.float64)


model_args = (delta, alpha, kappa, mu_k, sigma_k, theta_ell, pi_c_o, sigma_y, xi_a, xi_b, gamma_1, gamma_2, gamma_3, y_bar, theta, lambda_bar_second, vartheta_bar_second)


# postjump = hjb_post_damage_post_tech(
         # K, Y, model_args, v0=None,
        # epsilon=1., fraction=.2,tol=1e-8, max_iter=2000, print_iteration=True)

# with open("./res_data/gamma/post_jump_" + filename, "wb") as f:
    # pickle.dump(postjump, f)

# v_post = postjump["v"]
# v_post = pickle.load(open("./res_data/gamma/techIII_carbon_neutral", "rb"))["v"]
# res_post = pickle.load(open("./res_data/gamma/post_jump_post_damage_0.0_08-13:24", "rb"))
res_post = pickle.load(open("./res_data/gamma/pre_jump_post_damage_0.7706_IItoIII", "rb"))
V_post = res_post["v0"] # for I -> II
v0 = V_post
# v0 = np.zeros(K_mat.shape)
# for i in range(nL):
    # v0[:,:,i] = v_post
# V_post = v0
Guess = None
pi_c = np.array([temp * np.ones_like(K_mat) for temp in pi_c_o])
pi_c_o = pi_c.copy()
theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell])
# import pickle
# v0 = pickle.load(open("./res_data/res-1-12-7", "rb"))["v0"]
# Guess = pickle.load(open("./res_data/res-1-15-47", "rb"))
# v0 = Guess["v0"]
############# step up of optimization
# Guess = pickle.load(open("./res_data/gamma/pre_jump_post_damage_0.0_08-16:09", "rb"))
# v0 = Guess["v0"]

# vartheta_bar = vartheta_bar_first
# lambda_bar = lambda_bar_first
# v0 = (K_mat + L_mat)  -  beta_f * Y_mat
FC_Err   = 1
epoch    = 0
tol      = 1e-7
epsilon  = 0.1
fraction = 0.5

i_star = np.zeros(K_mat.shape)
# i_star = Guess["i_star"]
i_star = res_post["i_star"]
e_star = np.zeros(K_mat.shape)
# e_star = Guess["e_star"]
e_star = res_post["e_star"]
x_star = np.zeros(K_mat.shape)
# x_star = Guess["x_star"]
x_star = res_post["x_star"]
# csvfile = open("ResForRatio.csv", "w")
# fieldnames = ["epoch", "iterations", "residual norm", "PDE_Err", "FC_Err"]
# writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
# writer.writeheader()
max_iter = 4000
# file_iter = open("iter_c_compile.txt", "w")

# res = solver_3d(K_mat, R_mat, Y_mat, # FOC_func, Coeff_func,
        # args=(delta, eta, A_d, A_g, alpha_d, alpha_g, sigma_d, sigma_g, phi_d, phi_g, gamma_1, \
            # gamma_2, y_bar, varphi, varsigma, beta_f ),
        # linearsolver="petsc",
        # reporterror=True,
        # v0=v0, tol=1e-6, max_iter=10000, epsilon=0.1, fraction=0.5,
        # saveRes=True)

# exit()

dVec = np.array([hX1, hX2, hX3])
increVec = np.array([1, nX1, nX1 * nX2],dtype=np.int32)
dG  = gamma_1 + gamma_2 * Y_mat + gamma_3 * (Y_mat - y_bar) * (Y_mat > y_bar)
ddG = gamma_2 + gamma_3 * (Y_mat > y_bar)
while FC_Err > tol and epoch < max_iter:
    print("-----------------------------------")
    print("---------Epoch {}---------------".format(epoch))
    print("-----------------------------------")
    start_ep = time.time()
    vold = v0.copy()
    # Applying finite difference scheme to the value function
    ######## first order
    dX1  = finiteDiff_3D(v0,0,1,hX1)
    dX1[dX1 <= 1e-16] = 1e-16
    dK = dX1
    dX2  = finiteDiff_3D(v0,1,1,hX2)
    dY = dX2
    dX3  = finiteDiff_3D(v0,2,1,hX3)
    dX3[dX3 <= 1e-16] = 1e-16
    dL = dX3
    ######## second order
    ddX1 = finiteDiff_3D(v0,0,2,hX1)
    ddX2 = finiteDiff_3D(v0,1,2,hX2)
    ddY = ddX2
    ddX3 = finiteDiff_3D(v0,2,2,hX3)



    # if epoch > 2000:
        # epsilon = 0.1
    # elif epoch > 1000:
        # epsilon = 0.3
    # else:
        # pass

    # update control
    # if epoch == 0:
        # ii = 0.01 * np.ones(K_mat.shape)
        # ee = np.ones(K_mat.shape)
        # xx = 0.1 * np.ones(K_mat.shape)
        # temp = alpha - ii - alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta - xx
        # mc = 1 / temp
        # if Guess:
            # ii = Guess["ii"]
            # ee = Guess["ee"]
            # xx = Guess["xx"]
    if epoch == 0:
        fraction = 1
    else:
        fraction = 0.01

    # else:
     # updating controls
    if theta == 2 and psi_1 == 1:
        mc = dL * psi_1 * psi_0 * np.exp(K_mat - L_mat)
        temp2 = theta * vartheta_bar / lambda_bar * np.exp(- K_mat)
        F = dY  - dGamma
        G = ddY - ddGamma
        Omega_1 = mc * temp2 + F * beta_f
        Omega_2 = mc * temp2 / (alpha * lambda_bar * np.exp(K_mat)) - F * sigma_y**2
        e_new =  Omega_1 / Omega_2
        # e_new[e_new <= 1e-15] = 1e-15
        i_new = (1 - mc / dK) / kappa
        # i_new[i_new <= 1e-15] = 1e-15
        temp3 = alpha  - ii - alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
        x_new = temp3 * np.exp(K_mat - L_mat) - 1 / (dL * psi_0 * psi_1)
        # x_new[x_new <= 1e-15] = 1e-15
    elif theta == 3 and psi_1 == 1:

        G = dY  - dGamma
        F = ddY - ddGamma
        mc = dL * psi_1 * psi_0 * np.exp(K_mat - L_mat)
        mc[mc <= 1e-16] = 1e-16
        temp = mc * vartheta_bar * theta / (lambda_bar * np.exp(K_mat))
        a = temp / (alpha * lambda_bar * np.exp(K_mat)) ** (theta - 1)
        b = - 2 * temp / (alpha * lambda_bar * np.exp(K_mat)) + F * sigma_y ** 2
        c = temp + G * beta_f
        temp = b ** 2 - 4 * a * c
        temp[temp <=0] = 0
        # temp = temp * (temp > 0)
        root1 = (- b - np.sqrt(temp)) / (2 * a)
        root2 = (- b + np.sqrt(temp)) / (2 * a)
        if root1.all() > 0 :
            e_new = root1
        else:
            e_new = root2


        e_new[e_new < - 1000] = - 1000
        e_new[e_new > 1e3] = 1e3
        i_new = (1 - mc/ dK) / kappa
        i_new[i_new < - 1000] = - 1000
        temp3 = alpha - i_star - alpha * vartheta_bar * (1 - e_star / (alpha * lambda_bar * np.exp(K_mat)))**theta
        x_new = temp3 - 1 / mc
        x_new[x_new < -1000] = - 1000
        x_new[x_new > 1 - 1e-16] = 1 - 1e-16

    elif psi_1 != 1 and vartheta_bar != 0 and theta == 3:
        G = dY -  dG
        F = ddY - ddG
        j_star = alpha * vartheta_bar * (1 - e_star / (alpha * lambda_bar * np.exp(K_mat)))**theta
        j_star[j_star <= 1e-16] = 1e-16
        consumption = alpha - i_star - j_star - x_star
        consumption[consumption <= 1e-16] = 1e-16
        mc  = delta / consumption
        temp = mc * vartheta_bar * theta / (lambda_bar * np.exp(K_mat))
        a = temp / (alpha * lambda_bar * np.exp(K_mat))**(theta - 1)
        b = - 2 * temp / (alpha * lambda_bar * np.exp(K_mat)) +  F * sigma_y**2
        c = temp + G * np.sum(theta_ell * pi_c, axis=0)
        temp = b ** 2 - 4 * a * c
        temp = temp * (temp > 0)
        root1 = (- b - np.sqrt(temp)) / (2 * a)
        root2 = (- b + np.sqrt(temp)) / (2 * a)
        if root1.all() > 0 :
            e_new = root1
        else:
            e_new = root2
        e_new[e_new <= 1e-16] = 1e-16
        i_new = - (mc / dK - 1) / kappa
        i_new[i_new <= 1e-16] = 1e-16
        x_new = (mc / (dL * psi_0 * psi_1) * np.exp(psi_1 * (L_mat - K_mat)) )**(1 / (psi_1 - 1))

    ii = i_new * fraction + i_star * (1 - fraction)
    ee = e_new * fraction + e_star * (1 - fraction)
    xx = x_new * fraction + x_star * (1 - fraction)
    print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
    print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
    print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
    # ii = np.zeros(K_mat.shape)
    # ee = np.zeros(K_mat.shape)
    # xx = np.zeros(K_mat.shape)
    log_pi_c_ratio = - G * ee * theta_ell / xi_a
    pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
    pi_c = np.exp(pi_c_ratio) * pi_c_o
    pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
    pi_c = pi_c / np.sum(pi_c, axis=0)
    entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)
    gg = np.exp(1 / xi_g * (v0 - V_post))
    gg[gg <=1e-16] = 1e-16
    gg[gg >= 1] = 1
    jj =  alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
    jj[jj <= 1e-16] = 1e-16
    consumption = alpha - ii - jj - xx
    consumption[consumption <= 1e-16] = 1e-16
    # Step (2), solve minimization problem in HJB and calculate drift distortion
    # See remark 2.1.3 for more details
    A   = - delta * np.ones(K_mat.shape) - np.exp(L_mat) * gg
    C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
    C_2 = 0.5 * sigma_y**2 * ee**2
    C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
    start_time2 = time.time()
    if epoch == 0:
        # These are constant
        if linearsolver == 'petsc4py' or linearsolver == 'petsc' or linearsolver == 'both':
            petsc_mat = PETSc.Mat().create()
            petsc_mat.setType('aij')
            petsc_mat.setSizes([nX1 * nX2 * nX3, nX1 * nX2 * nX3])
            petsc_mat.setPreallocationNNZ(13)
            petsc_mat.setUp()
            ksp = PETSc.KSP()
            ksp.create(PETSc.COMM_WORLD)
            ksp.setType('bcgs')
            ksp.getPC().setType('ilu')
            ksp.setFromOptions()

            A_1d   = A.ravel(order = 'F')
            C_1_1d = C_1.ravel(order = 'F')
            C_2_1d = C_2.ravel(order = 'F')
            C_3_1d = C_3.ravel(order = 'F')

            if linearsolver == 'petsc4py':
                I_LB_1 = (stateSpace[:,0] == X1_min)
                I_UB_1 = (stateSpace[:,0] == X1_max)
                I_LB_2 = (stateSpace[:,1] == X2_min)
                I_UB_2 = (stateSpace[:,1] == X2_max)
                I_LB_3 = (stateSpace[:,2] == X3_min)
                I_UB_3 = (stateSpace[:,2] == X3_max)
                diag_0_base = A_1d[:]
                diag_0_base += (I_LB_1 * C_1_1d[:] + I_UB_1 * C_1_1d[:] - 2 * (1 - I_LB_1 - I_UB_1) * C_1_1d[:]) / dVec[0] ** 2
                diag_0_base += (I_LB_2 * C_2_1d[:] + I_UB_2 * C_2_1d[:] - 2 * (1 - I_LB_2 - I_UB_2) * C_2_1d[:]) / dVec[1] ** 2
                diag_0_base += (I_LB_3 * C_3_1d[:] + I_UB_3 * C_3_1d[:] - 2 * (1 - I_LB_3 - I_UB_3) * C_3_1d[:]) / dVec[2] ** 2
                diag_1p_base = - 2 * I_LB_1 * C_1_1d[:] / dVec[0] ** 2 + (1 - I_LB_1 - I_UB_1) * C_1_1d[:] / dVec[0] ** 2
                diag_1m_base = - 2 * I_UB_1 * C_1_1d[:] / dVec[0] ** 2 + (1 - I_LB_1 - I_UB_1) * C_1_1d[:] / dVec[0] ** 2
                diag_2p_base = - 2 * I_LB_2 * C_2_1d[:] / dVec[1] ** 2 + (1 - I_LB_2 - I_UB_2) * C_2_1d[:] / dVec[1] ** 2
                diag_2m_base = - 2 * I_UB_2 * C_2_1d[:] / dVec[1] ** 2 + (1 - I_LB_2 - I_UB_2) * C_2_1d[:] / dVec[1] ** 2
                diag_3p_base = - 2 * I_LB_3 * C_3_1d[:] / dVec[2] ** 2 + (1 - I_LB_3 - I_UB_3) * C_3_1d[:] / dVec[2] ** 2
                diag_3m_base = - 2 * I_UB_3 * C_3_1d[:] / dVec[2] ** 2 + (1 - I_LB_3 - I_UB_3) * C_3_1d[:] / dVec[2] ** 2
                diag_11p = I_LB_1 * C_1_1d[:] / dVec[0] ** 2
                diag_11m = I_UB_1 * C_1_1d[:] / dVec[0] ** 2
                diag_22p = I_LB_2 * C_2_1d[:] / dVec[1] ** 2
                diag_22m = I_UB_2 * C_2_1d[:] / dVec[1] ** 2
                diag_33p = I_LB_3 * C_3_1d[:] / dVec[2] ** 2
                diag_33m = I_UB_3 * C_3_1d[:] / dVec[2] ** 2


    # Step (6) and (7) Formulating HJB False Transient parameters
    # See remark 2.1.4 for more details
    B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
    B_2 = np.sum(theta_ell * pi_c, axis=0) * ee
    B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2

    D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(theta_ell * pi_c, axis=0) * ee  - 0.5 * ddG * sigma_y**2 * ee**2 +  xi_g * np.exp(L_mat) * (1 - gg + gg * np.log(gg)) + np.exp(L_mat) * gg *(V_post) + xi_a * entropy

    if linearsolver == 'eigen' or linearsolver == 'both':
        start_eigen = time.time()
        out_eigen = PDESolver(stateSpace, A, B_1, B_2, B_3, C_1, C_2, C_3, D, v0, epsilon, solverType = 'False Transient')
        out_comp = out_eigen[2].reshape(v0.shape,order = "F")
        print("Eigen solver: {:3f}s".format(time.time() - start_eigen))
        if epoch % 1 == 0 and reporterror:
            v = np.array(out_eigen[2])
            res = np.linalg.norm(out_eigen[3].dot(v) - out_eigen[4])
            print("Eigen residual norm: {:g}; iterations: {}".format(res, out_eigen[0]))
            PDE_rhs = A * v0 + B_1 * dX1 + B_2 * dX2 + B_3 * dX3 + C_1 * ddX1 + C_2 * ddX2 + C_3 * ddX3 + D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)))
            print("Episode {:d} (Eigen): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))

    if linearsolver == 'petsc4py':
        bpoint1 = time.time()
        # ==== original impl ====
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        D_1d = D.ravel(order = 'F')
        v0_1d = v0.ravel(order = 'F')
        # profiling
        # bpoint2 = time.time()
        # print("reshape: {:.3f}s".format(bpoint2 - bpoint1))
        diag_0 = diag_0_base - 1 / epsilon + I_LB_1 * B_1_1d[:] / -dVec[0] + I_UB_1 * B_1_1d[:] / dVec[0] - (1 - I_LB_1 - I_UB_1) * np.abs(B_1_1d[:]) / dVec[0] + I_LB_2 * B_2_1d[:] / -dVec[1] + I_UB_2 * B_2_1d[:] / dVec[1] - (1 - I_LB_2 - I_UB_2) * np.abs(B_2_1d[:]) / dVec[1] + I_LB_3 * B_3_1d[:] / -dVec[2] + I_UB_3 * B_3_1d[:] / dVec[2] - (1 - I_LB_3 - I_UB_3) * np.abs(B_3_1d[:]) / dVec[2]
        diag_1p = I_LB_1 * B_1_1d[:] /  dVec[0] + (1 - I_LB_1 - I_UB_1) * B_1_1d.clip(min=0.0) / dVec[0] + diag_1p_base
        diag_1m = I_UB_1 * B_1_1d[:] / -dVec[0] - (1 - I_LB_1 - I_UB_1) * B_1_1d.clip(max=0.0) / dVec[0] + diag_1m_base
        diag_2p = I_LB_2 * B_2_1d[:] /  dVec[1] + (1 - I_LB_2 - I_UB_2) * B_2_1d.clip(min=0.0) / dVec[1] + diag_2p_base
        diag_2m = I_UB_2 * B_2_1d[:] / -dVec[1] - (1 - I_LB_2 - I_UB_2) * B_2_1d.clip(max=0.0) / dVec[1] + diag_2m_base
        diag_3p = I_LB_3 * B_3_1d[:] /  dVec[2] + (1 - I_LB_3 - I_UB_3) * B_3_1d.clip(min=0.0) / dVec[2] + diag_3p_base
        diag_3m = I_UB_3 * B_3_1d[:] / -dVec[2] - (1 - I_LB_3 - I_UB_3) * B_3_1d.clip(max=0.0) / dVec[2] + diag_3m_base
        # profiling
        # bpoint3 = time.time()
        # print("prepare: {:.3f}s".format(bpoint3 - bpoint2))

        data = [diag_0, diag_1p, diag_1m, diag_11p, diag_11m, diag_2p, diag_2m, diag_22p, diag_22m, diag_3p, diag_3m, diag_33p, diag_33m]
        diags = np.array([0,-increVec[0],increVec[0],-2*increVec[0],2*increVec[0],
                        -increVec[1],increVec[1],-2*increVec[1],2*increVec[1],
                        -increVec[2],increVec[2],-2*increVec[2],2*increVec[2]])
        # The transpose of matrix A_sp is the desired. Create the csc matrix so that it can be used directly as the transpose of the corresponding csr matrix.
        A_sp = spdiags(data, diags, len(diag_0), len(diag_0), format='csc')
        # A_sp = A_sp * epsilon
        # b    = -v0_1d / epsilon - D_1d
        # b    = -v0_1d - D_1d * epsilon
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
        ksp.setTolerances(rtol=1e-10)
        ksp.solve(petsc_rhs, x)
        petsc_mat.destroy()
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(X1_mat.shape,order = "F")
        end_ksp = time.time()
        # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
        print("petsc4py total: {:.3f}s".format(end_ksp - bpoint1))
        print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        if epoch % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
            PDE_rhs = A * v0 + B_1 * dX1 + B_2 * dX2 + B_3 * dX3 + C_1 * ddX1 + C_2 * ddX2 + C_3 * ddX3 + D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0) / epsilon))
            print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
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
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        D_1d   = D.ravel(order = 'F')
        v0_1d  = v0.ravel(order = 'F')
        petsclinearsystem.formLinearSystem(X1_mat_1d, X2_mat_1d, X3_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
        # profiling
        # bpoint2 = time.time()
        # print("form petsc mat: {:.3f}s".format(bpoint2 - bpoint1))
        b = v0_1d + D_1d * epsilon
        # petsc4py setting
        # petsc_mat.scale(-1./ε)
        # b = -v0_1d/ε - D_1d
        petsc_rhs = PETSc.Vec().createWithArray(b)
        x = petsc_mat.createVecRight()
        # profiling
        # bpoint3 = time.time()
        # print("form rhs and workvector: {:.3f}s".format(bpoint3 - bpoint2))


        # create linear solver
        start_ksp = time.time()
        ksp.setOperators(petsc_mat)
        ksp.setTolerances(rtol=1e-13)
        ksp.solve(petsc_rhs, x)
        # petsc_mat.destroy()
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(X1_mat.shape,order = "F")
        end_ksp = time.time()
        # profiling
        # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
        num_iter = ksp.getIterationNumber()
        # file_iter.write("%s \n" % num_iter)
        print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        if epoch % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
            PDE_rhs = A * v0 + B_1 * dX1 + B_2 * dX2 + B_3 * dX3 + C_1 * ddX1 + C_2 * ddX2 + C_3 * ddX3 + D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)/ epsilon))
            print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_ep))
    # # step 9: keep iterating until convergence
    # # rowcontent = {
        # # "epoch": epoch,
        # # "iterations": num_iter,
        # # "residual norm": ksp.getResidualNorm(),
        # # "PDE_Err": PDE_Err,
        # # "FC_Err": FC_Err
    # # }
    # # writer.writerow(rowcontent)
    e_star = ee
    i_star = ii
    x_star = xx
    v0 = out_comp
    epoch += 1

if reporterror:
    print("===============================================")
    print("Fianal epoch {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch -1, PDE_Err, FC_Err))
print("--- Total running time: %s seconds ---" % (time.time() - start_time))


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


file = open("./res_data/gamma/pre_jump_" + filename, 'wb')
pickle.dump(my_shelf, file)
file.close()
