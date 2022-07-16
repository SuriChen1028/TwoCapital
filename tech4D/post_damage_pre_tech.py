"""
post_damage_pre_tech.py
====================================
Solver for pre tech jump HJB 
"""
# Optimization of pre technology jump HJB
# Required packages
import os
import sys
import csv
sys.path.append('../src')
from supportfunctions import *
sys.stdout.flush()
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
# import petsclinearsystem
from scipy.sparse import spdiags
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from datetime import datetime

from numpy.linalg import matrix_rank, norm, cond
import argparse

parser = argparse.ArgumentParser(description="Set gamma_3 value for pre tech jump model")
parser.add_argument("--gamma", type=int)
parser.add_argument("--eta", type=float, help="Value of eta, default = 0.17", default=0.17)
parser.add_argument("--epsilon", type=float, help="Value of epsilon, default = 0.1", default=0.1)
parser.add_argument("--fraction", type=float, help="Value of fraction of control update, default = 0.1", default=0.1)
parser.add_argument("--keep-log", default=False, action="store_true", help="Flag to keep a log of the computation")
parser.add_argument("--subset-climate", action="store_true", help="Flag to use only subset of 144 climate sensitivity parameters")
args = parser.parse_args()


reporterror = True
# Linear solver choices
# Chosse among petsc, petsc4py, eigen, both
# petsc: matrix assembled in C
# petsc4py: matrix assembled in Python
# eigen: matrix assembled in C++
# both: petsc+petsc4py
#
linearsolver = 'petsc4py'
now          = datetime.now()
current_time = now.strftime("%d-%H:%M")

print("Script starts: {:d}/{:d}-{:d}:{:d}".format(now.month, now.day, now.hour, now.minute))
print("Linear solver: " + linearsolver)


start_time = time.time()
xi_a = 100.
xi_g = 100.

gamma_3_list = np.linspace(0., 1./3., 10)
gamma_3      = gamma_3_list[args.gamma]
eta          = args.eta
# Parameters as defined in the paper
dirname = "../data/PostJump/eta_{:.4f}/".format(eta)
with open(dirname + "Ag-0.15-gamma-{:.4f}".format(gamma_3), "rb") as f:
    postjump = pickle.load(f)

delta     = postjump["delta"]
A_d       = postjump["A_d"]
A_g_prime = postjump["A_g"]
A_g       = 0.1000 # Pre tech jump productivity

alpha_d = postjump["alpha_d"]
alpha_g = postjump["alpha_g"]
sigma_d = postjump["sigma_d"]
sigma_g = postjump["sigma_g"]

varsigma = postjump["varsigma"]
# beta_f = postjump["beta_f"]
beta_f = pd.read_csv("../data/model144.csv", header=None).to_numpy()[:, 0]/1000.
# Choose subset of parameters
if args.subset_climate:
    beta_f = beta_f[[0, 35, 71, 107, 143]]
    
# beta_f = np.mean(beta_f)
phi_d = postjump["phi_d"]
phi_g = postjump["phi_g"]
########## arrival rate
# varphi = postjump["varphi"]
varphi  = 0.001
sigma_l = 0.016
alpha_l = 0.000
########## Scaling factor
# eta = postjump["eta"]

# Grids Specification

# log of Capital log K in the write-up
K_min = postjump["K_min"]
K_max = postjump["K_max"]
hK    = postjump["hK"]
K     = postjump["K"]
# Capital ratio, Kg / (Kd + Kg)
R_min = postjump["R_min"]
R_max = postjump["R_max"]
hR    = postjump["hR"]
R     = postjump["R"]
# Temperature, Y in the write-up
Temp_min = postjump["Y_min"]
Temp_max = postjump["Y_max"]
hTemp    = postjump["hY"]
Temp     = postjump["Y"]
# Post tech jump value function
v_post   = postjump["v0"]

# log of jump intensity, log lambda
L_min = - 4.
L_max = 0.
hL    = 0.5
L     = np.arange(L_min, L_max, hL)

X = K[:]
Y = R[:-40]
Z = Temp[:]
W = L

###### damage
gamma_1 = postjump["gamma_1"]
gamma_2 = postjump["gamma_2"]
# gamma_3 = postjump["gamma_3"]
y_bar   = 2.

# if os.path.exists(dirname)
filename =  "post_damage_pre_tech-" + "varphi-" + str(varphi) +"-gamma-{:.4f}".format(gamma_3) + "-{}".format(current_time)

hX    = X[1] - X[0]
nX    = len(X)
X_min = X.min()
X_max = X.max()
hY    = Y[1] - Y[0]
nY    = len(Y)
Y_min = Y.min()
Y_max = Y.max()
hZ    = Z[1] - Z[0]
nZ    = len(Z)
Z_min = Z.min()
Z_max = Z.max()
hW    = W[1] - W[0]
nW    = len(W)
W_min = W.min()
W_max = W.max()

print("Grid dimension: [{}, {}, {}, {}]".format(nX, nY, nZ, nW))
print("difference: [{}, {}, {}, {}]".format(hX, hY, hZ, hW))
print(Z.min(), Z.max())
# Discretization of the state space for numerical PDE solution.
######## post jump, 3 states
(X_mat, Y_mat, Z_mat, W_mat) = np.meshgrid(X, Y, Z, W,  indexing = 'ij')
stateSpace = np.hstack([X_mat.reshape(-1,1,order = 'F'), Y_mat.reshape(-1,1,order = 'F'), Z_mat.reshape(-1, 1, order='F'), W_mat.reshape(-1, 1, order="F")])

K_mat    = X_mat
R_mat    = Y_mat
Temp_mat = Z_mat
L_mat    = W_mat

print(X_mat.shape)
# For PETSc
X_1d = X_mat.ravel(order='F')
Y_1d = Y_mat.ravel(order='F')
Z_1d = Z_mat.ravel(order='F')
W_1d = W_mat.ravel(order="F")

lowerLims = np.array([X_min, Y_min, Z_min, W_min], dtype=np.float64)
upperLims = np.array([X_max, Y_max, Z_max, W_max], dtype=np.float64)

v0 = np.zeros(X_mat.shape)
for i in range(len(W)):
    v0[:,:,:, i] = v_post[:, :-40, :]
V_post = v0

# expand theta_ell
theta_ell = np.array([temp * np.ones_like(K_mat) for temp in beta_f])
pi_c_o = np.ones((len(beta_f), nX, nY, nZ, nW)) / len(beta_f)
pi_c = pi_c_o.copy()

with open("../data/PostJump/eta_0.1000/post_damage_pre_tech-varphi-0.001-gamma-0.0000-05-01:27", "rb") as f:
    data = pickle.load(f)
v0 = data["v0"]
# v0 = V_post
continue_mode = True

############# step up of optimization

FC_Err   = 1
epoch    = 0
tol      = 1e-6
epsilon  = args.epsilon
fraction = args.fraction
max_iter = 4000

# Emission proportional to dirty capital
ee = eta * A_d * (1 - R_mat) * np.exp(K_mat)

# First, second order derivatives of damage function
dG  = gamma_1 + gamma_2 * Temp_mat + gamma_3 * (Temp_mat - y_bar) * (Temp_mat > y_bar)
ddG = gamma_2 + gamma_3 * (Temp_mat > y_bar)

while  FC_Err > tol and epoch < max_iter:
    print("-----------------------------------")
    print("---------Epoch {}---------------".format(epoch))
    print("-----------------------------------")
    start_ep = time.time()
    vold = v0.copy()
    # Applying finite difference scheme to the value function
    ######## first order
    dX = finiteDiff(v0,0,1,hX)
    # dX[dX < 0.5] = 0.5
    dK = dX
    dY = finiteDiff(v0,1,1,hY)
    # dY[dY < 0.5] = 0.5
    dR = dY
    dZ = finiteDiff(v0,2,1,hZ)
    # dY[dY > -  1e-15] = -1e-15
    dTemp = dZ
    dW = finiteDiff(v0,3,1,hW)
    dW[dW <= 1e-5] = 1e-5
    dL = dW
    ######## second order
    ddX = finiteDiff(v0,0,2,hX)
    ddY = finiteDiff(v0,1,2,hY)
    ddZ = finiteDiff(v0,2,2,hZ)
    ddTemp = ddZ
    ddW = finiteDiff(v0,3,2,hW)

    # update control
    if epoch == 0:
        # i_d = np.zeros(Kd_mat.shape)
        # i_g = np.zeros(Kg_mat.shape)
        consumption =  A_d * (1 - R_mat ) + A_g * R_mat
        mc = delta / consumption
        mc_min = mc.min()
        mc_max = mc.max()
        i_d = 1 / phi_d - mc / phi_d / (dK - R_mat * dR)
        i_d_min = i_d.min()
        i_d_max = i_d.max()
        i_g = 1 / phi_g - mc / phi_g / (dK + (1 - R_mat) * dR)
        i_g_min = i_g.min()
        i_g_max = i_g.max()
        # i_l = (A_d - i_d ) * (1 - R_mat) + (A_g - i_g) * R_mat - delta / (np.exp(K_mat) * varphi * dL)
        # i_l = 1e-15 * np.ones(X_mat.shape)
        i_l = np.zeros(X_mat.shape)
        i_l_min = i_l.min()
        i_l_max = i_l.max()
        q = delta  * ( (A_d - i_d) * (1 - R_mat) + (A_g - i_g)  * R_mat - i_l  ) ** (-1)


        if continue_mode:
            i_d = data["i_d"]
            i_g = data["i_g"]
            i_l = data["i_l"]

    else:
        mc = np.exp(K_mat-L_mat) * varphi * dL
        mc_min = mc.min()
        mc_max = mc.max()

        i_d = 1 / phi_d - varphi * dL * np.exp(K_mat - L_mat) / phi_d / (dK - R_mat * dR)
        i_d = i_d * fraction + id_star * (1. - fraction)
        i_d_min = i_d.min()
        i_d_max = i_d.max()
        # i_d[i_d > A_d - 1e-16 ] = A_d - 1e-16
        # i_d[i_d <  0.00] =  0.000
        i_g = 1 / phi_g - varphi * dL  * np.exp(K_mat - L_mat)/ phi_g / (dK + (1 - R_mat) * dR)
        i_g = i_g * fraction + ig_star * (1. - fraction)
        i_g_min = i_g.min()
        i_g_max = i_g.max()
        # i_g[i_g > A_g  - 1e-16] = A_g - 1e-16
        # i_g[i_g <  0.000] = 0.0
        temp = (A_d - i_d ) * (1 - R_mat) + (A_g - i_g) * R_mat  - delta / (np.exp(K_mat - L_mat) * varphi * dL)
        # temp[ temp < 0.000 ] = 0.000
        # temp[ temp > A_g - 1e-15] = A_g - 1e-15
        i_l =  temp
        # i_l[i_l > 1.  - 1e-16] = 1. - 1e-16
        # i_l[i_l < 0.0] = 0.000000000
        # i_l = temp * fraction + il_star * (1. - fraction)
        i_l_min = i_l.min()
        i_l_max = i_l.max()
        # i_l[i_l > A_d - 1e-16] = A_d - 1e-16
        # i_l[i_l < -1 ] = -1
        # i_l = np.zeros(X_mat.shape)
        # i_l = 1e-5 * np.ones(X_mat.shape)

    F = ddTemp - ddG
    G = dTemp  - dG
    log_pi_c_ratio = - G * ee * theta_ell / xi_a
    pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
    pi_c = np.exp(pi_c_ratio) * pi_c_o
    pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
    pi_c = pi_c / np.sum(pi_c, axis=0)
    entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)
    # Technology
    gg = np.exp(1 / xi_g * (v0 - V_post))
    gg[gg <= 1e-16] = 1e-16
    gg[gg >= 1.] = 1.
    print("mc min: {:.10f}, mc max: {:.10f}".format(mc_min, mc_max))
    print("min id: {},\t min ig: {},\t min il: {}".format(np.min(i_d), np.min(i_g), np.min(i_l)))
    print("max id: {},\t max ig: {},\t max il: {}".format(np.max(i_d), np.max(i_g), np.max(i_l)))

    # Step (2), solve minimization problem in HJB and calculate drift distortion
    # Coefficient, details see Section 2 Pre jump on page 4.
    start_time2 = time.time()
    if epoch == 0:
        dVec = np.array([hX, hY, hZ, hW])
        increVec = np.array([1, nX, nX * nY, nX * nY * nZ], dtype=np.int32)
        # These are constant
        A = - delta * np.ones(K_mat.shape)  - np.exp(L_mat) * gg
        C_11 = 0.5 * (sigma_d * (1 - R_mat) + sigma_g * R_mat)**2
        C_22 = 0.5 * R_mat**2 * (1 - R_mat)**2 * (sigma_d + sigma_g)**2
        C_33 = 0.5 * (varsigma * ee)**2
        C_44 = 0.5 *  sigma_l**2 * np.ones(X_mat.shape)
        B_3 = np.sum(theta_ell * pi_c, axis=0) * ee

    if linearsolver == 'petsc4py' or linearsolver == 'petsc' or linearsolver == 'both':
        petsc_mat = PETSc.Mat().create()
        petsc_mat.setType('aij')
        petsc_mat.setSizes([nX * nY * nZ * nW, nX * nY * nZ * nW])
        petsc_mat.setPreallocationNNZ(13)
        petsc_mat.setUp()
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        ksp.setType('bcgs')
        ksp.getPC().setType('ilu')
        ksp.setFromOptions()

        A_1d = A.ravel(order = 'F')
        C_11_1d = C_11.ravel(order = 'F')
        C_22_1d = C_22.ravel(order = 'F')
        C_33_1d = C_33.ravel(order = 'F')
        C_44_1d = C_44.ravel(order = 'F')

        if linearsolver == 'petsc4py':
            I_LB_1 = (stateSpace[:,0] == X_min)
            I_UB_1 = (stateSpace[:,0] == X_max)
            I_LB_2 = (stateSpace[:,1] == Y_min)
            I_UB_2 = (stateSpace[:,1] == Y_max)
            I_LB_3 = (stateSpace[:,2] == Z_min)
            I_UB_3 = (stateSpace[:,2] == Z_max)
            I_LB_4 = (stateSpace[:,3] == W_min)
            I_UB_4 = (stateSpace[:,3] == W_max)
            diag_0_base = A_1d[:]
            diag_0_base += (I_LB_1 * C_11_1d[:] + I_UB_1 * C_11_1d[:] - 2 * (1 - I_LB_1 - I_UB_1) * C_11_1d[:]) / dVec[0] ** 2
            diag_0_base += (I_LB_2 * C_22_1d[:] + I_UB_2 * C_22_1d[:] - 2 * (1 - I_LB_2 - I_UB_2) * C_22_1d[:]) / dVec[1] ** 2
            diag_0_base += (I_LB_3 * C_33_1d[:] + I_UB_3 * C_33_1d[:] - 2 * (1 - I_LB_3 - I_UB_3) * C_33_1d[:]) / dVec[2] ** 2
            diag_0_base += (I_LB_4 * C_44_1d[:] + I_UB_4 * C_44_1d[:] - 2 * (1 - I_LB_4 - I_UB_4) * C_44_1d[:]) / dVec[3] ** 2

            diag_1_base = - 2 * I_LB_1 * C_11_1d[:] / dVec[0] ** 2 + (1 - I_LB_1 - I_UB_1) * C_11_1d[:] / dVec[0] ** 2
            diag_1m_base = - 2 * I_UB_1 * C_11_1d[:] / dVec[0] ** 2 + (1 - I_LB_1 - I_UB_1) * C_11_1d[:] / dVec[0] ** 2
            diag_2_base = - 2 * I_LB_2 * C_22_1d[:] / dVec[1] ** 2 + (1 - I_LB_2 - I_UB_2) * C_22_1d[:] / dVec[1] ** 2
            diag_2m_base = - 2 * I_UB_2 * C_22_1d[:] / dVec[1] ** 2 + (1 - I_LB_2 - I_UB_2) * C_22_1d[:] / dVec[1] ** 2
            diag_3_base = - 2 * I_LB_3 * C_33_1d[:] / dVec[2] ** 2 + (1 - I_LB_3 - I_UB_3) * C_33_1d[:] / dVec[2] ** 2
            diag_3m_base = - 2 * I_UB_3 * C_33_1d[:] / dVec[2] ** 2 + (1 - I_LB_3 - I_UB_3) * C_33_1d[:] / dVec[2] ** 2
            diag_4_base = - 2 * I_LB_4 * C_44_1d[:] / dVec[3] ** 2 + (1 - I_LB_4 - I_UB_4) * C_44_1d[:] / dVec[3] ** 2
            diag_4m_base = - 2 * I_UB_4 * C_44_1d[:] / dVec[3] ** 2 + (1 - I_LB_4 - I_UB_4) * C_44_1d[:] / dVec[3] ** 2
            diag_11 = I_LB_1 * C_11_1d[:] / dVec[0] ** 2
            diag_11m = I_UB_1 * C_11_1d[:] / dVec[0] ** 2
            diag_22 = I_LB_2 * C_22_1d[:] / dVec[1] ** 2
            diag_22m = I_UB_2 * C_22_1d[:] / dVec[1] ** 2
            diag_33 = I_LB_3 * C_33_1d[:] / dVec[2] ** 2
            diag_33m = I_UB_3 * C_33_1d[:] / dVec[2] ** 2
            diag_44 = I_LB_4 * C_44_1d[:] / dVec[3] ** 2
            diag_44m = I_UB_4 * C_44_1d[:] / dVec[3] ** 2


    mu_d = alpha_d + i_d - 0.5 * phi_d * i_d**2
    mu_g = alpha_g + i_g - 0.5 * phi_g * i_g**2
    B_1 = mu_d * (1 - R_mat) + mu_g * R_mat - C_11
    B_2 = (mu_g - mu_d - R_mat * sigma_g**2 + (1 - R_mat) * sigma_d**2) * R_mat * (1 - R_mat)
    temp2 =  (A_d - i_d ) * (1 - R_mat)  + (A_g - i_g) * R_mat
    B_4 =  varphi * temp2 * np.exp(K_mat - L_mat) - alpha_l - 0.5 * sigma_l**2

    consumption = (A_d - i_d) * (1 - R_mat)  +  (A_g - i_g) * R_mat  - i_l
    consumption_min = consumption.min()
    consumption_max = consumption.max()
    print("max consum: {},\t min consum: {}\t".format(np.max(consumption), np.min(consumption)))
    consumption[consumption <= 1e-14] = 1e-14

    D = delta * np.log(consumption) + delta * K_mat - delta - dG * np.sum(theta_ell * pi_c, axis=0) * ee  - 0.5 * ddG * (varsigma * ee)**2 + gg * np.exp(L_mat) * V_post + xi_g * np.exp(L_mat) * (1 - gg + gg * np.log(gg)) + xi_a * entropy

    if linearsolver == 'eigen' or linearsolver == 'both':
        start_eigen = time.time()
        out_eigen = PDESolver(stateSpace, A, B_1, B_2, B_3, C_11, C_22, C_33, D, v0, epsilon, tol=-9, solverType = 'False Transient')
        out_comp = out_eigen[2].reshape(v0.shape,order = "F")
        print("Eigen solver: {:3f}s".format(time.time() - start_eigen))
        if epoch % 1 == 0 and reporterror:
            v = np.array(out_eigen[2])
            res = np.linalg.norm(out_eigen[3].dot(v) - out_eigen[4])
            print("Eigen residual norm: {:g}; iterations: {}".format(res, out_eigen[0]))
            PDE_rhs = A * v0 + B_1 * dX + B_2 * dY + B_3 * dZ + B_4 * dW + C_11 * ddX + C_22 * ddY + C_33 * ddZ + C_44 * ddW + D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)/ epsilon))
            print("Episode {:d} (Eigen): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))

    if linearsolver == 'petsc4py':
        bpoint1 = time.time()
        # ==== original impl ====
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        B_4_1d = B_4.ravel(order = 'F')
        D_1d = D.ravel(order = 'F')
        v0_1d = v0.ravel(order = 'F')
        # profiling
        # bpoint2 = time.time()
        # print("reshape: {:.3f}s".format(bpoint2 - bpoint1))
        diag_0 = diag_0_base - 1 / epsilon + I_LB_1 * B_1_1d[:] / -dVec[0] + I_UB_1 * B_1_1d[:] / dVec[0] - (1 - I_LB_1 - I_UB_1) * np.abs(B_1_1d[:]) / dVec[0] + I_LB_2 * B_2_1d[:] / -dVec[1] + I_UB_2 * B_2_1d[:] / dVec[1] - (1 - I_LB_2 - I_UB_2) * np.abs(B_2_1d[:]) / dVec[1] + I_LB_3 * B_3_1d[:] / -dVec[2] + I_UB_3 * B_3_1d[:] / dVec[2] - (1 - I_LB_3 - I_UB_3) * np.abs(B_3_1d[:]) / dVec[2]+ I_LB_4 * B_4_1d[:] / -dVec[3] + I_UB_4 * B_4_1d[:] / dVec[3] - (1 - I_LB_4 - I_UB_4) * np.abs(B_4_1d[:]) / dVec[3]
        diag_1 = I_LB_1 * B_1_1d[:] / dVec[0] + (1 - I_LB_1- I_UB_1) * B_1_1d.clip(min=0.0) / dVec[0] + diag_1_base
        diag_1m = I_UB_1 * B_1_1d[:] / -dVec[0] - (1 - I_LB_1 - I_UB_1) * B_1_1d.clip(max=0.0) / dVec[0] + diag_1m_base
        diag_2 = I_LB_2 * B_2_1d[:] / dVec[1] + (1 - I_LB_2 - I_UB_2) * B_2_1d.clip(min=0.0) / dVec[1] + diag_2_base
        diag_2m = I_UB_2 * B_2_1d[:] / -dVec[1] - (1 - I_LB_2 - I_UB_2) * B_2_1d.clip(max=0.0) / dVec[1] + diag_2m_base
        diag_3 = I_LB_3 * B_3_1d[:] / dVec[2] + (1 - I_LB_3 - I_UB_3) * B_3_1d.clip(min=0.0) / dVec[2] + diag_3_base
        diag_3m = I_UB_3 * B_3_1d[:] / -dVec[2] - (1 - I_LB_3 - I_UB_3) * B_3_1d.clip(max=0.0) / dVec[2] + diag_3m_base
        diag_4 = I_LB_4 * B_4_1d[:] / dVec[3] + (1 - I_LB_4 - I_UB_4) * B_4_1d.clip(min=0.0) / dVec[3] + diag_4_base
        diag_4m = I_UB_4 * B_4_1d[:] / -dVec[3] - (1 - I_LB_4 - I_UB_4) * B_4_1d.clip(max=0.0) / dVec[3] + diag_4m_base
        # profiling
        # bpoint3 = time.time()
        # print("prepare: {:.3f}s".format(bpoint3 - bpoint2))

        data = [diag_0, diag_1, diag_1m, diag_11, diag_11m, diag_2, diag_2m, diag_22, diag_22m, diag_3, diag_3m, diag_33, diag_33m, diag_4, diag_4m, diag_44, diag_44m]
        diags = np.array([0,-increVec[0],increVec[0],-2*increVec[0],2*increVec[0],
                        -increVec[1],increVec[1],-2*increVec[1],2*increVec[1],
                        -increVec[2],increVec[2],-2*increVec[2],2*increVec[2],
                        -increVec[3],increVec[3],-2*increVec[3],2*increVec[3]])
        # The transpose of matrix A_sp is the desired. Create the csc matrix so that it can be used directly as the transpose of the corresponding csr matrix.
        A_sp = spdiags(data, diags, len(diag_0), len(diag_0), format='csc')
        A_sp = A_sp * epsilon
        # A_dense = A_sp.todense()
        # b = -v0_1d/epsilon - D_1d
        b = -v0_1d - D_1d * epsilon

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
        ksp.setTolerances(rtol=1e-9)
        ksp.solve(petsc_rhs, x)
        petsc_mat.destroy()
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(X_mat.shape,order = "F")
        end_ksp = time.time()
        # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
        print("petsc4py total: {:.3f}s".format(end_ksp - bpoint1))
        print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        if epoch % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
            PDE_rhs = A * v0 + B_1 * dX + B_2 * dY + B_3 * dZ  + B_4 * dW + C_11 * ddX + C_22 * ddY + C_33 * ddZ + C_44 * ddW +  D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)/epsilon))
            print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
            # profling
            # bpoint7 = time.time()
            # print("compute error: {:.3f}s".format(bpoint7 - bpoint6))
        # if linearsolver == 'both':
            # compare
            # csr_mat = csr_mat*(-ε)
            # b = b * (-ε)
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
        B_4_1d = B_4.ravel(order = 'F')
        D_1d = D.ravel(order = 'F')
        v0_1d = v0.ravel(order = 'F')
        petsclinearsystem.formLinearSystem(X_1d, Y_1d, Z_1d, W_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, B_4_1d, C_11_1d, C_22_1d, C_33_1d, C_44_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
        # profiling
        # bpoint2 = time.time()
        # print("form petsc mat: {:.3f}s".format(bpoint2 - bpoint1))
        b = v0_1d + D_1d*epsilon
        # petsc4py setting
        # petsc_mat.scale(-1./ε)
        # b = -v0_1d/ε - D_1d
        petsc_rhs = PETSc.Vec().createWithArray(b)
        print(matrix_rank(petsc_rhs))
        # print(matrix_rank(petsc_mat))
        x = petsc_mat.createVecRight()
        # profiling
        # bpoint3 = time.time()
        # print("form rhs and workvector: {:.3f}s".format(bpoint3 - bpoint2))
        # x.set(0)
        # viewer = PETSc.Viewer().createBinary('A.dat', 'w')
        # petsc_mat.view(viewer)
        # viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_b.dat', 'w')
        # petsc_rhs.view(viewer)
        # ai, aj, av = petsc_mat.getValuesCSR()
        # x.set(0)
        # viewer = PETSc.Viewer().createBinary('A.dat', 'w')
        # petsc_mat.view(viewer)
        # viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_b.dat', 'w')
        # petsc_rhs.view(viewer)
        # ai, aj, av = petsc_mat.getValuesCSR()
        # print(type(x))
        print(type(petsc_mat))
        print(type(petsc_rhs))
        # print(aj)


        # create linear solver
        start_ksp = time.time()
        ksp.setOperators(petsc_mat)
        print(petsc_mat.norm())
        ksp.setTolerances(rtol=1e-12)
        ksp.solve(petsc_rhs, x)
        # petsc_mat.destroy()
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(Kd_mat.shape,order = "F")
        end_ksp = time.time()
        # profiling
        # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
        num_iter = ksp.getIterationNumber()
        # file_iter.write("%s \n" % num_iter)
        print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        if epoch % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
            PDE_rhs = A * v0 + B_1 * dX + B_2 * dY + B_3 * dZ + B_4 * dW + C_11 * ddX + C_22 * ddY + C_33 * ddZ + C_44 * ddW + D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)/ epsilon))
            print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_ep))
    # step 9: keep iterating until convergence
    id_star = i_d
    ig_star = i_g
    il_star = i_l
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


file = open(dirname + filename, 'wb')
pickle.dump(my_shelf, file)
file.close()
