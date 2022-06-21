"""
post-jump-gamma.py
========================
This file is used to solve post damage, post technology jump HJB models with different damage function realizations.
"""
# Optimization of post jump HJB
#Required packages
import os
import sys
sys.path.append('../src')
import csv
from supportfunctions import *
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import petsclinearsystem
from scipy.sparse import spdiags
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from datetime import datetime
from solver import solver_3d
import argparse
import pdb

parser = argparse.ArgumentParser(description="Set damage curvature value, and hyper parameters for the optimization problem.")
parser.add_argument("--gamma", type=int, help="Index number of gamma_3 in the list of gamma_3 values. By default, we are solving with 10 damage function, then the value could be 0,1,...,9.", default=0)
parser.add_argument("--eta", type=float, help="Value of eta, default = 0.17", default=0.17)
parser.add_argument("--epsilon", type=float, help="Value of epsilon, default = 0.1", default=0.1)
parser.add_argument("--fraction", type=float, help="Value of fraction of control update, default = 0.1", default=0.1)
parser.add_argument("--keep-log", default=False, action="store_true", help="Flag to keep a log of the computation")
args = parser.parse_args()


reporterror = True
# Linear solver choices
# Chosse among petsc, petsc4py, eigen, both
# petsc: matrix assembled in C
# petsc4py: matrix assembled in Python
# eigen: matrix assembled in C++
# both: petsc+petsc4py
#
linearsolver = 'petsc'



start_time = time.time()
# Uncertainty parameters
xi_a = 1000. # Smooth ambiguity
# Parameters as defined in the paper
delta = 0.01
A_d = 0.12
A_g = 0.15

alpha_d = -0.02
alpha_g = -0.02
sigma_d = 0.016
sigma_g = 0.016

varsigma = 1.2 * 1.86 / 1000
phi_d = 8.
phi_g = 8.
########## Scaling factor
eta = args.eta


###### damage
gamma_1 = 0.00017675
gamma_2 = 2. * 0.0022
gamma_3_list = np.linspace(0., 1./3., 10)
gamma_3 = gamma_3_list[args.gamma]

y_bar     = 2.
beta_f    = 1.86 / 1000
theta_ell = pd.read_csv("../data/model144.csv", header=None).to_numpy()[:,0]/1000.
# theta_ell = np.ones_like(theta_ell) * beta_f
# theta_ell = theta_ell[[0, 35, 71, 107, 143]]
pi_c_o    = np.ones_like(theta_ell)/len(theta_ell)
sigma_y   = 1.2 * np.mean(theta_ell)

# Grids Specification
# Coarse Grids
# range of capital
K_min     = 4.00
K_max     = 7.30
R_min     = 0.14
R_max     = 0.99
Y_min     = 1e-8
Y_max     = 2.50
hK        = 0.10
hR        = 0.01
hY        = 0.05 # make sure it is float instead of int

K         = np.arange(K_min, K_max + hK, hK)
nK        = len(K)
R         = np.arange(R_min, R_max + hR, hR)
nR        = len(R)
Y         = np.arange(Y_min, Y_max + hY, hY)
nY        = len(Y)

now       = datetime.now()
current_time = now.strftime("%d-%H:%M")

dirname  = "eta_{:.4f}".format(eta)
if not os.path.exists("../data/PostJump/" + dirname):
    os.mkdir("../data/PostJump/" + dirname + "/")

filename =  "Ag-" + str(A_g) + "-" + "gamma-{:.4f}".format(gamma_3)  + "-{}".format(current_time)

if args.keep_log:
    sys.stdout = open("../data/PostJump/eta_{:.4f}/LOG_gamma_{:.4f}_eta_{:.3f}.log".format(eta, gamma_3, eta), 'w')

print("Grid dimension: [{}, {}, {}]\n".format(nK, nR, nY))
# Discretization of the state space for numerical PDE solution.
######## post jump, 3 states
(K_mat, R_mat, Y_mat) = np.meshgrid(K, R, Y, indexing = 'ij')
stateSpace = np.hstack([K_mat.reshape(-1,1,order = 'F'), R_mat.reshape(-1,1,order = 'F'), Y_mat.reshape(-1, 1, order='F')])

# For PETSc
K_mat_1d  = K_mat.ravel(order='F')
R_mat_1d  = R_mat.ravel(order='F')
Y_mat_1d  = Y_mat.ravel(order='F')
lowerLims = np.array([K_min, R_min, Y_min], dtype=np.float64)
upperLims = np.array([K_max, R_max, Y_max], dtype=np.float64)


v0 = K_mat - (gamma_1 + gamma_2 * Y_mat)
# import pickle
# data = pickle.load(open("../data/PostJump/eta_0.0500/Ag-0.15-gamma-0.3333-10-12:45", "rb"))
# v0 = data["v0"]
pi_c = np.array([temp * np.ones_like(K_mat) for temp in pi_c_o])
pi_c_o = pi_c.copy()
theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell])
############# step up of optimization
FC_Err   = 1
epoch    = 0
tol      = 1e-7
epsilon  = args.epsilon
fraction = args.fraction

csvfile = open("HJB_3D_{:.3f}_{}.csv".format(gamma_3, current_time), "w")
fieldnames = [
        "epoch", 
        "iterations", 
        "residual norm", 
        "PDE_Err", 
        "FC_Err", 
        "id_min",
        "id_max",
        "ig_min",
        "ig_max",
        "DELTA_min",
        "DELTA_max",
        "multi_1_min",
        "multi_1_max",
        "multi_2_min",
        "multi_2_max",
        "aa_min",
        "aa_max",
        "bb_min",
        "bb_max",
        "AA_min",
        "AA_max",
        "BB_min",
        "BB_max",
        "CC_min",
        "CC_max",
        "dK_min",
        "dK_max",
        "dR_min",
        "dR_max",
        ]

writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()
max_iter = 10000

id_star = np.zeros_like(K_mat)
ig_star = np.zeros_like(K_mat)
# id_star = data["id_star"]
# ig_star = data["ig_star"]

continue_mode = True

ee  = eta * A_d * np.exp(K_mat) * (1 - R_mat)

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
    dK = finiteDiff(v0,0,1,hK)
    dK[dK <= 1e-14] = 1e-14
    dR = finiteDiff(v0,1,1,hR)
    # dR[dR <= 1e-14] = 1e-14
    dY = finiteDiff(v0,2,1,hY)
    ######## second order
    ddK = finiteDiff(v0,0,2,hK)
    ddR = finiteDiff(v0,1,2,hR)
    ddY = finiteDiff(v0,2,2,hY)


    # update control
    if epoch == 0:
        
        if continue_mode:
            i_d = id_star
            i_g = ig_star

        else:
            i_d = np.zeros(K_mat.shape)
            i_g = np.zeros(R_mat.shape)
            consumption_0 = A_d * (1 - R_mat) + A_g * R_mat
            consumption = consumption_0
            mc = delta / consumption
            i_d = 1 -  mc / (dK - R_mat *  dR)
            i_d /= phi_d
            i_d[i_d < 0] = 0
            i_g = 1 - mc / (dK + (1 - R_mat) * dR)
            i_g /= phi_g
            q = delta * ((A_g * R_mat - i_g * R_mat) + (A_d * (1 - R_mat) - i_d * (1 - R_mat))) ** (-1)
        # DELTA = np.zeros_like(K_mat)

    else:
        pass


    multi_1 = dK + (1 - R_mat) * dR
    multi_2 = dK - R_mat * dR

    dK_min = dK.min()
    dK_max = dK.max()
    dR_min = dR.min()
    dR_max = dR.max()

    multi_2_min = multi_2.min()
    multi_2_max = multi_2.max()

    print("dK min: {};\t dK max: {}\t".format(dK_min, dK_max))
    print("dR min: {};\t dR max: {}\t".format(dR_min, dR_max))

    print("m2 min: {};\t m2 max: {}\t".format(multi_2_min, multi_2_max))

    if multi_2.any() <= 0:
        import pdb; pdb.set_trace()

    multi_2[multi_2 <= 0.01] = 0.01

    aa = (1 - multi_1 / multi_2) / phi_d
    bb = phi_g / phi_d * multi_1 / multi_2

    AA = phi_g * ((1 - R_mat) * bb + R_mat)
    BB = phi_g * ((1 - R_mat) * A_d + R_mat * A_g - (1 - R_mat) * aa) + (1 - R_mat) * bb + R_mat
    CC = (1 - R_mat) * A_d + R_mat * A_g - (1 - R_mat) * aa - delta / multi_1
    DELTA = BB**2 - 4 * AA * CC

    print("DELTA min: {}\t; DELTA max: {}\t".format(DELTA.min(), DELTA.max()))

    if DELTA.any() <= 0:
        import pdb; pdb.set_trace()

    # DELTA[DELTA <= 0] = 0
    i_g_new = (BB - np.sqrt(DELTA)) / (2 * AA)
    i_d_new = aa + bb * i_g_new

    i_d = i_d_new * fraction + id_star * (1 - fraction)
    i_g = i_g_new * fraction + ig_star * (1 - fraction)
    print("Before 1e-14 constraint:")
    print("id min: {}\t; id max: {}\t".format(np.min(i_d), np.max(i_d)))
    print("ig min: {}\t; ig max: {}\t".format(np.min(i_g), np.max(i_g)))
    i_d_min_new = i_d.min()
    i_d[i_d >= 1 / phi_d - 1e-14] = 1 / phi_d - 1e-14
    i_g[i_g >= 1 / phi_g - 1e-14] = 1 / phi_g - 1e-14

    print("After 1e-14 constraint:")
    print("min id: {:.12f};\t max ig: {:.12f}\t".format(np.min(i_d), np.min(i_g)) )
    print("max id: {:.12f};\t max ig: {:.12f}\t".format(np.max(i_d), np.max(i_g)))
    consumption = (A_d -i_d) * (1 - R_mat) + (A_g - i_g) * R_mat
    consumption[consumption <= 1e-14] = 1e-14
    print("min consum: {:.12f};\t max consum: {:.12f}\t".format(np.min(consumption), np.max(consumption)))

    G = dY  - dG
    F = ddY - ddG

    log_pi_c_ratio = - G * ee * theta_ell / xi_a
    pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
    pi_c = np.exp(pi_c_ratio) * pi_c_o
    pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
    pi_c = pi_c / np.sum(pi_c, axis=0)
    entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)

    # Step (2), solve minimization problem in HJB and calculate drift distortion
    start_time2 = time.time()
    if epoch == 0:
        dVec = np.array([hK, hR, hY])
        increVec = np.array([1, nK, nK * nR],dtype=np.int32)
        # These are constant
        A    = - delta * np.ones(K_mat.shape)
        C_11 = 0.5 * ( sigma_d * (1 - R_mat) + sigma_g * R_mat )**2
        C_22 = 0.5 * (1- R_mat)**2 * R_mat**2 * (sigma_d + sigma_g)**2
        C_33 = 0.5 * (varsigma * ee)** 2
        if linearsolver == 'petsc4py' or linearsolver == 'petsc' or linearsolver == 'both':
            petsc_mat = PETSc.Mat().create()
            petsc_mat.setType('aij')
            petsc_mat.setSizes([nK*nR*nY, nK*nR*nY])
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

            if linearsolver == 'petsc4py':
                I_LB_1 = (stateSpace[:,0] == K_min)
                I_UB_1 = (stateSpace[:,0] == K_max)
                I_LB_2 = (stateSpace[:,1] == R_min)
                I_UB_2 = (stateSpace[:,1] == R_max)
                I_LB_3 = (stateSpace[:,2] == Y_min)
                I_UB_3 = (stateSpace[:,2] == Y_max)
                diag_0_base = A_1d[:]
                diag_0_base += (I_LB_1 * C_11_1d[:] + I_UB_1 * C_11_1d[:] - 2 * (1 - I_LB_1 - I_UB_1) * C_11_1d[:]) / dVec[0] ** 2
                diag_0_base += (I_LB_2 * C_22_1d[:] + I_UB_2 * C_22_1d[:] - 2 * (1 - I_LB_2 - I_UB_2) * C_22_1d[:]) / dVec[1] ** 2
                diag_0_base += (I_LB_3 * C_33_1d[:] + I_UB_3 * C_33_1d[:] - 2 * (1 - I_LB_3 - I_UB_3) * C_33_1d[:]) / dVec[2] ** 2
                diag_d_base = - 2 * I_LB_1 * C_11_1d[:] / dVec[0] ** 2 + (1 - I_LB_1 - I_UB_1) * C_11_1d[:] / dVec[0] ** 2
                diag_dm_base = - 2 * I_UB_1 * C_11_1d[:] / dVec[0] ** 2 + (1 - I_LB_1 - I_UB_1) * C_11_1d[:] / dVec[0] ** 2
                diag_g_base = - 2 * I_LB_2 * C_22_1d[:] / dVec[1] ** 2 + (1 - I_LB_2 - I_UB_2) * C_22_1d[:] / dVec[1] ** 2
                diag_gm_base = - 2 * I_UB_2 * C_22_1d[:] / dVec[1] ** 2 + (1 - I_LB_2 - I_UB_2) * C_22_1d[:] / dVec[1] ** 2
                diag_y_base = - 2 * I_LB_3 * C_33_1d[:] / dVec[2] ** 2 + (1 - I_LB_3 - I_UB_3) * C_33_1d[:] / dVec[2] ** 2
                diag_ym_base = - 2 * I_UB_3 * C_33_1d[:] / dVec[2] ** 2 + (1 - I_LB_3 - I_UB_3) * C_33_1d[:] / dVec[2] ** 2
                diag_dd = I_LB_1 * C_11_1d[:] / dVec[0] ** 2
                diag_ddm = I_UB_1 * C_11_1d[:] / dVec[0] ** 2
                diag_gg = I_LB_2 * C_22_1d[:] / dVec[1] ** 2
                diag_ggm = I_UB_2 * C_22_1d[:] / dVec[1] ** 2
                diag_yy = I_LB_3 * C_33_1d[:] / dVec[2] ** 2
                diag_yym = I_UB_3 * C_33_1d[:] / dVec[2] ** 2


    # Step (6) and (7) Formulating HJB False Transient parameters
    # See remark 2.1.4 for more details

    B_1 = (alpha_d + i_d - 0.5* phi_d * i_d**2) * (1 - R_mat) +  (alpha_g + i_g - 0.5 * phi_g * i_g**2) * R_mat - C_11
    B_2 = ((alpha_g + i_g - 0.5 * phi_g * i_g**2) -  (alpha_d + i_d - 0.5* phi_d * i_d**2) - R_mat * sigma_g**2 + (1 - R_mat)* sigma_d**2) * R_mat * (1 - R_mat)
    B_3 = np.sum(pi_c * theta_ell, axis=0) * ee
    # B_3 = beta_f * ee

    D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(pi_c * theta_ell, axis=0) * ee  - 0.5 * ddG * (varsigma * ee)**2 + xi_a * entropy

    if linearsolver == 'eigen' or linearsolver == 'both':
        start_eigen = time.time()
        out_eigen = PDESolver(stateSpace, A, B_1, B_2, B_3, C_11, C_22, C_33, D, v0, epsilon, solverType = 'False Transient')
        out_comp = out_eigen[2].reshape(v0.shape,order = "F")
        print("Eigen solver: {:3f}s".format(time.time() - start_eigen))
        if epoch % 1 == 0 and reporterror:
            v = np.array(out_eigen[2])
            res = np.linalg.norm(out_eigen[3].dot(v) - out_eigen[4])
            print("Eigen residual norm: {:g}; iterations: {}".format(res, out_eigen[0]))
            PDE_rhs = A * v0 + B_1 * dK + B_2 * dR + B_3 * dY + C_11 * ddK + C_22 * ddR + C_33 * ddY + D
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
        diag_R = I_LB_1 * B_1_1d[:] / dVec[0] + (1 - I_LB_1 - I_UB_1) * B_1_1d.clip(min=0.0) / dVec[0] + diag_R_base
        diag_Rm = I_UB_1 * B_1_1d[:] / -dVec[0] - (1 - I_LB_1 - I_UB_1) * B_1_1d.clip(max=0.0) / dVec[0] + diag_Rm_base
        diag_F = I_LB_2 * B_2_1d[:] / dVec[1] + (1 - I_LB_2 - I_UB_2) * B_2_1d.clip(min=0.0) / dVec[1] + diag_F_base
        diag_Fm = I_UB_2 * B_2_1d[:] / -dVec[1] - (1 - I_LB_2 - I_UB_2) * B_2_1d.clip(max=0.0) / dVec[1] + diag_Fm_base
        diag_K = I_LB_3 * B_3_1d[:] / dVec[2] + (1 - I_LB_3 - I_UB_3) * B_3_1d.clip(min=0.0) / dVec[2] + diag_K_base
        diag_Km = I_UB_3 * B_3_1d[:] / -dVec[2] - (1 - I_LB_3 - I_UB_3) * B_3_1d.clip(max=0.0) / dVec[2] + diag_Km_base
        # profiling
        # bpoint3 = time.time()
        # print("prepare: {:.3f}s".format(bpoint3 - bpoint2))

        data = [diag_0, diag_R, diag_Rm, diag_RR, diag_RRm, diag_F, diag_Fm, diag_FF, diag_FFm, diag_K, diag_Km, diag_KK, diag_KKm]
        diags = np.array([0,-increVec[0],increVec[0],-2*increVec[0],2*increVec[0],
                        -increVec[1],increVec[1],-2*increVec[1],2*increVec[1],
                        -increVec[2],increVec[2],-2*increVec[2],2*increVec[2]])
        # The transpose of matrix A_sp is the desired. Create the csc matrix so that it can be used directly as the transpose of the corresponding csr matrix.
        A_sp = spdiags(data, diags, len(diag_0), len(diag_0), format='csc')
        b = -v0_1d/epsilon - D_1d
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
        ksp.setTolerances(rtol=1e-14)
        ksp.solve(petsc_rhs, x)
        petsc_mat.destroy()
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(R_mat.shape,order = "F")
        end_ksp = time.time()
        # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
        print("petsc4py total: {:.3f}s".format(end_ksp - bpoint1))
        print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        if epoch % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
            PDE_rhs = A * v0 + B_1 * dK + B_2 * dR + B_3 * dY + C_11 * ddK + C_22 * ddR + C_33 * ddY + D
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
        D_1d = D.ravel(order = 'F')
        v0_1d = v0.ravel(order = 'F')
        petsclinearsystem.formLinearSystem(K_mat_1d, R_mat_1d, Y_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_11_1d, C_22_1d, C_33_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
        # profiling
        # bpoint2 = time.time()
        # print("form petsc mat: {:.3f}s".format(bpoint2 - bpoint1))
        b = v0_1d + D_1d*epsilon
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
        ksp.setTolerances(rtol=1e-16)
        ksp.solve(petsc_rhs, x)
        # petsc_mat.destroy()
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(K_mat.shape,order = "F")
        end_ksp = time.time()
        # profiling
        # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
        num_iter = ksp.getIterationNumber()
        # file_iter.write("%s \n" % num_iter)
        print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        if epoch % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
            PDE_rhs = A * v0 + B_1 * dK + B_2 * dR + B_3 * dY + C_11 * ddK + C_22 * ddR + C_33 * ddY + D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)/ epsilon))
            print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_ep))
    # step 9: keep iterating until convergence
    rowcontent = {
        "epoch": epoch,
        "iterations": num_iter,
        "residual norm": ksp.getResidualNorm(),
        "PDE_Err": PDE_Err,
        "FC_Err": FC_Err,
        "id_min": i_d_min_new,
        "id_max": i_d.max(),
        "ig_min": i_g.min(),
        "ig_max": i_g.max(),
        "DELTA_min": DELTA.min(),
        "DELTA_max": DELTA.max(),
        "multi_1_min": multi_1.min(),
        "multi_1_max": multi_1.max(),
        "multi_2_min": multi_2_min,
        "multi_2_max": multi_2_max,
        "aa_min": aa.min(),
        "aa_max": aa.max(),
        "bb_min": bb.min(),
        "bb_max": bb.max(),
        "AA_min": AA.min(),
        "AA_max": AA.max(),
        "BB_min": BB.min(),
        "BB_max": BB.max(),
        "CC_min": CC.min(),
        "CC_max": CC.max(),
        "dK_min": dK.min(),
        "dK_max": dK.max(),
        "dR_min": dR.min(),
        "dR_max": dR.max(),
    }
    writer.writerow(rowcontent)
    id_star = i_d
    ig_star = i_g
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


file = open("../data/PostJump/" + dirname + "/" + filename, 'wb')
pickle.dump(my_shelf, file)
file.close()

sys.stdout.close()
