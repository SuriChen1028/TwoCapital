# Optimization of poat jump HJB
#Required packages
import os
import sys
import csv
sys.path.append('../src')
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

from numpy.linalg import matrix_rank, norm, cond

reporterror = True
# Linear solver choices
# Chosse among petsc, petsc4py, eigen, both
# petsc: matrix assembled in C
# petsc4py: matrix assembled in Python
# eigen: matrix assembled in C++
# both: petsc+petsc4py
#
linearsolver = 'petsc'
current_time = datetime.now()

log_name = linearsolver + "-log-" + "{:d}-{:d}-{:d}".format(current_time.day, current_time.hour, current_time.minute)
# sys.stdout = open("./logs/" + log_name, "a")

filename =  "res-" + linearsolver  + '-' + "{:d}-{:d}-{:d}".format(current_time.day, current_time.hour, current_time.minute)

print("Script starts: {:d}/{:d}-{:d}:{:d}".format(current_time.month, current_time.day, current_time.hour, current_time.minute))
print("Linear solver: " + linearsolver)


start_time = time.time()
# Parameters as defined in the paper
delta = 0.01
A_d = 0.12
A_g = 0.15

alpha_d = -0.02
alpha_g = -0.02
sigma_d = 0.016
sigma_g = 0.016

varsigma = 1.2 * 1.86 / 1000
phi_d = 8
phi_g = 8
########## arrival rate
varphi = 0.1
sigma_lam = 0.016
########## Scaling factor
eta = 0.17


###### damage
gamma_1 = 0.00017675
gamma_2 = 2. * 0.0022
gamma_3 = 0

y_bar = 2
beta_f = 1.86 / 1000

# Grids Specification
# temperature anomaly
Y_min = 0.
Y_max = 4.
# range of capital
Kd_min = 600
Kd_max = 10000.
Kg_min = 600.
Kg_max = 10000.

# hR = 0.05
hY  = 0.2 # make sure it is float instead of int
hKd = 200.
hKg = 200.

# R = np.arange(R_min, R_max + hR, hR)
# nR = len(R)
Y = np.arange(Y_min, Y_max + hY, hY)
nY = len(Y)
Kd = np.arange(Kd_min, Kd_max + hKd, hKd)
nKd = len(Kd)
Kg = np.arange(Kg_min, Kg_max + hKg, hKg)
nKg = len(Kg)



print("Grid dimension: [{}, {}, {}]".format(nKd, nKg, nY))
print("difference: [{}, {}, {}]".format(hKd, hKg, hY))
# Discretization of the state space for numerical PDE solution.
######## post jump, 3 states
(Kd_mat, Kg_mat, Y_mat) = np.meshgrid(Kd, Kg, Y, indexing = 'ij')
stateSpace = np.hstack([Kd_mat.reshape(-1,1,order = 'F'), Kg_mat.reshape(-1,1,order = 'F'), Y_mat.reshape(-1, 1, order='F')])

print(Kd_mat.shape)
# For PETSc
Kd_mat_1d =Kd_mat.ravel(order='F')
Kg_mat_1d = Kg_mat.ravel(order='F')
Y_mat_1d = Y_mat.ravel(order='F')
lowerLims = np.array([Kd_min, Kg_min, Y_min], dtype=np.float64)
upperLims = np.array([Kd_max, Kg_max, Y_max], dtype=np.float64)


v0 = np.log(Kd_mat + Kg_mat) - beta_f * Y_mat - beta_f * eta * A_d * Kd_mat
# import pickle
# data = pickle.load(open("data/res_13-1-37", "rb"))
# v0 = data["v0"]
############# step up of optimization
FC_Err = 1
epoch = 0
tol = 1e-6
epsilon = 1.
fraction = 0.01


# test_name = "test"
# csvfile = open(test_name + ".csv", "w")
# fieldnames = ["epoch", "iterations", "residual norm",  "PDE_Err", "FC_Err", "ig_min", "ig_max", "mc_min", "mc_max"]
# writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
# writer.writeheader()

max_iter = 10000
# file_iter = open("iter_c_compile.txt", "w")
while FC_Err > tol and epoch < max_iter:
    print("-----------------------------------")
    print("---------Epoch {}---------------".format(epoch))
    print("-----------------------------------")
    start_ep = time.time()
    vold = v0.copy()
    # Applying finite difference scheme to the value function
    ######## first order
    dKd = finiteDiff(v0,0,1,hKd)
    # dKd[dKd < 1+ 1e-8] = 1 + 1e-8
    dKg = finiteDiff(v0,1,1,hKg)
    # dKg[dKg < 1 +1e-8] = 1+ 1e-8
    dY = finiteDiff(v0,2,1,hY)
    ######## second order
    ddKd = finiteDiff(v0,0,2,hKd)
    ddKg = finiteDiff(v0,1,2,hKg)
    ddY = finiteDiff(v0,2,2,hY)

    # if epoch > 2000:
        # epsilon = 0.1
    # elif epoch > 1000:
        # epsilon = 0.2
    # else:
        # pass

    # update control
    if epoch == 0:
        # i_d = np.zeros(Kd_mat.shape)
        # i_g = np.zeros(Kg_mat.shape)
        consumption_0 = A_d * Kd_mat  + A_g * Kg_mat
        consumption = consumption_0
        mc = delta / consumption
        i_d = 1 / phi_d - mc / phi_d / dKd
        # i_d[i_d < 0] = 0
        # i_d[i_d > 1] = 1
        i_g = 1 / phi_g - mc / phi_g / dKg
        # i_g[i_g < 0] = 0
        # i_g[i_g > 1] = 1
        q = delta * ((A_g * Kg_mat - i_g * Kg_mat) + (A_d * Kd_mat - i_d * Kd_mat)) ** (-1)
    else:
        pass

        # consumption_new = (A_d - i_d) * Kd_mat + (A_g - i_g) * Kg_mat
        # consumption_new[consumption_new <= 1e-8] = 1e-8
        # consumption_new[consumption_new > consumption_0] = consumption_0[consumption_new > consumption_0]
        # mc = delta / consumption_new
        # i_d = (1 / phi_d - mc / phi_d / dKd) * fraction + i_d * (1 - fraction)
        # i_d[i_d < 1e-8] = 1e-8
        # i_d[i_d > A_d] = A_d - 1e-8
        # i_g = (1 / phi_g - mc / phi_g / dKg) * fraction + i_g * (1 - fraction)
        # i_g[i_g < 1e-8] = 1e-8
        # i_g[i_g > A_g] = A_g - 1e-8
     # updating controls
        Converged = 0
        num = 0

        while Converged == 0 and num < 1000:
            # i_d_1 = np.zeros(Kd_mat.shape)
            i_g_1 = (1 - q / dKg ) / phi_g
            i_d_1 = (1 - q / dKd ) / phi_d
            i_d_1[i_d_1 >= A_d] = A_d - 1e-8
            i_d_1[i_d_1 <= 1e-8] = 1e-8
            i_g_1[i_g_1 >= A_g] = A_g - 1e-8
            i_g_1[i_g_1 <= 1e-2] = 1e-2

            if np.max(abs(i_g_1 - i_g)) <= 1e-8 and np.max(abs(i_d_1 - i_d)) <= 1e-8:
                Converged = 1
                i_g = i_g_1
                i_d = i_d_1
            else:
                i_g = i_g_1
                i_d = i_d_1

                q = delta * ( (A_g * Kg_mat - i_g * Kg_mat) + (A_d * Kd_mat - i_d * Kd_mat) ) ** (-1) * fraction + (1 - fraction) * q
            num += 1

        print(num)
    # i_d = np.zeros(Kd_mat.shape)
    # # Quadratic function of ig
    # AA = phi_g * Kg_mat
    # BB = - Kg_mat - (A_d * Kd_mat + A_g * Kg_mat) * phi_g
    # CC = A_d * Kd_mat + A_g * Kg_mat - delta / dKg
    # Delta = BB**2 - 4 * AA * CC
    # Delta[Delta < 1e-8] = 1e-8
    # i_g = ( - BB - np.sqrt(Delta)) / (2 * AA)
    # mc = delta / (A_d * Kd_mat - i_d * Kd_mat + A_g * Kg_mat - A_g * i_g)
    # print(np.min(Delta), np.max(Delta))
    print(np.min(i_d), np.min(i_g))
    print(np.max(i_d), np.max(i_g))
    print(np.min(dKg), np.max(dKg))

    # i_d = (1 / 8 - np.sqrt(0.68) / 8) * np.ones(Kd_mat.shape)
    # i_g = (1 / 8 - np.sqrt(0.68) / 8) * np.ones(Kg_mat.shape)

    i_d = np.zeros(Kd_mat.shape)
    i_g = np.zeros(Kg_mat.shape)
    # i_d[i_d <= 1e-15] = 1e-15
    # i_g[i_g <= 1e-15] = 1e-15
    # i_g[i_g > A_g - 1e-15] = A_g - 1e-15

    consumption = (A_d -i_d) * Kd_mat + (A_g - i_g) * Kg_mat
    consumption[consumption < 1e-15] = 1e-15
    # i_d[i_d >= A_d] = A_d - 1e-8
    # i_g[i_g >= A_g] = A_g - 1e-8
    # Step (2), solve minimization problem in HJB and calculate drift distortion
    # See remark 2.1.3 for more details
    start_time2 = time.time()
    if epoch == 0:
        dVec = np.array([hKd, hKg, hY])
        increVec = np.array([1, nKd, nKd * nKg],dtype=np.int32)
        # These are constant
        A = - delta * np.ones(Kd_mat.shape)
        C_dd = 0.5 * sigma_d**2 * Kd_mat**2
        C_gg = 0.5 * sigma_g**2 * Kg_mat**2
        C_yy = 0.5 * (eta * varsigma * A_d * Kd_mat)** 2
        if linearsolver == 'petsc4py' or linearsolver == 'petsc' or linearsolver == 'both':
            petsc_mat = PETSc.Mat().create()
            petsc_mat.setType('aij')
            petsc_mat.setSizes([nKd*nKg*nY, nKd*nKg*nY])
            petsc_mat.setPreallocationNNZ(13)
            petsc_mat.setUp()
            ksp = PETSc.KSP()
            ksp.create(PETSc.COMM_WORLD)
            ksp.setType('bcgs')
            ksp.getPC().setType('ilu')
            ksp.setFromOptions()

            A_1d = A.ravel(order = 'F')
            C_dd_1d = C_dd.ravel(order = 'F')
            C_gg_1d = C_gg.ravel(order = 'F')
            C_yy_1d = C_yy.ravel(order = 'F')

            if linearsolver == 'petsc4py':
                I_LB_d = (stateSpace[:,0] == Kd_min)
                I_UB_d = (stateSpace[:,0] == Kd_max)
                I_LB_g = (stateSpace[:,1] == Kg_min)
                I_UB_g = (stateSpace[:,1] == Kg_max)
                I_LB_y = (stateSpace[:,2] == Y_min)
                I_UB_y = (stateSpace[:,2] == Y_max)
                diag_0_base = A_1d[:]
                diag_0_base += (I_LB_d * C_dd_1d[:] + I_UB_d * C_dd_1d[:] - 2 * (1 - I_LB_d - I_UB_d) * C_dd_1d[:]) / dVec[0] ** 2
                diag_0_base += (I_LB_g * C_gg_1d[:] + I_UB_g * C_gg_1d[:] - 2 * (1 - I_LB_g - I_UB_g) * C_gg_1d[:]) / dVec[1] ** 2
                diag_0_base += (I_LB_y * C_yy_1d[:] + I_UB_y * C_yy_1d[:] - 2 * (1 - I_LB_y - I_UB_y) * C_yy_1d[:]) / dVec[2] ** 2
                diag_d_base = - 2 * I_LB_d * C_dd_1d[:] / dVec[0] ** 2 + (1 - I_LB_d - I_UB_d) * C_dd_1d[:] / dVec[0] ** 2
                diag_dm_base = - 2 * I_UB_d * C_dd_1d[:] / dVec[0] ** 2 + (1 - I_LB_d - I_UB_d) * C_dd_1d[:] / dVec[0] ** 2
                diag_g_base = - 2 * I_LB_g * C_gg_1d[:] / dVec[1] ** 2 + (1 - I_LB_g - I_UB_g) * C_gg_1d[:] / dVec[1] ** 2
                diag_gm_base = - 2 * I_UB_g * C_gg_1d[:] / dVec[1] ** 2 + (1 - I_LB_g - I_UB_g) * C_gg_1d[:] / dVec[1] ** 2
                diag_y_base = - 2 * I_LB_y * C_yy_1d[:] / dVec[2] ** 2 + (1 - I_LB_y - I_UB_y) * C_yy_1d[:] / dVec[2] ** 2
                diag_ym_base = - 2 * I_UB_y * C_yy_1d[:] / dVec[2] ** 2 + (1 - I_LB_y - I_UB_y) * C_yy_1d[:] / dVec[2] ** 2
                diag_dd = I_LB_d * C_dd_1d[:] / dVec[0] ** 2
                diag_ddm = I_UB_d * C_dd_1d[:] / dVec[0] ** 2
                diag_gg = I_LB_g * C_gg_1d[:] / dVec[1] ** 2
                diag_ggm = I_UB_g * C_gg_1d[:] / dVec[1] ** 2
                diag_yy = I_LB_y * C_yy_1d[:] / dVec[2] ** 2
                diag_yym = I_UB_y * C_yy_1d[:] / dVec[2] ** 2


    # Step (6) and (7) Formulating HJB False Transient parameters
    # See remark 2.1.4 for more details

    B_d = (alpha_d + i_d - 0.5 * phi_d * i_d**2) * Kd_mat
    B_g = (alpha_g + i_g - 0.5 * phi_g * i_g**2) * Kg_mat
    B_y = beta_f * eta * A_d * Kd_mat

    D = delta * np.log(consumption)  - (gamma_1 + gamma_2 * Y_mat + gamma_3 * (Y_mat - 2) * (Y_mat > 2) )* beta_f * eta * A_d * Kd_mat  - 0.5 * (gamma_2 + gamma_3 * (Y_mat > 2)) * (varsigma * eta * A_d * Kd_mat )**2

    if linearsolver == 'eigen' or linearsolver == 'both':
        start_eigen = time.time()
        out_eigen = PDESolver(stateSpace, A, B_d, B_g, B_y, C_dd, C_gg, C_yy, D, v0, epsilon, tol=-9, solverType = 'False Transient')
        out_comp = out_eigen[2].reshape(v0.shape,order = "F")
        print("Eigen solver: {:3f}s".format(time.time() - start_eigen))
        if epoch % 1 == 0 and reporterror:
            v = np.array(out_eigen[2])
            res = np.linalg.norm(out_eigen[3].dot(v) - out_eigen[4])
            print("Eigen residual norm: {:g}; iterations: {}".format(res, out_eigen[0]))
            PDE_rhs = A * v0 + B_d * dKd + B_g * dKg + B_y * dY + C_dd * ddKd + C_gg * ddKg + C_yy * ddY + D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)/ epsilon))
            print("Episode {:d} (Eigen): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))

    if linearsolver == 'petsc4py':
        bpoint1 = time.time()
        # ==== original impl ====
        B_d_1d = B_d.ravel(order = 'F')
        B_g_1d = B_g.ravel(order = 'F')
        B_y_1d = B_y.ravel(order = 'F')
        D_1d = D.ravel(order = 'F')
        v0_1d = v0.ravel(order = 'F')
        # profiling
        # bpoint2 = time.time()
        # print("reshape: {:.3f}s".format(bpoint2 - bpoint1))
        diag_0 = diag_0_base - 1 / epsilon + I_LB_d * B_d_1d[:] / -dVec[0] + I_UB_d * B_d_1d[:] / dVec[0] - (1 - I_LB_d - I_UB_d) * np.abs(B_d_1d[:]) / dVec[0] + I_LB_g * B_g_1d[:] / -dVec[1] + I_UB_g * B_g_1d[:] / dVec[1] - (1 - I_LB_g - I_UB_g) * np.abs(B_g_1d[:]) / dVec[1] + I_LB_y * B_y_1d[:] / -dVec[2] + I_UB_y * B_y_1d[:] / dVec[2] - (1 - I_LB_y - I_UB_y) * np.abs(B_y_1d[:]) / dVec[2]
        diag_d = I_LB_d * B_d_1d[:] / dVec[0] + (1 - I_LB_d- I_UB_d) * B_d_1d.clip(min=0.0) / dVec[0] + diag_d_base
        diag_dm = I_UB_d * B_d_1d[:] / -dVec[0] - (1 - I_LB_d - I_UB_d) * B_d_1d.clip(max=0.0) / dVec[0] + diag_dm_base
        diag_g = I_LB_g * B_g_1d[:] / dVec[1] + (1 - I_LB_g - I_UB_g) * B_g_1d.clip(min=0.0) / dVec[1] + diag_g_base
        diag_gm = I_UB_g * B_g_1d[:] / -dVec[1] - (1 - I_LB_g - I_UB_g) * B_g_1d.clip(max=0.0) / dVec[1] + diag_gm_base
        diag_y = I_LB_y * B_y_1d[:] / dVec[2] + (1 - I_LB_y - I_UB_y) * B_y_1d.clip(min=0.0) / dVec[2] + diag_y_base
        diag_ym = I_UB_y * B_y_1d[:] / -dVec[2] - (1 - I_LB_y - I_UB_y) * B_y_1d.clip(max=0.0) / dVec[2] + diag_ym_base
        # profiling
        # bpoint3 = time.time()
        # print("prepare: {:.3f}s".format(bpoint3 - bpoint2))

        data = [diag_0, diag_d, diag_dm, diag_dd, diag_ddm, diag_g, diag_gm, diag_gg, diag_ggm, diag_y, diag_ym, diag_yy, diag_yym]
        diags = np.array([0,-increVec[0],increVec[0],-2*increVec[0],2*increVec[0],
                        -increVec[1],increVec[1],-2*increVec[1],2*increVec[1],
                        -increVec[2],increVec[2],-2*increVec[2],2*increVec[2]])
        # The transpose of matrix A_sp is the desired. Create the csc matrix so that it can be used directly as the transpose of the corresponding csr matrix.
        A_sp = spdiags(data, diags, len(diag_0), len(diag_0), format='csc')
        # A_dense = A_sp.todense()
        b = -v0_1d/epsilon - D_1d
        # A_rank = matrix_rank(A_dense)
        # b_rank = matrix_rank(b)
        # A_norm = norm(A_dense, ord=2)
        # b_norm = norm(b)
        # A_cond = cond(A_dense)
        # b_cond = cond(b)

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
        ksp.setTolerances(rtol=1e-6)
        ksp.solve(petsc_rhs, x)
        petsc_mat.destroy()
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(Kd_mat.shape,order = "F")
        end_ksp = time.time()
        # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
        print("petsc4py total: {:.3f}s".format(end_ksp - bpoint1))
        print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        if epoch % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
            PDE_rhs = A * v0 + B_d * dKd + B_g * dKg + B_y * dY + C_dd * ddKd + C_gg * ddKg + C_yy * ddY + D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)))
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
        # rowcontent = {
            # "epoch": epoch,
            # "iterations": num_iter,
            # "residual norm": ksp.getResidualNorm(),
            # "PDE_Err": PDE_Err,
            # "FC_Err": FC_Err,
            # # "A_rank": A_rank,
            # "A_norm": A_norm,
            # "A_cond": A_cond,
            # "b_rank": b_rank,
            # "b_norm": b_norm,
            # "b_cond": b_cond
        # }
        # writer.writerow(rowcontent)

    if linearsolver == 'petsc' or linearsolver == 'both':
        bpoint1 = time.time()
        B_d_1d = B_d.ravel(order = 'F')
        B_g_1d = B_g.ravel(order = 'F')
        B_y_1d = B_y.ravel(order = 'F')
        D_1d = D.ravel(order = 'F')
        v0_1d = v0.ravel(order = 'F')
        petsclinearsystem.formLinearSystem(Kd_mat_1d, Kg_mat_1d, Y_mat_1d, A_1d, B_d_1d, B_g_1d, B_y_1d, C_dd_1d, C_gg_1d, C_yy_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
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
        x.set(0)
        viewer = PETSc.Viewer().createBinary('A.dat', 'w')
        petsc_mat.view(viewer)
        viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_b.dat', 'w')
        petsc_rhs.view(viewer)
        ai, aj, av = petsc_mat.getValuesCSR()
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
            PDE_rhs = A * v0 + B_d * dKd + B_g * dKg + B_y * dY + C_dd * ddKd + C_gg * ddKg + C_yy * ddY + D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)/ epsilon))
            print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_ep))

    # step 9: keep iterating until convergence
    # rowcontent = {
        # "epoch": epoch,
        # "iterations": num_iter,
        # "residual norm": ksp.getResidualNorm(),
        # "PDE_Err": PDE_Err,
        # "FC_Err": FC_Err,
        # "ig_min": np.min(i_g),
        # "ig_max": np.max(i_g),
        # # "dKg_min": np.min(dKg),
        # # "dKg_max": np.max(dKg),
        # # "dKd_min": np.min(dKd),
        # # "dKd_max": np.max(dKd),
        # "mc_min": np.min(mc),
        # "mc_max": np.max(mc),
    # }
    # writer.writerow(rowcontent)
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


file = open("data/" + filename, 'wb')
pickle.dump(my_shelf, file)
file.close()
