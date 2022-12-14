"""
solver.py
For 3D abatement solver
"""
import os
import sys
sys.path.append("../../src/")
from supportfunctions import finiteDiff_3D
import SolveLinSys
import numpy as np
import petsc4py
from petsc4py import PETSc
import petsclinearsystem
import time
from datetime import datetime

# def PDESolver(stateSpace, A, B_r, B_f, B_k, C_rr, C_ff, C_kk, D, v0, ε = 1, tol = -10, smartguess = False, solverType = 'False Transient'):

#     if solverType == 'False Transient':
#         A = A.reshape(-1,1,order = 'F')
#         B = np.hstack([B_r.reshape(-1,1,order = 'F'),B_f.reshape(-1,1,order = 'F'),B_k.reshape(-1,1,order = 'F')])
#         C = np.hstack([C_rr.reshape(-1,1,order = 'F'), C_ff.reshape(-1,1,order = 'F'), C_kk.reshape(-1,1,order = 'F')])
#         D = D.reshape(-1,1,order = 'F')
#         v0 = v0.reshape(-1,1,order = 'F')
#         out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0, ε, tol)

#         return out

#     elif solverType == 'Feyman Kac':
        
#         if smartguess:
#             iters = 1
#         else:
#             iters = 400000
            
#         A = A.reshape(-1, 1, order='F')
#         B = np.hstack([B_r.reshape(-1, 1, order='F'), B_f.reshape(-1, 1, order='F'), B_k.reshape(-1, 1, order='F')])
#         C = np.hstack([C_rr.reshape(-1, 1, order='F'), C_ff.reshape(-1, 1, order='F'), C_kk.reshape(-1, 1, order='F')])
#         D = D.reshape(-1, 1, order='F')
#         v0 = v0.reshape(-1, 1, order='F')
#         out = SolveLinSys.solveFK(stateSpace, A, B, C, D, v0, iters)

#         return out

def pde_one_interation(ksp, petsc_mat, X1_mat_1d, X2_mat_1d, X3_mat_1d, lowerLims, upperLims, dVec, increVec, v0, A, B_1, B_2, B_3, C_1, C_2, C_3, D, tol, epsilon):

    bpoint1 = time.time()
    A_1d   = A.ravel(order = 'F')
    C_1_1d = C_1.ravel(order = 'F')
    C_2_1d = C_2.ravel(order = 'F')
    C_3_1d = C_3.ravel(order = 'F')
    B_1_1d = B_1.ravel(order = 'F')
    B_2_1d = B_2.ravel(order = 'F')
    B_3_1d = B_3.ravel(order = 'F')
    D_1d   = D.ravel(order = 'F')
    v0_1d  = v0.ravel(order = 'F')
    petsclinearsystem.formLinearSystem(X1_mat_1d, X2_mat_1d, X3_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
    b = v0_1d + D_1d * epsilon
    petsc_rhs = PETSc.Vec().createWithArray(b)
    x = petsc_mat.createVecRight()


    # create linear solver
    start_ksp = time.time()
    ksp.setOperators(petsc_mat)
    ksp.setTolerances(rtol=tol)
    ksp.solve(petsc_rhs, x)
    petsc_rhs.destroy()
    x.destroy()
    out_comp = np.array(ksp.getSolution()).reshape(A.shape,order = "F")
    end_ksp = time.time()
    num_iter = ksp.getIterationNumber()
    print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
    print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
    return out_comp

def _FOC_update(v0, steps= (), states = (), args=(), controls=(), fraction=0.5):

    hX1, hX2, hX3 = steps
    K_mat, Y_mat, L_mat = states
    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, sigma_g, V_post_tech, dG, ddG, xi_a, xi_g = args

    i_star, e_star, x_star = controls
    # First order derivative
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

        G = dY  - dG
        F = ddY - ddG
        mc = dL * psi_1 * psi_0 * np.exp(K_mat - L_mat)
        mc[mc <= 1e-16] = 1e-16
        temp = mc * vartheta_bar * theta / (lambda_bar * np.exp(K_mat))
        a = temp / (alpha * lambda_bar * np.exp(K_mat)) ** (theta - 1)
        b = - 2 * temp / (alpha * lambda_bar * np.exp(K_mat)) + F * sigma_y ** 2
        c = temp + G * np.sum(theta_ell * pi_c, axis=0)
        temp = b ** 2 - 4 * a * c
        temp[temp <=0] = 0
        # temp = temp * (temp > 0)
        root1 = (- b - np.sqrt(temp)) / (2 * a)
        root2 = (- b + np.sqrt(temp)) / (2 * a)
        if root1.all() > 0 :
            e_new = root1
        else:
            e_new = root2

        i_new = (1 - mc/ dK) / kappa
        j_star = alpha * vartheta_bar * (1 - e_star / (alpha * lambda_bar * np.exp(K_mat)))**theta
        j_star[j_star <= 1e-16] = 1e-16
        temp3 = alpha - i_star - j_star
        x_new = temp3 - delta / mc

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
    # update smooth ambiguity
    log_pi_c_ratio = - G * ee * theta_ell / xi_a
    pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
    pi_c = np.exp(pi_c_ratio) * pi_c_o
    pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
    pi_c = pi_c / np.sum(pi_c, axis=0)
    entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)
    # Technology
    gg = np.exp(1 / xi_g * (v0 - V_post_tech))
    gg[gg <=1e-16] = 1e-16
    # gg[gg >= 1] = 1
    jj =  alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
    jj[jj <= 1e-16] = 1e-16
    consumption = alpha - ii - jj - xx
    consumption[consumption <= 1e-16] = 1e-16
    # Step (2), solve minimization problem in HJB and calculate drift distortion
    A   = - delta * np.ones(K_mat.shape) - np.exp(L_mat) * gg
    B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
    B_2 = np.sum(theta_ell * pi_c, axis=0) * ee
    B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
    C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
    C_2 = 0.5 * sigma_y**2 * ee**2
    C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
    D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(theta_ell * pi_c, axis=0) * ee  - 0.5 * ddG * sigma_y**2 * ee**2  + xi_g * np.exp(L_mat) * (1 - gg + gg * np.log(gg)) + np.exp(L_mat) * gg * V_post_tech + entropy

    return A, B_1, B_2, B_3, C_1, C_2, C_3, D, dX1, dX2, dX3, ddX1, ddX2, ddX3, ii, ee, xx, pi_c

def hjb_pre_tech(
        state_grid=(), model_args=(), V_post_damage=None, 
        tol=1e-8, epsilon=0.1, fraction=0.5, max_iter=10000,
        v0=None,
        smart_guess=None,
        ):

    now = datetime.now()
    current_time = now.strftime("%d-%H:%M")
    K, Y, L = state_grid

    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, sigma_g, V_post_tech, gamma_1, gamma_2, gamma_3, y_bar, xi_a, xi_g, xi_p = model_args


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
    #### Model type
    if isinstance(gamma_3, (np.ndarray, list)):
        model = "Pre damage"
        pi_d_o = np.ones(len(gamma_3)) / len(gamma_3)
        pi_d_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_d_o ])
        y_bar_lower = 1.5
        r_1 = 1.5
        r_2 = 2.5
        Intensity = r_1 * (np.exp(r_2 / 2 * (Y_mat - y_bar_lower)**2) -1) * (Y_mat > y_bar_lower)
        v_i = V_post_damage
        dG  = gamma_1 + gamma_2 * Y_mat
        ddG = gamma_2 
    else:
        model = "Post damage"
        dG  = gamma_1 + gamma_2 * Y_mat + gamma_3 * (Y_mat - y_bar) * (Y_mat > y_bar)
        ddG = gamma_2 + gamma_3 * (Y_mat > y_bar)

    # Initial setup of HJB
    FC_Err   = 1
    epoch    = 0

    if v0 is None:
        v0 = K_mat + L_mat - np.average(pi_c_o, axis=0) * Y_mat

    i_star = np.zeros(K_mat.shape)
    e_star = np.zeros(K_mat.shape)
    x_star = np.zeros(K_mat.shape)
    
    if smart_guess:
        v0     = smart_guess["v0"]
        i_star = smart_guess["i_star"]
        e_star = smart_guess["e_star"]
        x_star = smart_guess["x_star"]

    dVec = np.array([hX1, hX2, hX3])
    increVec = np.array([1, nX1, nX1 * nX2],dtype=np.int32)

    FOC_args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, sigma_g, V_post_tech, dG, ddG, xi_a, xi_g )

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

    # Enter the optimization
    while FC_Err > tol and epoch < max_iter:
        print("-----------------------------------")
        print("---------Epoch {}---------------".format(epoch))
        print("-----------------------------------")
        start_ep = time.time()
        A, B_1, B_2, B_3, C_1, C_2, C_3, D, dX1, dX2, dX3, ddX1, ddX2, ddX3, ii, ee, xx, pi_c = _FOC_update(v0, steps= (hX1, hX2, hX3), states = (K_mat, Y_mat, L_mat), args=FOC_args, controls=(i_star, e_star, x_star), fraction=fraction)

        if model == "Pre damage":
            D -= xi_p * Intensity * (np.sum(pi_d_o * np.exp(- v_i / xi_p), axis=0) - np.exp(- v0 / xi_p)) / np.exp(- v0 / xi_p)


        out_comp = pde_one_interation(
                ksp,
                petsc_mat,X1_mat_1d, X2_mat_1d, X3_mat_1d, 
                lowerLims, upperLims, dVec, increVec,
                v0, A, B_1, B_2, B_3, C_1, C_2, C_3, D, 1e-13, epsilon)
        # if epoch % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
        PDE_rhs = A * v0 + B_1 * dX1 + B_2 * dX2 + B_3 * dX3 + C_1 * ddX1 + C_2 * ddX2 + C_3 * ddX3 + D
        PDE_Err = np.max(abs(PDE_rhs))
        FC_Err = np.max(abs((out_comp - v0)/ epsilon))
        print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        print("Epoch time: {:.4f}".format(time.time() - start_ep))

        v0     = out_comp
        i_star = ii
        e_star = ee
        x_star = xx
        epoch += 1

    g_tech = np.exp(1. / xi_g * (v0 - V_post_tech))
    if model == "Pre damage":
        g_damage = np.exp(1 / xi_p * (v0 - v_i))

    res = {
            "v0"    : v0,
            "i_star": i_star,
            "e_star": e_star,
            "x_star": x_star,
            "pi_c"  : pi_c,
            "g_tech": g_tech,
            }
    if model == "Pre damage":
        res = {
                "v0"    : v0,
                "i_star": i_star,
                "e_star": e_star,
                "x_star": x_star,
                "pi_c"  : pi_c,
                "g_tech": g_tech,
                "g_damage": g_damage,
                }
    return res

# def hjb_pre_damage_pre_tech
