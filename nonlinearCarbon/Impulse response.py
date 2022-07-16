#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Wed Jan 13 20:48:20 2021

@author: erik
"""

# # version including anthropogenic emissions

import os
import numpy as np
import configparser 
import sys
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import SymLogNorm
import matplotlib.mlab
import scipy.io as sio
import pandas as pd
import scipy.optimize as optim
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy import fft, arange, signal

# +
sa = 1
Ts = 286.7 + 0.56 # 282.9
Cs = 389 # 275.5

Q0 = 342.5
p = 0.3
## outgoing radiation linearized
kappa = 1.74
Tkappa = 154
# land albedo
alphaland = 0.28328
## Volcanism
V = 0.028
## CO2 radiative forcing
# Greenhouse effect parameter
B = 5.35
# CO2 params. C0 is the reference C02 level
C0 = 280
#wa = 0.05
#cod = 0.15
alphaland = 0.28
bP = 0 #0.05
bB = 0 #0.08
cod = 3.035

cearth = 0.107
tauc = 20
# cearth = 35.
# tauc = 6603.

coc0 =350
## Ocean albedo parameters
Talphaocean_low = 219
Talphaocean_high = 299
alphaocean_max = 0.84
alphaocean_min = 0.255

Cbio_low = 50
Cbio_high = 700

T0 = 298
C0 = 280

## CO2 uptake by vegetation
wa = 0.015
vegcover = 0.4

Thigh = 315
Tlow = 282
Topt1 = 295
Topt2 = 310
acc = 5
# -

# ### INPUT PARAMETERS

# +
PULSE = 1

Ce = 0. * np.ones(102)
if PULSE == 1:
    
    Ce[0] = 1000 / 2.13 # 100 GtC carbon impule, uncoment this -> with impulse
# Ce[0] += 1 / 2.13
Cc = 1.34 * 12 / 44 * 1000 / 2.13 + np.cumsum(Ce)
Cc 
# -

# ### FUNCTIONS

# +
# Anthropogenic carbon fitting with cubic spline
t_val = np.linspace(0, len(Ce)-1, len(Ce))

def Yam(t):
    t_points = t_val
    em_points = Ce
    
    tck = interpolate.splrep(t_points, em_points)
    return interpolate.splev(t,tck)

# Ocean albedo
def alphaocean(T):
    if T < Talphaocean_low:
        return alphaocean_max
    elif T < Talphaocean_high:
        return alphaocean_max + (alphaocean_min - alphaocean_max) / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
    else: # so T is higher
        return alphaocean_min

#Fraction of ocean covered by ice
def fracseaice(T):
    if T < Talphaocean_low:
        return 1
    elif T < Talphaocean_high:
        return 1 - 1 / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
    else: # so T is higher
        return 0

def biopump(Cc):
    if Cc < Cbio_low:
        return 1
    elif Cc < Cbio_high:
        return 1 - 1 / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
        #return 1 - 2 / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
    else: # so Cc is higher
        return 0
        #return -1

biopump = np.vectorize(biopump)
biomodulation = [biopump(val) for val in Cc]
biomod = np.float_(biomodulation)


def bioefficiency(t):
    t_points = t_val
    em_points = biomod
    
    tck = interpolate.splrep(t_points, em_points)
    return interpolate.splev(t,tck)

# Vegetation growth function
def veggrowth(T):
    if T < Tlow:
        return 0
    if (T >= Tlow) and (T < Topt1):
        return acc / (Topt1 - Tlow) * (T - Tlow)
    if (T >= Topt1) and (T <= Topt2):
        return acc
    if (T > Topt2) and (T < Thigh):
        #return acc
        return acc / (Topt2 - Thigh) * (T - Thigh)
    if T > Thigh:
        #return acc
        return 0
    
#Incoming radiation modified by albedo
def Ri(T):
    return 1/cearth * (Q0 * (1 - p * alphaland - (1 - p) * alphaocean(T)))

# Outgoing radiation modified by greenhouse effect
def Ro(T, C):
    return 1/cearth * (kappa * (T - Tkappa) -  B * np.log(C / C0))

#Solubility of atmospheric carbon into the oceans
# carbon pumps
def kappaP(T):
    np.exp(-bP * (T - T0))

# def kappaB(T):
#     np.exp(bB * (T - T0))

#Sum of two terms that reflect, respectively, the physical (or solubility) carbon pump in the ocean and Wally Broecker’s “biopump”, due to thermally enhanced bioproductivity (Fowler et al., 2013)
def oceanatmphysflux(T):
    return 1 / tauc * (coc0 * (np.exp(-bP * (T - T0))))

def oceanbioflux(T,t):
     return 1/tauc * (coc0 * (np.exp(bB * bioefficiency(t) * (T - T0))))

def oceanatmcorrflux(C):
    return 1 / tauc * (- cod * C)


# -

def dydt(t, y):
    T = y[0]
    C = y[1]

    dT = Ri(T) 
    dT -= Ro(T, C)
    
    dC = V
    dC += Yam(t) * sa                                  #  anthropogenic emissions from Ca spline                                                # volcanism 
    dC -= wa * C * vegcover * veggrowth(T)             # carbon uptake by vegetation
    dC += oceanatmphysflux(T) * (1 - fracseaice(T))    # physical solubility into ocean * fraction of ice-free ocean
    dC += oceanbioflux(T,t) * (1 - fracseaice(T))      # biological pump flux * fraction sea ice
#     dC += oceanbioflux(T) * (1 - fracseaice(T))      # biological pump flux * fraction sea ice
    dC += oceanatmcorrflux(C) * (1 - fracseaice(T))    # correction parameter

    return dT, dC


# +
init = [Ts, Cs]
t_eval = np.linspace(0, 100, 100000)
sol = solve_ivp(dydt, t_eval[[0, -1]], init, t_eval=t_eval, method='RK45', max_step=0.1)

#Extract values of temperature and C02
Tv = sol.y[0, :]
Cv = sol.y[1, :]
tv = sol.t
Tvmid = Tv - 286.7
# -

plt.plot(tv, Tvmid)

# +
if Ce[0] > 0:
    filename = f"pulse_new_{cearth}_{tauc}_{int(Ce[0])}.npy"
else:
    filename = f"base_new_{cearth}_{tauc}_{int(Ce[0])}.npy"

np.save(filename, [tv, Tvmid, Cv])
