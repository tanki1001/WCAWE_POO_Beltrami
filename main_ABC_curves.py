import numpy as np
from scipy import special
from geometries import cubic_domain, spherical_domain, half_cubic_domain, broken_cubic_domain
from postprocess import relative_errZ,import_FOM_result
from dolfinx.fem import (form, Function, FunctionSpace, petsc)
import petsc4py
from petsc4py import PETSc
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

from operators_POO import Mesh, B1p, Loading, Simulation, import_frequency_sweep

geometry1 = 'cubic'
geometry2 = 'small'
geometry  = geometry1 + '_'+ geometry2

if   geometry2 == 'small':
    side_box = 0.11
    lc       = 8e-3
elif geometry2 == 'large':
    side_box = 0.40
    lc       = 2e-2
else :
    print("Enter your own side_box and mesh size in the code")
    side_box = 0.40
    lc       = 1e-2 #Typical mesh size : Small case : 8e-3 Large case : 2e-2

from_data_b1p = True
from_data_b2p = True

radius = 0.1

rho0 = 1.21
c0   = 343.8

freqvec = np.arange(80, 2001, 20)

from operators_POO import import_COMSOL_result

comsol_data = True

if comsol_data:
    s = geometry
    frequency, results = import_COMSOL_result(s)

if   geometry1 == 'cubic':
    geo_fct = cubic_domain
elif geometry1 == 'spherical':
    geo_fct = spherical_domain
elif geometry1 == 'half_cubic':
    geo_fct = half_cubic_domain
elif geometry1 == 'broken_cubic':
    geo_fct = broken_cubic_domain
else :
    print("WARNING : May you choose an implemented geometry")

mesh_   = Mesh(1, side_box, radius, lc, geo_fct)
loading = Loading(mesh_)

from operators_POO import B1p
ope1           = B1p(mesh_)
list_coeff_Z_j = ope1.deriv_coeff_Z(0)
simu1 = Simulation(mesh_, ope1, loading)

from operators_POO import store_results
#ope1.import_matrix(freq = 2000)
if from_data_b1p:
    s1 = 'FOM_b1p'
    s = s1 + '_' + geometry
    freqvec1, PavFOM1 = import_frequency_sweep(s)
else :
    PavFOM1 = simu1.FOM(freqvec)
    s1 = 'FOM_b1p'
    s = s1 + '_' + geometry
    freqvec1 = freqvec
    store_results(s, freqvec, PavFOM1)



from operators_POO import B2p

mesh_.set_deg(2)

ope2           = B2p(mesh_)
list_coeff_Z_j = ope2.deriv_coeff_Z(0)

loading        = Loading(mesh_)
list_coeff_F_j = loading.deriv_coeff_F(0)

simu2 = Simulation(mesh_, ope2, loading)

#ope2.import_matrix(freq = 2000)
if from_data_b2p:
    s1 = 'FOM_b2p'
    s  = s1 + '_' + geometry
    freqvec2, PavFOM2 = import_frequency_sweep(s)
else :
    freqvec2 = freqvec
    PavFOM2 = simu2.FOM(freqvec2)
    s1 = 'FOM_b2p'
    s  = s1 + '_' + geometry
    store_results(s, freqvec2, PavFOM2)

from operators_POO import B2p_beltrami

mesh_.set_deg(2)

ope2belt   = B2p_beltrami(mesh_)
loading   = Loading(mesh_)
simu2belt  = Simulation(mesh_, ope2belt, loading)

#ope2.import_matrix(freq = 2000)
from_data_b2pbelt = False
if from_data_b2pbelt:
    s1 = 'FOM_b2pbelt'
    s  = s1 + '_' + geometry
    freqvec2belt, PavFOM2belt = import_frequency_sweep(s)
else :
    freqvec2belt = np.arange(80,2001,20)
    PavFOM2belt = simu2belt.FOM(freqvec2belt)
    s1 = 'FOM_b2pbelt'
    s  = s1 + '_' + geometry
    store_results(s, freqvec2belt, PavFOM2belt)


#print(PavFOM2belt)


from operators_POO import B3p

mesh_.set_deg(3)

ope3belt    = B3p(mesh_)
loading = Loading(mesh_)

simu3belt   = Simulation(mesh_, ope3belt, loading)
freqvec3belt = np.arange(1400, 1401, 20)
#PavFOM3belt = simu3belt.FOM(freqvec3belt)


from operators_POO import plot_analytical_result_sigma

fig, ax = plt.subplots(figsize=(16,9))
simu1.plot_radiation_factor(ax, freqvec1, PavFOM1, s = 'FOM_b1p')
simu2.plot_radiation_factor(ax, freqvec2, PavFOM2,  s = 'FOM_b2p')
simu2belt.plot_radiation_factor(ax, freqvec2belt, PavFOM2belt,  s = 'FOM_b2pbelt')
#simu3belt.plot_radiation_factor(ax, freqvec3belt, PavFOM3belt,  s = 'FOM_b3pbelt')
if comsol_data:
    ax.plot(frequency, results, c = 'black', label=r'$\sigma_{COMSOL}$')
    ax.legend()

plot_analytical_result = True
if plot_analytical_result:
    plot_analytical_result_sigma(ax, freqvec, radius)
plt.savefig("test.png")
from operators_POO import least_square_err, compute_analytical_radiation_factor

Z_ana = compute_analytical_radiation_factor(freqvec, radius)

err_B1p = least_square_err(freqvec, Z_ana.real, freqvec1, simu1.compute_radiation_factor(freqvec1, PavFOM1).real)
print(f'For lc = {lc} - L2_err(B1p) = {err_B1p}')

err_B2p = least_square_err(freqvec, Z_ana.real, freqvec2, simu2.compute_radiation_factor(freqvec2, PavFOM2).real)
print(f'For lc = {lc} - L2_err(B2p) = {err_B2p}')

err_B2p_belt = least_square_err(freqvec, Z_ana.real, freqvec2belt, simu2belt.compute_radiation_factor(freqvec2belt, PavFOM2belt).real)
print(f'For lc = {lc} - L2_err(err_B2p_belt) = {err_B2p_belt}')

#err_B3p_belt = least_square_err(freqvec, Z_ana.real, freqvec3belt, simu3belt.compute_radiation_factor(freqvec3belt, PavFOM3belt).real)
#print(f'For lc = {lc} - L2_err(err_B3p_belt) = {err_B3p_belt}')