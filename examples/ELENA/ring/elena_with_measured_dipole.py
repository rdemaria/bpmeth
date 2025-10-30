import xtrack as xt
import numpy as np
import bpmeth
import matplotlib.pyplot as plt


phi = 60/180*np.pi
rho = 0.927  # https://edms.cern.ch/ui/file/1311860/2.0/LNA-MBHEK-ER-0001-20-00.pdf
dipole_h = 1/rho
l_magn = rho*phi
gap=0.076

data = np.loadtxt("../dipole/ELENA_fieldmap.csv", skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]


kf=1.0008781
guess=[ 4.27855927e-01, 7.63365939e+02, -1.44485719e+02, 2.59387643e+01, 5.81379154e-01]
dipole = bpmeth.DipoleFromFieldmap(data, 1/rho, l_magn,
                                   design_field=dipole_h*kf, shape="enge", hgap=gap/2, apt=gap,
                                   radius=0.0025, order=2, plot=False, nphi=2,
                                   guess = guess )
line=xt.Line([dipole])
line.particle_ref = xt.Particles(p0c=1.0, mass0=0.9382720813)
line.build_tracker()
p0=line.build_particles(x=[0.00])
tw=line.twiss(betx=1,bety=1,include_collective=True)
mat_dipole=line.compute_one_turn_matrix_finite_differences(p0,include_collective=True)['R_matrix']
print(mat_dipole[1,0],mat_dipole[3,2],tw.x[-1],tw.px[-1],np.linalg.det(mat_dipole))


