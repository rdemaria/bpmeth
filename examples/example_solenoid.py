import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
import bpmeth
from bpmeth import poly_fit


# peak gradient
k1 = 0.01  # m^-1

# Defining a smooth function from 0 to 1 of length 1 m
y = np.array([0, k1 / 2, k1])
s = np.array([0, 0.5, 1])
poly_entry = poly_fit.poly_fit(
    N=4,
    xdata=s,
    ydata=y,
    x0=[0, s[-1]],
    y0=[0, y[-1]],
    xp0=[0, s[-1]],
    yp0=[0, 0],
    xpp0=[0],
    ypp0=[0],
)

# Plotting entry
ss = np.linspace(0, 1, 100)
plt.plot(ss, poly_fit.poly_val(poly_entry, ss))

# Entry
#bs = poly_fit.poly_print(poly_entry, x="s")
bs = -0.03*s**4 + 0.04*s**3
b1 = "0.0"  # k0
b2 = "0.0"  # k1
b3 = "0.0"  # k2
a1 = "0.0"  # ks0
#a2 = -eval(bs, {"s": A_magnet_entry.s}).diff(A_magnet_entry.s) / 2  # ks1
a2 = 0.06 * s**3 - 0.06 * s**2  # ks1
a3 = "0.0"  # ks2
a4 = -0.27 * s + 0.09  # ks1
h = "0.0"
length = 2
A_magnet_entry = bpmeth.GeneralVectorPotential(
    hs=h, b=(b1, b2, b3), a=(a1, a2, a3, a4), bs=bs, nphi=10
)
x = A_magnet_entry.x
y = A_magnet_entry.y
s = A_magnet_entry.s

Bx_np, By_np, Bs_np = A_magnet_entry.get_Bfield()
Bx_sp, By_sp, Bs_sp = A_magnet_entry.get_Bfield(lambdify=False)


(
    (Bx_sp.expand().collect(A_magnet_entry.s) / x).simplify().collect(x**2 + y**2),
    (By_sp.expand().collect(A_magnet_entry.s) / y).simplify().collect(x**2 + y**2),
    Bs_sp.expand().collect(A_magnet_entry.s),
)

# div
(Bx_sp.diff(x) + By_sp.diff(y) + Bs_sp.diff(s)).simplify()

# curl
list(
    map(
        sp.simplify,
        (
            sp.diff(Bs_sp, y) - sp.diff(By_sp, s),
            sp.diff(Bx_sp, s) - sp.diff(Bs_sp, x),
            sp.diff(By_sp, x) - sp.diff(Bx_sp, y),
        ),
    )
)
