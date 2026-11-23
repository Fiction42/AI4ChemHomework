# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "code")
import numpy as np
import basis_eval as be
import mpmath as mp
xs = np.concatenate([np.logspace(-8, 0, 30), np.linspace(1.0, 30, 40)])
err = 0.0
for l in range(7):
    mine = np.asarray(be.sph_bessel_j(l, xs))
    ref = np.array([float(mp.besselj(l + 0.5, x) * mp.sqrt(mp.pi / (2 * x))) for x in xs])
    e = float(np.max(np.abs(mine - ref) / np.maximum(1.0, np.abs(ref))))
    err = max(err, e)
    print("l=", l, "rel/abs err=", "%.3e" % e)
print("max err =", "%.3e" % err)
