# -*- coding: utf-8 -*-
import sys, json
import numpy as np
sys.path.insert(0, "code")
import geom, cases

D1, D2 = cases.D1, cases.D2
X1 = geom.reconstruct(D1)
X2 = geom.reconstruct(D2)
print("X1 =", np.round(X1, 6).tolist())
print("X2 =", np.round(X2, 6).tolist())
d1 = np.sort(D1[np.triu_indices(4, 1)])
d2 = np.sort(D2[np.triu_indices(4, 1)])
print("sorted distances case1:", d1.tolist())
print("sorted distances case2:", d2.tolist())
print("max |sorted diff| =", float(np.abs(d1 - d2).max()))
print("recon err1 =", float(np.abs(geom.dist_matrix(X1) - D1).max()))
print("recon err2 =", float(np.abs(geom.dist_matrix(X2) - D2).max()))
print("CM det1 =", geom.cm_det(D1), " V1 =", geom.cm_volume(D1))
print("CM det2 =", geom.cm_det(D2), " V2 =", geom.cm_volume(D2))
rms, perm = geom.unlabeled_rmsd(X1, X2)
print("min unlabeled RMSD =", rms, "perm =", perm)
print("labeled RMSD (identity) =", geom.kabsch_rmsd(X1, X2))
a1 = sorted([x[3] for x in geom.bond_angles(X1)])
a2 = sorted([x[3] for x in geom.bond_angles(X2)])
print("angles1:", [round(v, 3) for v in a1])
print("angles2:", [round(v, 3) for v in a2])
print("max |sorted angle diff| =", float(np.abs(np.array(a1) - np.array(a2)).max()))
print("angle multiset L2 =", float(np.linalg.norm(np.array(a1) - np.array(a2))))
dE = (cases.E2_HARTREE - cases.E1_HARTREE)
print("dE(Ha) =", dE, " =", dE * cases.HARTREE2KCAL, "kcal/mol =", dE * cases.HARTREE2EV, "eV =", dE * 1000, "mHa")
json.dump({"X1": X1.tolist(), "X2": X2.tolist(), "rmsd": rms, "perm": list(perm),
           "cm1": geom.cm_det(D1), "cm2": geom.cm_det(D2)}, open("results/geometry.json", "w"), indent=1)
