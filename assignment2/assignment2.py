import numpy as np
import functions as fs
import chladni_show
import examplematrices as e_mat

# Part A
print("Part A")
Kmat = np.load("Chladni-Kmat.npy")

print("centers and radii of K mat:")
Kmat_centers, Kmat_radii = fs.gershgorin(Kmat)
for i in range(Kmat_centers.size):
    print(f"center:{np.round(Kmat_centers[i], 3)},radius:{np.round(Kmat_radii[i], 3)}")

# Part B
print()
print("Part B")
mats = [e_mat.A1, e_mat.A2, e_mat.A3, e_mat.A4, e_mat.A5, e_mat.A6]
eig_vals = [e_mat.eigvals1, e_mat.eigvals2, e_mat.eigvals3, e_mat.eigvals4, e_mat.eigvals5, e_mat.eigvals6]

for idx, (A, eig_val) in enumerate(zip(mats, eig_vals)):
    x, k = fs.power_iterate(A)
    approx = fs.rayleigh_qt(A, x)
    residual = np.sqrt(np.sum((A @ x - approx * x) ** 2))
    print(
        f"A{idx + 1} converged after {k} iterations,Approximate Largest Eigenvalue: "
        f"{np.round(approx, 4)},Residual: {np.round(residual, 4)}")
    print(f"Exact Largest Eigenvalue:{np.amax(eig_val)}")
    print()

x, k = fs.power_iterate(Kmat, epsilon=1e-8, max_iter=40)
approx = fs.rayleigh_qt(Kmat, x)
residual = np.sqrt(np.sum((Kmat @ x - approx * x) ** 2))
print(f'K-mat approximate largest eigenvalue: {np.round(approx, 3)}')
print(f'K-mat Residual: {np.round(residual, 3)}')
print(f'K-mat iterations: {k}')
chladni_show.show_nodes(x)
chladni_show.show_waves(x)

# Part c
print()
print("Part c")
for idx, (A, eig_val) in enumerate(zip(mats, eig_vals)):
    x, k = fs.rayleigh_iterate(A, np.random.random(size=len(A)), 0)
    approx = fs.rayleigh_qt(A, x)
    residual = np.sqrt(np.sum((A @ x - approx * x) ** 2))
    print(
        f"A{idx + 1} converged after {k} iterations, founded eigenvalue: "
        f"{np.round(approx, 4)},Residual: {np.round(residual, 4)}")
    print()

# Part d
# d2
print()
print("Part d")
K_eig_val = []
for i in range(len(Kmat_centers)):
    x, _ = fs.rayleigh_iterate(Kmat, shift=Kmat_centers[i])
    K_eig_val.append(fs.rayleigh_qt(Kmat, x))
    xl, _ = fs.rayleigh_iterate(Kmat, shift=Kmat_centers[i] - Kmat_radii[i])
    K_eig_val.append(fs.rayleigh_qt(Kmat, xl))
    xr, _ = fs.rayleigh_iterate(Kmat, shift=Kmat_centers + Kmat_radii[i])
    K_eig_val.append(fs.rayleigh_qt(Kmat, xr))

exact_K_eig = np.linalg.eigvals(Kmat)
K_eig_val = fs.unique_eig(np.array(K_eig_val))
print(f'Is all the eigenvalue found?:{exact_K_eig.size == K_eig_val.size}')

print(f'exact eigenvalues of K:\n{np.sort(exact_K_eig)}')
print(f'approx eigenvalues of K:\n{K_eig_val}')

eig_min = K_eig_val[0]
x_min, _ = fs.rayleigh_iterate(Kmat, shift=eig_min)
chladni_show.show_nodes(x_min)

# d3
K_eig_vec = []
for eig_v in K_eig_val:
    x, _ = fs.rayleigh_iterate(Kmat, shift=eig_v)
    K_eig_vec.append(x)

T = np.zeros_like(Kmat)
for i in range(15):
    T[:, i] = K_eig_vec[i]

Lambda = np.diag(K_eig_val)

print(f'norm of K minus T@Lambda@inv(T):{np.linalg.norm(Kmat - T @ Lambda @ np.linalg.inv(T))}')

# d4
chladni_show.show_all_wavefunction_nodes(T, np.array(K_eig_val))
