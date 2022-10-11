import numpy as np


def maxNorm(mat):
    maxRowSum = float("-inf")
    for row in mat:
        rowSum = np.sum(np.abs(row))
        if rowSum > maxRowSum:
            maxRowSum = rowSum
    return maxRowSum


def LU_decomposition(M):
    n = len(M[0])
    L = np.zeros([n, n])
    U = np.zeros([n, n])
    for i in range(n):
        L[i][i] = 1
        if i == 0:
            U[0][0] = M[0][0]
            for j in range(1, n):
                U[0][j] = M[0][j]
                L[j][0] = M[j][0] / U[0][0]
        else:
            for j in range(i, n):  # U
                temp = 0
                for k in range(0, i):
                    temp = temp + L[i][k] * U[k][j]
                U[i][j] = M[i][j] - temp
            for j in range(i + 1, n):  # L
                temp = 0
                for k in range(0, i):
                    temp = temp + L[j][k] * U[k][i]
                L[j][i] = (M[j][i] - temp) / U[i][i]
    return L, U


def forward_substitute(L, b):
    n = len(L[0])
    x = np.zeros(n)
    for i in range(n):
        if L[i][i] != 0:
            x[i] = (b[i] - np.dot(L[i][0:i + 1], x[0:i + 1])) / L[i][i]
    return x


def back_substitute(U, y):
    n = len(U[0])
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i][i] != 0:
            x[i] = (y[i] - np.dot(U[i][i + 1:], x[i + 1:])) / U[i][i]
    return x


def LU_solve(A, b):
    L, U = LU_decomposition(A)
    y = forward_substitute(L, b)
    x = back_substitute(U, y)
    return x


def qr_decomposition(A):
    m, n = A.shape
    t = min(m, n)
    Q = np.eye(m)
    R = A.copy()
    for k in range(t - (m == n)):
        x = R[k:, [k]]
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        alpha = np.linalg.norm(x)
        # construct vector v for Householder reflection
        v = x + np.sign(x[0]) * alpha * e1
        v /= np.linalg.norm(v)
        # construct the Householder matrix
        Q_k = np.eye(m - k) - 2.0 * v @ v.T
        Q_k = np.block([[np.eye(k), np.zeros((k, m - k))], [np.zeros((m - k, k)), Q_k]])
        Q = Q @ Q_k.T
        R = Q_k @ R
    return Q, R


def qr_solve(A, b):
    m, n = A.shape
    Q, R = qr_decomposition(A)
    y = Q.T @ b
    x = back_substitute(R[:n], y[:n])
    return x


def gershgorin(A):
    centers = np.diag(A)
    radii = np.empty(0)
    for i in range(A.shape[0]):
        radii = np.append(radii, np.sum(np.abs(A[i])) - np.abs(centers[i]))
    return centers, radii


def rayleigh_qt(A, x):
    return np.dot(x, np.dot(A, x)) / np.dot(x, x)


def power_iterate(A, x0=None, max_iter=30, epsilon=1e-6):
    m, _ = A.shape
    x = np.random.random(size=m) if x0 is None else x0
    iteration = 0
    while iteration <= max_iter:
        iteration += 1
        y = A @ x
        x = y / maxNorm(y)
        lambda_cur = rayleigh_qt(A, x)
        e = np.linalg.norm(A @ x - lambda_cur * x)
        if e <= epsilon:
            break
    return x, iteration


def inverse_iterate(A, x0, shift=0., epsilon=1e-6, max_iter=50):
    n, _ = A.shape
    B = A - np.eye(n) * shift
    x = np.copy(x0)
    lambda_last = rayleigh_qt(A, x)
    for i in range(max_iter):
        y = qr_solve(B, x)
        if y is None:
            return None, None
        x = y / maxNorm(y)
        lambda_new = rayleigh_qt(A, x)
        res = np.linalg.norm(lambda_new * x - lambda_last * x)
        if res <= epsilon:
            break
        lambda_last = lambda_new
    return x


def rayleigh_iterate(A, x0=None, shift=0., epsilon=1e-6, max_iter=50):
    m, _ = A.shape
    x = np.random.random(size=m) if x0 is None else x0
    if shift is not None:
        x = inverse_iterate(A, x, shift)
        if x is None:
            return None, None
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        y = qr_solve(A - shift * np.eye(m), x)
        x = y / maxNorm(y)
        shift = rayleigh_qt(A, x)
        e = np.linalg.norm(np.dot(A, x) - shift * x, np.inf)
        if e <= epsilon:
            break
    return x, iteration


def unique_eig(eigs):
    unique_eigs = []
    while len(eigs) > 0:
        close_set = np.isclose(eigs[0], eigs)
        mean_buddies = np.mean(eigs[close_set])
        unique_eigs.append(mean_buddies)
        eigs = eigs[~close_set]
    return np.sort(unique_eigs)
