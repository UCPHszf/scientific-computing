import numpy as np
import matplotlib.pyplot as plt

Amat = np.array([
    [22.13831203, 0.16279204, 0.02353879, 0.02507880, -0.02243145, -0.02951967, -0.02401863],
    [0.16279204, 29.41831006, 0.02191543, -0.06341569, 0.02192010, 0.03284020, 0.03014052],
    [0.02353879, 0.02191543, 1.60947260, -0.01788177, 0.07075279, 0.03659182, 0.06105488],
    [0.02507880, -0.06341569, -0.01788177, 9.36187184, -0.07751218, 0.00541094, -0.10660903],
    [-0.02243145, 0.02192010, 0.07075279, -0.07751218, 0.71033323, 0.10958126, 0.12061597],
    [-0.02951967, 0.03284020, 0.03659182, 0.00541094, 0.10958126, 8.38326265, 0.06673979],
    [-0.02401863, 0.03014052, 0.06105488, -0.10660903, 0.12061597, 0.06673979, 1.15733569]])

Bmat = np.array([
    [-0.03423002, 0.09822473, -0.00832308, -0.02524951, -0.00015116, 0.05321264, 0.01834117],
    [0.09822473, -0.51929354, -0.02050445, 0.10769768, -0.02394699, -0.04550922, -0.02907560],
    [-0.00832308, -0.02050445, -0.11285991, 0.04843759, -0.06732213, -0.08106876, -0.13042524],
    [-0.02524951, 0.10769768, 0.04843759, -0.10760461, 0.09008724, 0.05284246, 0.10728227],
    [-0.00015116, -0.02394699, -0.06732213, 0.09008724, -0.07596617, -0.02290627, -0.12421902],
    [0.05321264, -0.04550922, -0.08106876, 0.05284246, -0.02290627, -0.07399581, -0.07509467],
    [0.01834117, -0.02907560, -0.13042524, 0.10728227, -0.12421902, -0.07509467, -0.16777868]])

yvec = np.array([-0.05677315, -0.00902581, 0.16002152, 0.07001784, 0.67801388, -0.10904168, 0.90505180])

# Part a
print("Part a")


def maxNorm(mat):
    maxRowSum = float("-inf")
    for row in mat:
        rowSum = np.sum(np.abs(row))
        if rowSum > maxRowSum:
            maxRowSum = rowSum
    return maxRowSum


def maxNormConditionNumber(mat):
    mat_inv = np.linalg.inv(mat)
    matMaxNorm = maxNorm(mat)
    invMatMaxNorm = maxNorm(mat_inv)
    return matMaxNorm * invMatMaxNorm


E = np.vstack((np.hstack((Amat, Bmat)), np.hstack((Bmat, Amat))))
S = np.vstack((np.hstack((np.eye(7), np.zeros([7, 7]))), np.hstack((np.zeros([7, 7]), -np.eye(7)))))
z = np.hstack((yvec, -yvec))
omegas = [0.800, 1.146, 1.400]
for omega in omegas:
    conditionNumber = maxNormConditionNumber(E - omega * S)
    print('omega:' + str(omega) + ' condition number:' + str(conditionNumber))

# Part b
print()
print('Part b')
sigma_omega = 0.0005


def compute_relative_forward_error_bound(omega):
    return maxNormConditionNumber(E - omega * S) * (maxNorm(sigma_omega * S)) / maxNorm(E - omega * S)


# if an approximate value has a
# relative error of about 10^-p, then its decimal representation has about p correct
# signi cant digits
for omega in omegas:
    error_bound = compute_relative_forward_error_bound(omega)
    significantDigits = np.floor(-np.log10(error_bound))
    print('omega:' + str(omega) + ' ' + 'relative forward error:' + str(error_bound) + ' significant digits:' + str(
        significantDigits))

# Part c
print()
print("Part c")
testA = np.array([[2, 1, 1], [4, 1, 4], [-6, -5, 3]])
test_b = np.array([4, 11, 4])


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


# compare the result between my own implemented function and np.linalg.solve()
accurate_x = np.linalg.solve(testA, test_b)
print(accurate_x)
lu_solved_x = LU_solve(testA, test_b)
print(lu_solved_x)

# Part d
print()
print('Part d')


def solve_alpha(omega):
    x = LU_solve(E - omega * S, z)
    return np.dot(np.transpose(z), x)


for omega in omegas:
    omega0 = omega + sigma_omega
    omega1 = omega - sigma_omega
    print('omega:' + str(omega) + ' alpha:' + str(np.around(solve_alpha(omega), 3)))
    print('omega:' + str(omega) + '+' + str(sigma_omega) + ' alpha:' + str(np.around(solve_alpha(omega0), 3)))
    print('omega:' + str(omega) + '-' + str(sigma_omega) + ' alpha:' + str(np.around(solve_alpha(omega1), 3)))

# Part e
print()
print("Part e")
intervals = np.linspace(0.7, 1.5, 1000)

alphas = []
for omega in intervals:
    alphas.append(solve_alpha(omega))
plt.plot(intervals, alphas)
plt.xlabel('omega')
plt.ylabel('alpha')
plt.savefig('e.jpg')
plt.show()

specified_omega = 1.146307999
# print the condition number, alpha(omega) when omega=1.146307999.
print(np.around(maxNormConditionNumber(E - specified_omega * S), 3))
print(np.around(solve_alpha(specified_omega), 3))

# Part f
print()
print("Part f")
# test data in HHexamples.py
A0 = np.array([[1, 2], [3, 4]])
A1 = np.array([[1, 2], [3, 4], [5, 6]])
A2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

b0 = np.array([1, 2])
b1 = np.array([1, 2, 3])
b2 = np.array([1, 2, 3, 4])

x0 = np.array([-0., 0.5])
x1 = np.array([-0., 0.5])
x2 = np.array([-0.33333333, 0.66666667, 0.])


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


test_Q1, test_R1 = qr_decomposition(A1)
print("R1:\n" + str(np.round(test_R1, 3)))


# print("Multiplication of Q and its transpose:\n" + str(np.around(test_Q1.T @ test_Q1, 3)))
# print("Multiplication of Q and R:\n" + str(test_Q1 @ test_R1))


def least_squares(A, b):
    m, n = A.shape
    Q, R = qr_decomposition(A)
    y = Q.T @ b
    x = back_substitute(R[:n], y[:n])
    return x


test_x1 = least_squares(A1, b1)
print("test x1:" + str(np.round(test_x1, 3)))
# print("Multiplication of A2 and test_x2 solved by least squares:\n" + str(np.round(A1 @ test_x1, 3)))

# Part g
print("")
print("Part g")


def P_polynomial(omega_range, alpha, n):
    m = len(omega_range)
    A = np.zeros((m, n + 1))
    for j in range(n + 1):
        A[:, j] = omega_range ** (2 * j)
    params = least_squares(A, alpha)
    P = np.zeros(m)
    for j in range(n + 1):
        P = P + params[j] * omega_range ** (2 * j)
    return params, P


omega_p = 1.1
omega_range = np.linspace(0.7, omega_p, 1000)
m = len(omega_range)
alpha = np.zeros(m)
for i in range(m):
    alpha[i] = solve_alpha(omega_range[i])
params_p4, P_4 = P_polynomial(omega_range, alpha, 4)
params_p6, P_6 = P_polynomial(omega_range, alpha, 6)

rel_p4 = np.abs((P_4 - alpha) / alpha)
rel_p6 = np.abs((P_6 - alpha) / alpha)
# g(2)
print('coefficient when n=4 in polynomial P')
print(np.round(params_p4, 3))
# Plot relative error in logarithmic scale
# g(3)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
ax1.plot(omega_range, rel_p4, label='n=4')
ax1.plot(omega_range, rel_p6, label='n=6')
ax1.set_yscale('log')
ax1.set_xlabel(r'$\omega$')
ax1.set_ylabel(r'relative error')
ax1.legend()

# significant digits = -log10(relative_error), by using np.floor() we got the integer
# g(4)
ax2.plot(omega_range, np.floor(-np.log10(rel_p4)), label='n=4')
ax2.plot(omega_range, np.floor(-np.log10(rel_p6)), label='n=6')
ax2.legend()
ax2.set_xlabel(r'$\omega$')
ax2.set_ylabel(r'Number of significant digits')
fig.tight_layout()
fig.savefig('g.jpg')

# Part h
# h(1)
print()
print("Part h")
omega_range = np.linspace(0.7, 1.5, 1000)
m = len(omega_range)
alpha = np.zeros(m)
for i in range(m):
    alpha[i] = solve_alpha(omega_range[i])


def Q_polynomial(omega_range, alpha, n):
    m = omega_range.size
    N = 2 * n + 1
    param_b_startIdx = n + 1
    A = np.zeros((m, N))
    for j in range(n + 1):
        A[:, j] = omega_range ** j
    for j in range(1, n + 1):
        A[:, j + param_b_startIdx - 1] = -alpha * omega_range ** j
    params = least_squares(A, alpha)
    Q = Calculate_Q_polynomial(omega_range, params)
    return params, Q


def Calculate_Q_polynomial(omega_range, params):
    param_length = len(params)
    splitIdx = int((param_length - 1) / 2) + 1
    a = params[:splitIdx]
    b = np.array([0, *params[splitIdx:]])
    numerator = np.zeros(omega_range.shape)
    denominator = np.zeros(omega_range.shape)
    for i in range(splitIdx):
        numerator = numerator + a[i] * omega_range ** i
        denominator = denominator + b[i] * omega_range ** i
    result = numerator / (1 + denominator)
    return result


# h(2)
params_q2, Q_2 = Q_polynomial(omega_range, alpha, 2)
params_q4, Q_4 = Q_polynomial(omega_range, alpha, 4)
aj = params_q2[:3]
bj = params_q4[3:]
print("coefficient when n=2 in polynomial Q")
print("a_j:")
print(np.round(aj, 3))
print("b_j")
print(np.round(bj, 3))
rel_q2 = np.abs((Q_2 - alpha) / alpha)
rel_q4 = np.abs((Q_4 - alpha) / alpha)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
ax1.plot(omega_range, rel_q2, label='n=2')
ax1.plot(omega_range, rel_q4, label='n=4')
ax1.set_yscale('log')
ax1.set_xlabel(r'$\omega$')
ax1.set_ylabel(
    r'relative error')
ax1.legend()

ax2.plot(omega_range, np.floor(-np.log10(rel_q2)), label='n=2')
ax2.plot(omega_range, np.floor(-np.log10(rel_q4)), label='n=4')
ax2.legend()
ax2.set_xlabel(r'$\omega$')
ax2.set_ylabel(r'significant digits')
fig.tight_layout()
fig.savefig('h.jpg')
