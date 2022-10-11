import math

import LJhelperfunctions as lj
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Part a
print("Part a")
V_function = lj.LJ()


def two_particles(x):
    x0 = np.array([x, 0, 0])
    x1 = np.array([0, 0, 0])
    points = np.stack((x0, x1))
    pot = V_function(points)
    return pot


def four_particles(x):
    x0 = np.array([x, 0, 0])
    x1 = np.array([0, 0, 0])
    x2 = np.array([14, 0, 0])
    x3 = np.array([7, 3.2, 0])
    points = np.stack((x0, x1, x2, x3))
    pot = V_function(points)
    return pot


x_range = np.linspace(3, 11, 100)
y_2 = []
y_4 = []
# Calculate potential for each x
for x in x_range:
    y_2.append(two_particles(x))
    y_4.append(four_particles(x))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_range, y_2, label='Two particles')
ax.plot(x_range, y_4, label='Four particles')
ax.set_xlabel('x')
ax.set_ylabel(r'$V_{LJ}$')
ax.set_title('Potential')
ax.legend()
plt.savefig('a.jpg')
plt.show()

# Part b
print()
print("Part b")


def bisection_root(f, a, b, tolerance=1e-13):
    a = min(a, b)
    b = max(a, b)
    f_a = f(a)
    f_b = f(b)
    n_calls = 2
    if f_a * f_b > 0:
        return None
    while b - a > tolerance:
        middle = (a + b) / 2
        f_m = f(middle)
        if f_a * f_m > 0:
            a = middle
        else:
            b = middle
        n_calls += 1
    return b, n_calls


tolerance = 1e-13
x, k = bisection_root(two_particles, 2, 6, tolerance)
print(f'Found x:{x},number of calls to the bisection function needed:{k}')
print(f'result check:{x - lj.SIGMA < tolerance}')

# Part c
print()
print("Part c")


def pair_potential_derivative(r):
    return 4 * lj.EPSILON * ((6 * lj.SIGMA ** 6) / r ** 7 - (12 * lj.SIGMA ** 12) / r ** 13)


def newton_root(f, df, x0, tolerance, max_iterations=50):
    x, n_calls = x0, 0
    while n_calls < max_iterations:
        f_x = f(x)
        n_calls += 1
        if abs(f_x) < tolerance:
            return x, n_calls
        x = x - f_x / df(x)
        n_calls += 1


tolerance = 1e-12
x, k = newton_root(two_particles, pair_potential_derivative, 2, tolerance)
print(f'Found x:{x},number of calls to the newton function needed:{k}')
print(f'result check:{x - lj.SIGMA < tolerance}')

# Part d
print()
print('Part d')


def newton_bisection_hybrid(f, df, a, b, tolerance):
    a = min(a, b)
    b = max(a, b)
    f_a = f(a)
    n_calls = 1
    while True:
        m = (a + b) / 2
        c = m - f(m) / df(m)
        if a < c < b:
            x = c
        else:
            x = (a + b) / 2
        n_calls += 2
        f_x = f(x)
        if abs(f_x) < tolerance:
            n_calls += 1
            return x, n_calls
        if f_a * f_x < 0:
            b = x
        else:
            a = x
        n_calls += 1


tolerance = 1e-13
x, k = newton_bisection_hybrid(two_particles, pair_potential_derivative, 2, 6, tolerance)
print(f'Found x:{x},number of calls to the energy function needed:{k}')
print(f'result check:{x - lj.SIGMA < tolerance}')

# Part e
print()
print('Part e')
gradV_function = lj.gradV


def grad_two_particles(x):
    x0 = np.array([x, 0, 0])
    x1 = np.array([0, 0, 0])
    points = np.stack((x0, x1))
    gradV = gradV_function(points)
    return gradV


x_range = np.linspace(3, 10, 100)
y_2 = []
grads = []
print("gradiant 2-particle system when x = 5:")
print(grad_two_particles(5))
for x in x_range:
    grad = grad_two_particles(x)
    grads.append(grad[0][0])
    y_2.append(two_particles(x))

fig, ax = plt.subplots(figsize=(8, 6))
# Gradient
ax.plot(x_range, grads, '-', label='Gradient')
# Potential
ax.plot(x_range, y_2, '-', label='LJ Potential')
ax.set_title('Two Particle System', fontsize=14)
ax.set_xlabel('nonzero component for the derivative of the x-coordinate of x0')
ax.hlines(0, 3, 10, linestyle='dashed')
ax.legend()
ax.set_ylim(-10, 10)
plt.savefig('e.jpg')
plt.show()

# Part f
print()
print('Part f')


def line_segment_derivative(F, X0, d):
    def directional_derivative(alpha):
        return d @ F(X0 + alpha * d)

    return directional_derivative


def line_search_bisection(F, a, b, tolerance):
    a = min(a, b)
    b = max(a, b)
    f_a = F(a)
    f_b = F(b)
    if f_a * f_b > 0:
        return None, None
    n_calls = 2
    while abs(b - a) > tolerance:
        m = (a + b) / 2
        f_m = F(m)
        n_calls += 1
        if f_a * f_m < 0:
            b = m
        else:
            a = m
    return b, n_calls


def linesearch(F, X0, d, alpha_max, alpha_min=0, tolerance=1e-12, max_iterations=100):
    x0_flatten = X0.flatten()
    d_flatten = d.flatten()
    f = line_segment_derivative(F, x0_flatten, d_flatten)
    alpha, n_calls = line_search_bisection(f, alpha_min, alpha_max, tolerance)
    return alpha, n_calls


x0 = np.array([[4, 0, 0], [0, 0, 0], [14, 0, 0], [7, 3.2, 0]])
d = -gradV_function(x0)
f = lj.flat_gradV
alpha, k = linesearch(f, x0, d, 1)
print(f'Found alpha:{alpha},number of calls:{k}')

# Part g
print()
print('Part g')


def golden_section_min(f, a, b, tolerance=1e-3):
    tau = (np.sqrt(5) - 1) / 2
    x1 = a + (1 - tau) * (b - a)
    f1 = f(x1)
    x2 = a + tau * (b - a)
    f2 = f(x2)
    n_calls = 2
    while abs(b - a) > tolerance:
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + tau * (b - a)
            f2 = f(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (1 - tau) * (b - a)
            f1 = f(x1)
        n_calls += 1
    return b, n_calls


def line_segment(f, X0, d):
    def P(alpha):
        line = X0 + alpha * d
        return f(line)

    return P


f = line_segment(lj.flat_V, x0.flatten(), d.flatten())
alp, calls = golden_section_min(f, 0, 1)
print(f'alpha: {alp},function calls:{calls} calls.')

x, k = golden_section_min(two_particles, 2, 6)
print(f'optimal (minimal-energy) distance r0 between two Ar atoms'
      f':{x},number of calls :{k}')

# Part h
print()
print("Part h")


def BFGS_direct(f, gradf, X, tolerance=1e-6, max_iterations=10000):
    n = len(X)
    B = np.eye(n)
    x = X
    grad = gradf(x)
    n_calls = 1
    converged = True
    while np.linalg.norm(grad) > tolerance and n_calls < max_iterations:
        s = -np.dot(B, grad)
        x_new = x + s
        grad_new = gradf(x_new)
        y = grad_new - grad
        n_calls += 1
        rho = 1 / (y @ s)
        B = (np.identity(n) - rho * np.outer(s, y)) @ B @ (
                np.identity(n) - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x = x_new
        grad = grad_new
    if n_calls >= max_iterations:
        converged = False
    return x, n_calls, converged


ArStart = np.load("ArStart.npz")
Xstart2 = ArStart["Xstart2"]
x, n_calls, converged = BFGS_direct(lj.flat_V, lj.flat_gradV, Xstart2)
print(f"Number of calls to the functions:{n_calls},Converged:{converged}")

points = x.reshape(2, 3)
distance = lj.distance(points)
print(f"minimum of the two-particle system: {distance[0, 1]}")

# Part i
print()
print("Part i")
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 8), subplot_kw=dict(projection='3d'))
r0 = distance[0, 1]
file_names = ArStart.files[:-1]
print(file_names)
for i in range(len(file_names)):
    file_name = file_names[i]
    Xstart = ArStart[file_name]
    x, n_calls, converged = BFGS_direct(lj.flat_V, lj.flat_gradV, Xstart)
    if i == len(file_names) - 1:
        N = 20
    else:
        N = i + 2
    points = x.reshape(N, -3)
    print(f'\nFor {N} particles the functions is called {n_calls} times. Converged: {converged}')
    if converged:
        distance = lj.distance(points)
        cnt = np.sum(abs(distance - r0) / r0 <= 0.01)
        if N == 3:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r')
            ax.set_title(f'{N} Particle System')
plt.savefig('i.jpg')
plt.show()

# Part j
print()
print("Part j")


def BFGS(f, gradf, X, tolerance=1e-6, max_iterations=10000):
    d = len(X)
    B = np.eye(d)
    gradient = gradf(X)
    H = np.linalg.inv(B)
    converged = True
    n_calls = 1
    while n_calls < max_iterations:
        if np.linalg.norm(gradient) < tolerance:
            break
        p = -1 * np.dot(H, gradient)
        f1d = line_segment(f, X, p)
        alpha, extra_calls = golden_section_min(f1d, -1, 1)
        n_calls += extra_calls
        s = alpha * p
        X_new = X + s
        gradient_new = gradf(X_new)
        y = gradient_new - gradient
        if np.dot(s, y) >= 0:
            rho = 1 / (y @ s)
            H = (np.identity(d) - rho * np.outer(s, y)) @ H @ (
                    np.identity(d) - rho * np.outer(y, s)) + rho * np.outer(s, s)
        n_calls += 1
        X = X_new
        gradient = gradient_new
    if n_calls > max_iterations:
        converged = False
    return x, n_calls, converged


for i in range(len(file_names)):
    file_name = file_names[i]
    Xstart = ArStart[file_name]
    x, n_calls, converged = BFGS(lj.flat_V, lj.flat_gradV, Xstart)
    if i == len(file_names) - 1:
        N = 20
    else:
        N = i + 2
    print(f'\nFor {N} particles the functions is called {n_calls} times. Converged: {converged}')
