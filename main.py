import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.linalg import solve

# --- Problem Parameters ---
a, b = 0, 1
alpha, beta = 1, 2
h = 0.1
l = b - a  # For variation approach, length of the domain

# --- Common x-points for output consistency (for h=0.1) ---
x_output_points = np.arange(a, b + h, h)

print("--- HW11: Boundary-value problem for O.D.E. ---")
print(f"Given equation: y'' = -(x+1)y' + 2y + (1-x^2)e^(-x), {a} <= x <= {b}, y({a}) = {alpha}, y({b}) = {beta}, h = {h}")

# Define p(x), q(x), r(x)
def p_orig(x): return -(x + 1)
def q_orig(x): return 2
def r_orig(x): return (1 - x**2) * np.exp(-x)

# --- a. Shooting Method ---
print("\n--- a. Shooting Method ---")
def ode_y1(x, Y):
    y1, y1p = Y
    return [y1p, p_orig(x) * y1p + q_orig(x) * y1 + r_orig(x)]

def ode_y2(x, Y):
    y2, y2p = Y
    return [y2p, p_orig(x) * y2p + q_orig(x) * y2]

sol_y1 = solve_ivp(ode_y1, [a, b], [alpha, 0], t_eval=x_output_points, rtol=1e-6, atol=1e-6)
sol_y2 = solve_ivp(ode_y2, [a, b], [0, 1], t_eval=x_output_points, rtol=1e-6, atol=1e-6)

y1_vals = sol_y1.y[0]
y2_vals = sol_y2.y[0]

if y2_vals[-1] == 0:
    print("Warning: y2(b) = 0, C undefined.")
    C_shoot = np.nan
else:
    C_shoot = (beta - y1_vals[-1]) / y2_vals[-1]

y_shooting = y1_vals + C_shoot * y2_vals

print(f"  C = {C_shoot}")
print(f"  y values: {y_shooting}")

# --- b. Finite-Difference Method ---
print("\n--- b. Finite-Difference Method ---")
n_fd = int((b - a) / h) - 1
x_points_fd_internal = np.linspace(a, b, n_fd + 2)

A_fd = np.zeros((n_fd, n_fd))
F_fd = np.zeros(n_fd)

for i in range(n_fd):
    xi = x_points_fd_internal[i + 1]
    pi = p_orig(xi)
    qi = q_orig(xi)
    ri = r_orig(xi)

    A_fd[i, i] = 2 + h**2 * qi
    if i > 0:
        A_fd[i, i - 1] = -(1 + 0.5 * h * pi)
    if i < n_fd - 1:
        A_fd[i, i + 1] = -(1 - 0.5 * h * pi)

    F_fd[i] = -h**2 * ri

    if i == 0:
        F_fd[i] += (1 + 0.5 * h * p_orig(x_points_fd_internal[1])) * alpha
    if i == n_fd - 1:
        F_fd[i] += (1 - 0.5 * h * p_orig(x_points_fd_internal[n_fd])) * beta

y_interior_fd = solve(A_fd, F_fd)
y_finite_difference = np.insert(y_interior_fd, 0, alpha)
y_finite_difference = np.append(y_finite_difference, beta)

print(f"  y values: {y_finite_difference}")

# --- c. Variation Approach (Galerkin Method with sine basis) ---
print("\n--- c. Variation Approach (Galerkin Method) ---")

def y1_var(x):
    return alpha * (l - x) / l + beta * x / l

def y1_prime_var(x):
    return -alpha / l + beta / l

def y1_double_prime_var(x):
    return 0.0

def F_var(x):
    return r_orig(x) + p_orig(x) * y1_prime_var(x) + q_orig(x) * y1_var(x)

def phi_var(i, x):
    return np.sin(i * np.pi * x / l)

def phi_prime_var(i, x):
    return (i * np.pi / l) * np.cos(i * np.pi * x / l)

def phi_double_prime_var(i, x):
    return -(i * np.pi / l)**2 * np.sin(i * np.pi * x / l)

N_basis = 5
A_var = np.zeros((N_basis, N_basis))
b_vec_var = np.zeros(N_basis)

for i in range(1, N_basis + 1):
    for j in range(1, N_basis + 1):
        term1, _ = quad(lambda x: phi_double_prime_var(j, x) * phi_var(i, x), a, b)
        term2, _ = quad(lambda x: -p_orig(x) * phi_prime_var(j, x) * phi_var(i, x), a, b)
        term3, _ = quad(lambda x: -q_orig(x) * phi_var(j, x) * phi_var(i, x), a, b)
        A_var[i - 1, j - 1] = term1 + term2 + term3
    b_vec_var[i - 1], _ = quad(lambda x: F_var(x) * phi_var(i, x), a, b)

c_coeffs_var = solve(A_var, b_vec_var)

def y2_approx_var(x_vals_arr, c_coeffs_arr):
    y2_vals = np.zeros_like(x_vals_arr)
    for k in range(len(c_coeffs_arr)):
        y2_vals += c_coeffs_arr[k] * phi_var(k + 1, x_vals_arr)
    return y2_vals

y_variation_approach = y1_var(x_output_points) + y2_approx_var(x_output_points, c_coeffs_var)

print(f"  Coefficients c: {c_coeffs_var}")
print(f"  y values: {y_variation_approach}")

# --- Summary Table ---
print("\n--- Summary Table ---")
print(f"{'x':>4} {'Shooting':>12} {'FD':>12} {'Variation':>12}")
for i in range(len(x_output_points)):
    print(f"{x_output_points[i]:4.1f} {y_shooting[i]:12.6f} {y_finite_difference[i]:12.6f} {y_variation_approach[i]:12.6f}")
