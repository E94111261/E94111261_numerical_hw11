import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.linalg import solve

# --- Problem Parameters ---
a, b = 0, 1
alpha, beta = 1, 2
h = 0.1
l = b - a # For variation approach, length of the domain

# --- Common x-points for output consistency (for h=0.1) ---
x_output_points = np.arange(a, b + h, h)

print("--- HW11: Boundary-value problem for O.D.E. ---")
print(f"Given equation: y'' = -(x+1)y' + 2y + (1-x^2)e^(-x), {a} <= x <= {b}, y({a}) = {alpha}, y({b}) = {beta}, h = {h}")

# Define p(x), q(x), r(x) from the original ODE y'' = p(x)y' + q(x)y + r(x)
def p_orig(x):
    return -(x + 1)

def q_orig(x):
    return 2

def r_orig(x):
    return (1 - x**2) * np.exp(-x)

# --- a. Shooting Method (最符合講義的雙 IVP 求解法) ---
print("\n--- a. Shooting Method ---")

# Define the ODE for y1(x) [cite: 4]
# y1'' = p(x)y1' + q(x)y1 + r(x)
def ode_y1(x, Y):
    y1, y1p = Y
    return [y1p, p_orig(x) * y1p + q_orig(x) * y1 + r_orig(x)]

# Define the ODE for y2(x) [cite: 4]
# y2'' = p(x)y2' + q(x)y2
def ode_y2(x, Y):
    y2, y2p = Y
    return [y2p, p_orig(x) * y2p + q_orig(x) * y2]

# Solve the first IVP: y1(a)=alpha, y1'(a)=0 [cite: 4]
sol_y1 = solve_ivp(ode_y1, [a, b], [alpha, 0], t_eval=x_output_points, rtol=1e-6, atol=1e-6)
y1_vals = sol_y1.y[0]

# Solve the second IVP: y2(a)=0, y2'(a)=1 [cite: 4]
sol_y2 = solve_ivp(ode_y2, [a, b], [0, 1], t_eval=x_output_points, rtol=1e-6, atol=1e-6)
y2_vals = sol_y2.y[0]

# Calculate the constant C [cite: 6]
if y2_vals[-1] == 0:
    print("Warning: y2(b) is zero, cannot determine C unique. Consider adjusting y2'(a).")
    C_shoot = np.nan # Indicate an issue
else:
    C_shoot = (beta - y1_vals[-1]) / y2_vals[-1] # [cite: 6]

# The solution y(x) = y1(x) + C * y2(x) [cite: 5]
y_shooting = y1_vals + C_shoot * y2_vals

print(f"  C = {C_shoot}")
print(f"  x values: {x_output_points}")
print(f"  y values: {y_shooting}")


# --- b. Finite-Difference Method ---
print("\n--- b. Finite-Difference Method ---")

# Number of interior points (n in the notes, for y_1 to y_n)
n_fd = int((b - a) / h) - 1

# Create x_i points (x_0, x_1, ..., x_n, x_{n+1})
x_points_fd_internal = np.linspace(a, b, n_fd + 2) # This includes x_0 and x_{n+1}

# Initialize the matrix A and vector F [cite: 1]
A_fd = np.zeros((n_fd, n_fd))
F_fd = np.zeros(n_fd)

# Populate the matrix A and vector F based on the formula [cite: 1]
# -(1 + 0.5*h*p_i)y_{i-1} + (2 + h^2*q_i)y_i - (1 - 0.5*h*p_i)y_{i+1} = -h^2*r_i [cite: 1]
for i in range(n_fd): # i refers to the index in the A matrix (0 to n_fd-1), corresponding to y_1 to y_n
    xi = x_points_fd_internal[i + 1] # Corresponding x-value for y_{i+1} in the notes (x_1 to x_n)
    pi = p_orig(xi)
    qi = q_orig(xi)
    ri = r_orig(xi)

    A_fd[i, i] = (2 + h**2 * qi) # Coefficient for y_i [cite: 1]

    if i > 0: # If not the first equation (i.e., not for y_1)
        A_fd[i, i-1] = -(1 + 0.5 * h * pi) # Coefficient for y_{i-1} [cite: 1]
    
    if i < n_fd - 1: # If not the last equation (i.e., not for y_n)
        A_fd[i, i+1] = -(1 - 0.5 * h * pi) # Coefficient for y_{i+1} [cite: 1]

    F_fd[i] = -h**2 * ri # Right-hand side term [cite: 1]

    # Handle boundary conditions (y_0 and y_{n+1} terms move to F) [cite: 1]
    if i == 0: # For the first equation (for y_1), y_0 is involved
        F_fd[i] += (1 + 0.5 * h * p_orig(x_points_fd_internal[1])) * alpha # Term from y_0 [cite: 1]
    
    if i == n_fd - 1: # For the last equation (for y_n), y_{n+1} is involved
        F_fd[i] += (1 - 0.5 * h * p_orig(x_points_fd_internal[n_fd])) * beta # Term from y_{n+1} [cite: 1]

# Solve the linear system A*Y = F [cite: 1]
y_interior_fd = solve(A_fd, F_fd)

# Combine with boundary conditions to get the full solution array
y_finite_difference = np.insert(y_interior_fd, 0, alpha)
y_finite_difference = np.append(y_finite_difference, beta)

print(f"  x values: {x_points_fd_internal}")
print(f"  y values: {y_finite_difference}")


# --- c. Variation Approach (using Galerkin Method with sine basis functions) ---
print("\n--- c. Variation Approach (Galerkin Method) ---")

# First, define y1(x) for transforming boundary conditions to homogeneous ones [cite: 10]
# y1(x) = a(l-x/l) + b(x/l) [cite: 10]
def y1_var(x):
    return alpha * (l - x) / l + beta * x / l

# The ODE for y2(x) where y(x) = y1(x) + y2(x) [cite: 10]
# Original ODE: y'' - p(x)y' - q(x)y = r(x)
# Substitute y = y1 + y2:
# (y1'' + y2'') - p(x)(y1' + y2') - q(x)(y1 + y2) = r(x)
# y2'' - p(x)y2' - q(x)y2 = r(x) + p(x)y1' + q(x)y1 - y1''
# Let F(x) = r(x) + p(x)y1' + q(x)y1 - y1'' (This is F(x) from the notes, not G(x) used previously) [cite: 10]
def y1_prime_var(x): # Derivative of y1(x)
    return -alpha / l + beta / l

def y1_double_prime_var(x): # Second derivative of y1(x)
    return 0.0 # Since y1(x) is linear

def F_var(x): # This is the F(x) in the notes, used for y2's ODE [cite: 10]
    return r_orig(x) + p_orig(x) * y1_prime_var(x) + q_orig(x) * y1_var(x) - y1_double_prime_var(x)

# Define basis functions phi_i(x) = sin(i*pi*x/l) with phi_i(0)=phi_i(l)=0 [cite: 9, 11, 13]
# Here l=1, so phi_i(x) = sin(i*pi*x)
def phi_var(i, x):
    return np.sin(i * np.pi * x / l)

def phi_prime_var(i, x):
    return (i * np.pi / l) * np.cos(i * np.pi * x / l)

def phi_double_prime_var(i, x):
    return -(i * np.pi / l)**2 * np.sin(i * np.pi * x / l)

# Number of basis functions to use [cite: 9, 11]
N_basis = 5 

# Initialize A matrix and b vector for the system [A]{c} = {b} [cite: 9, 11]
# For the Galerkin method (equivalent to variational approach for this ODE form),
# we test the residual with phi_i:
# integral [ y2'' - p(x)y2' - q(x)y2 - F(x) ] phi_i(x) dx = 0
# integral [ sum(c_j * (phi_j'' - p(x)phi_j' - q(x)phi_j)) - F(x) ] phi_i(x) dx = 0
# sum(c_j * integral [ (phi_j'' - p(x)phi_j' - q(x)phi_j) phi_i(x) ] dx) = integral [ F(x)phi_i(x) ] dx
# So, A_ij = integral [ (phi_j'' - p(x)phi_j' - q(x)phi_j) phi_i(x) ] dx
# and b_i = integral [ F(x)phi_i(x) ] dx

A_var = np.zeros((N_basis, N_basis))
b_vec_var = np.zeros(N_basis)

# Populate A and b_vec using numerical integration
for i in range(1, N_basis + 1): # Corresponds to phi_1 to phi_N_basis
    for j in range(1, N_basis + 1): # Corresponds to phi_1 to phi_N_basis
        # Integrand for A_ij
        integrand_A_term1 = lambda x: phi_double_prime_var(j, x) * phi_var(i, x)
        integrand_A_term2 = lambda x: -p_orig(x) * phi_prime_var(j, x) * phi_var(i, x)
        integrand_A_term3 = lambda x: -q_orig(x) * phi_var(j, x) * phi_var(i, x)
        
        val_term1, _ = quad(integrand_A_term1, a, b)
        val_term2, _ = quad(integrand_A_term2, a, b)
        val_term3, _ = quad(integrand_A_term3, a, b)
        
        A_var[i-1, j-1] = val_term1 + val_term2 + val_term3
    
    # Integrand for b_i [cite: 12]
    integrand_b = lambda x: F_var(x) * phi_var(i, x)
    val_b, _ = quad(integrand_b, a, b)
    b_vec_var[i-1] = val_b

# Solve for coefficients c [cite: 9, 11]
c_coeffs_var = solve(A_var, b_vec_var)

# Reconstruct y2(x) = sum(c_i * phi_i(x)) [cite: 11]
def y2_approx_var(x_vals_arr, c_coeffs_arr):
    y2_vals = np.zeros_like(x_vals_arr, dtype=float)
    for k in range(len(c_coeffs_arr)):
        y2_vals += c_coeffs_arr[k] * phi_var(k + 1, x_vals_arr)
    return y2_vals

# Reconstruct the full solution y(x) = y1(x) + y2(x) [cite: 10]
y_variation_approach = y1_var(x_output_points) + y2_approx_var(x_output_points, c_coeffs_var)

print(f"  Coefficients c: {c_coeffs_var}")
print(f"  x values: {x_output_points}")
print(f"  y values: {y_variation_approach}")