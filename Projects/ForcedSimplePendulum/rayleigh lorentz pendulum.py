import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from scipy.integrate import RK45

save_loc = "C:\\Repositories\\PHYS-6017-Labs\\Projects\\ForcedSimplePendulum\\Plots"

# SOLVING THE MATRIX EQUATION Y' = AY + b
def pendulum_solver(m, L0, k, g, F, eta, time_range, initial_angle, initial_ang_vel, max_step):
    # parameters:
    def alphas(k, m, g, L):
        return (k / (m * L * np.sqrt(g * L)))

    def betas(F, m, g):
        return (F / (m * g))

    # Random walk for pendulum length
    def pendulum_length_random_walk(L0, sigma, num_steps):
        L_values = [L0]
        for _ in range(num_steps):
            L_values.append(max(0.1, L_values[-1] + np.random.normal(scale=sigma)))
        return np.array(L_values)

    # Forcing function with changing frequency due to changing length
    def bvector(t, L_values):
        return np.array([0, beta * np.cos((1 - eta) * t) / np.sqrt(L_values[int(t)])])

    # solving with RK45 module:
    def solve_rk45(bvector, y0, time_range, max_step=0.1):
        def ode_system(t, y, L_values):
            A = np.array([[0, 1], [-np.sin(np.pi / 2), -alphas(k, m, g, L_values[int(t)])]])
            return A @ y + bvector(t, L_values)

        ode_solver = RK45(fun=lambda t, y: ode_system(t, y, L_values), t0=time_range[0],
                          y0=y0, t_bound=time_range[1], max_step=max_step)

        # Lists to store the results
        t_values = [ode_solver.t]
        y_values = [ode_solver.y]

        while ode_solver.status == 'running':
            ode_solver.step()
            t_values.append(ode_solver.t)
            y_values.append(ode_solver.y)

        return np.array(t_values), np.array(y_values)

    # for a pendulum starting from rest:
    y0 = np.array([initial_angle, initial_ang_vel])  # [initial displacement angle, initial angular velocity]
    beta = betas(F, m, g)

    # Generate pendulum length random walk
    num_steps = int((time_range[1] - time_range[0]) / max_step)
    L_values = pendulum_length_random_walk(L0, sigma=0.01, num_steps=num_steps)

    t_values, theta_values = solve_rk45(bvector, y0, time_range, max_step=0.05)

    return t_values, theta_values, L_values

# Example usage
m = 1.0  # mass
L0 = 1.0  # initial length
k = 1.0  # spring constant
g = 9.81  # gravity
F = 0.5  # driving force amplitude
eta = 0.1  # damping coefficient
time_range = (0, 100)  # time range
initial_angle = np.pi / 4  # initial displacement angle
initial_ang_vel = 0.0  # initial angular velocity

t_values, theta_values, L_values = pendulum_solver(m, L0, k, g, F, eta, time_range, initial_angle, initial_ang_vel, max_step=5)

plt.plot(t_values, theta_values[:, 0])
plt.show

plt.plot(theta_values[:, 0], theta_values[:, 1])
plt.show()

plt.plot(t_values, L_values)
plt.show()


