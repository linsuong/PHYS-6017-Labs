import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from scipy.integrate import RK45
from scipy.interpolate import interp1d

save_loc = "C:\\Repositories\\PHYS-6017-Labs\\Projects\\ForcedSimplePendulum\\Plots"

# SOLVING THE MATRIX EQUATION Y' = AY + b
def pendulum_solver_random_walk(m, L0, k, g, F, sigma, time_range, initial_angle=0, initial_ang_vel=0, loc=0, max_step = 0.05):
    # parameters:
    def alphas(k, m, g, L):
        return (k / (m * L * np.sqrt(g * L)))

    def betas(F, m, g):
        return (F / (m * g))

    def random_walk(L0, num_steps, sigma=sigma):
        L_values = [L0]
        for _ in range(num_steps):
            s = np.random.default_rng().normal(loc, sigma)
            L_values.append(L_values[-1] + s)

        return np.array(L_values)

    def ode_system(t, y, A, bvector, L, alpha):
        return A @ y + bvector(t, alpha)

    # solving with RK45 module:
    def solve_rk45(bvector, y0, time_range, L_values, max_step=0.05):
        # Lists to store the results
        t_values = []
        y_values = []

        t = time_range[0]
        y = y0
        idx = 0  # Index for L_values

        while t < time_range[1]:
            # Update L value if provided
            if L_values is not None:
                L = L_values[idx]
                alpha = alphas(k, m, g, L)
                idx += 1
            else:
                print("using L0")
                alpha = alphas(k, m, g, L0)  # Use initial L value if L_values is not provided

            #print(L)
            A = np.array([[0, 1], [-np.sin(np.pi / 2), -alpha]])
            ode_solver = RK45(fun=lambda t, y: ode_system(t, y, A, bvector, L, alpha), t0=t,
                            y0=y, t_bound=min(t + max_step, time_range[1]), max_step=max_step)

            while ode_solver.status == 'running':
                ode_solver.step()
                t_values.append(ode_solver.t)
                y_values.append(ode_solver.y)
                t = ode_solver.t
                y = ode_solver.y

        return np.array(t_values), np.array(y_values), L_values

    num_steps = int((time_range[1] - time_range[0]) / max_step)
    L_values = random_walk(L0, num_steps=num_steps, sigma=0.01)
    y0 = np.array([initial_angle, initial_ang_vel])
    beta = betas(F, m, g)

    bvector = lambda t, alpha: np.array([0, beta])

    t_values, theta_values, L_values = solve_rk45(bvector, y0, time_range, L_values, max_step = 0.05)

    num_points_to_remove = len(t_values) - len(L_values)
    t_values = t_values[:-num_points_to_remove]
    theta_values= theta_values[:-num_points_to_remove, :]

    return t_values, theta_values, L_values

def pendulum_position(angle, length):
    x = length * np.cos(angle - (np.pi/2))
    y = length * np.sin(angle - (np.pi/2))
    
    return x, y

def set_pi_ticks(ax, data):
    ax_ticks = np.arange(min(data[:, 0]), max(data[:, 0] + np.pi), np.pi)
    ax.set_yticks(ax_ticks)
    ax.set_yticklabels([f"{i//np.pi}$\\pi$" for i in ax_ticks])
    
def set_pi_ticks_x(ax, data):
    ax_ticks = np.arange(min(data[:, 0]), max(data[:, 0] + np.pi), np.pi)
    ax.set_xticks(ax_ticks)
    ax.set_xticklabels([f"{i//np.pi}$\\pi$" for i in ax_ticks])

def update(frame):
    for ax in axs.flatten():
        ax.clear()
    
    axs[0, 0].set_xlim(0, interval[1])
    axs[0, 0].plot(t_values, theta_values[:, 0], color = 'blue', label='Angular Displacement, $\\theta$ ($rad s^{-1}$)')
    axs[0, 0].plot(t_values, theta_values[:, 1], color='green', label='Angular Velocity, $\\frac{d\\theta}{dt}$ ($rad s^{-1}$)')
    axs[0, 0].plot(t_values[frame], theta_values[frame, 1], 'ro')
    axs[0, 0].plot(t_values[frame], theta_values[frame, 0], 'ro')  # Red tracer ball    
    axs[0, 0].set_xlabel('Time, t, (s)')
    axs[0, 0].set_title("Angular Displacement vs Time (blue), Angular Velocity vs Time (green)")
    axs[0, 0].legend(loc = 'upper right')
    #set_pi_ticks(axs[0, 0], theta_values)

    axs[0, 1].clear()
    axs[0, 1].set_xlim(-1.2 * L, 1.2 * L)
    axs[0, 1].set_ylim(-1.2 * L, 1.2 * L)
    axs[0, 1].set_aspect('equal', adjustable = 'box')
    axs[0, 1].add_patch(plt.Circle((0, 0), L, color = 'black', fill=False))
    angle = theta_values[frame, 0]
    x, y = pendulum_position(angle, L_values[frame])
    axs[0, 1].plot([0, x], [0, y], color='blue', linewidth = 2)
    axs[0, 1].plot(x, y, 'o', color='red', markersize = 10)
    axs[0, 1].text(1.15, 0.08, f'Frame: {frame} of {len(theta_values[:, 0])}', transform=axs[0, 1].transAxes, color='black')
    axs[0, 1].axis('off')
    axs[0, 1].set_title("Pendulum Visualisation")

    axs[1, 0].clear()
    axs[1, 0].plot(theta_values[:, 0], theta_values[:, 1], color='orange', label=f'Phase Plot')
    axs[1, 0].plot(theta_values[frame, 0], theta_values[frame, 1], 'ro')  # Red tracer ball
    axs[1, 0].set_xlabel('Angle $\\theta$ (rad)')
    axs[1, 0].set_ylabel('Angular Velocity, $\\frac{d\\theta}{dt}$ ($rad s^{-1}$)')
    axs[1, 0].set_title("Phase Portrait")
    axs[1, 0].legend(loc='upper right')
    #set_pi_ticks(axs[1, 0], theta_values)

    axs[1, 1].clear()
    axs[1, 1].set_xlim(0, interval[1])
    axs[1, 1].plot(t_values, (theta_values[:, 0] % (2 * np.pi)), color='black')
    axs[1, 1].plot(t_values[frame], (theta_values[frame, 0] % (2 * np.pi)), color='red', marker='o')
    axs[1, 1].set_title("Normalized Theta")
    set_pi_ticks(axs[1, 1], theta_values % (2 * np.pi))
    
    for angle_label in [1, 1/2, 0, 3/2]:
        if angle_label == 0:
            angle_label_text = '2$\\pi$ / 0'
        else:
            angle_label_text = f'{angle_label}$\\pi$'
        
        axs[0, 1].text((L + 0.2) * np.cos(angle_label * np.pi - (np.pi/2)), (L + 0.1) * np.sin(angle_label * np.pi - (np.pi/2)), 
                    angle_label_text, color='black', ha='center', va='center')
        
def ratio_calculator(t_values, angular_displacement, angular_velocity, L_values):
    freq_values = []
    energy_values = []
    for i in range(len(t_values)):
        freq = (1/(2 * np.pi)) * np.sqrt(g/L_values[i])
        freq_values.append(freq)

    #print(freq_values)
    
    for j in range(len(t_values)): #calculate energies
        energy = (1/2) * m * ((L_values[j] * angular_velocity[j]) ** 2) - m * g * (L_values[j] * np.cos(angular_displacement[j]))
        energy_values.append(energy)
        
    #print(energy_values)

    #calculate ratio:
    ratios = [0] * len(freq_values)
    ratio_diff = [0] * len(freq_values)

    for i in range(len(freq_values)):
        ratios[i] = freq_values[i]/energy_values[i]
        
        ratio_diff[i] = ratios[i] - ratios[1]
        
    #print(ratios)
    #print(ratio_diff)

    return freq_values, energy_values, ratios, ratio_diff

Animate = False

if Animate == True:
    m = 1.0  # mass
    L = 2.0  # initial length
    k = 1.0  # damping coefficient
    g = 9.81  # gravity
    F = 9.81  # driving force amplitude
    sigma = 0.5  # std deviation of distribution (non negative)
    interval = [0, 50]  # time range
    initial_angle = 0  # initial displacement angle
    initial_ang_vel = 0  # initial angular velocity

    t_values, theta_values, L_values = pendulum_solver_random_walk(m, L, k, g, F, sigma, interval, initial_ang_vel = 0, loc = 0)
    
    fig, axs = plt.subplots(2, 2, figsize =(19.20, 10.80))

    animation = FuncAnimation(fig, update, frames=len(theta_values[:, 0]), interval=5)
    
    animation_filename = input('Save animation as: (input n to skip save)')
    
    if animation_filename == 'n':
        plt.show()

    else:
        animation.save(os.path.join(save_loc, animation_filename + '.gif'))
        
        print('animation saved')

for o in range(0, 10):
    m = 1.0
    L = 2.0
    k = 0.5
    g = 9.81  
    F = 20
    interval = [0, 50]
    sigma = 1e-20
    num_iterations = 500

    avg_ang_disp = []
    avg_ang_vel = []
    avg_L_values = []

    ang_disps = []
    ang_vels = []
    L_values_total = []

    for i in range(num_iterations):
        t_values, theta_values, L_values = pendulum_solver_random_walk(m, L, k, g, F, sigma, interval)
        ang_disp = theta_values[:, 0]
        ang_vel = theta_values[:, 1]
        
        ang_disps.append(ang_disp)
        ang_vels.append(ang_vel)
        L_values_total.append(L_values)
        
    ang_disps = np.array(ang_disps)
    ang_vels = np.array(ang_vels)
    L_values_total = np.array(L_values_total)


    for k in range(len(t_values)):
        avg_ang_disp.append(np.average(ang_disps[:, k]))
        avg_ang_vel.append(np.average(ang_vels[:, k]))
        avg_L_values.append(np.average(L_values_total[:, k]))
        
    print(np.shape(avg_ang_disp))
    print(np.shape(avg_ang_vel))
    print(np.shape(avg_L_values))
        
    freq_values, energy_values, ratios, ratio_diff = ratio_calculator(t_values, avg_ang_disp, avg_ang_vel, L_values)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].plot(t_values, avg_ang_disp, label = 'Average Angular Displacement, $\\theta$ (rad)')
    ax[0, 0].plot(t_values, avg_ang_vel, label = 'Average Angular Velocity, $\\theta$ (rad)')
    ax[0, 0].set_xlabel('Time, t, (s)')
    ax[0, 0].set_title(f'Average time-domain plot over {num_iterations} iterations')
    ax[0, 0].legend(loc = 'upper right')

    ax[0, 1].plot(avg_ang_disp, avg_ang_vel)
    ax[0, 1].set_xlabel('Average Angular Displacement, $\\theta$ (rad)')
    ax[0, 1].set_ylabel('Average Angular Velocity, $\\theta$ (rad s$^{-1}$)')
    ax[0, 1].set_title(f'Average phase plot over {num_iterations} iterations')

    ax[1, 0].plot(t_values, avg_L_values)
    ax[1, 0].set_xlabel('Time, t (s)')
    ax[1, 0].set_ylabel('Length, L (m)')
    ax[1, 0].set_title('Average random walk of L')

    ax[1, 1].plot(t_values, ratio_diff)
    ax[1, 0].set_xlabel('Time, t (s)')
    ax[1, 1].set_ylabel('$\\Delta\\frac{E}{F}$, (Js)')
    ax[1, 1].set_title('$\\frac{E(t)}{f(t)} - \\frac{E(t)}{f(t)}$')

    plt.text(0.0, -0.1, f"m = {m}, L0= {L}, k = {k}, g = {g}, F = {F}, $\\sigma$ = {sigma}", horizontalalignment='center', verticalalignment='center', transform=ax[1, 1].transAxes)

    plt.savefig(os.path.join(save_loc, f"m = {m}, L0= {L}, k = {k}, g = {g}, F = {F}, sigma = {sigma}, run number {o}" + '.png'))

print("all done")
