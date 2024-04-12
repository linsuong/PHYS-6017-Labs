import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from scipy.integrate import RK45

save_loc = "C:\\Repositories\\PHYS-6017-Labs\\Projects\\ForcedSimplePendulum\\Plots"

#SOLVING THE MATRIX EQUATION Y' = AY + b
def pendulum_solver(m, L, k, g, F, eta, time_range, initial_angle, initial_ang_vel):
    #parameters:
    def alphas(k, m, g, L):
        return (k/(m * L * np.sqrt(g * L)))

    def betas(F, m, g):
        return (F/(m * g))

    def ode_system(t, y, A, bvector):
        
        return A @ y + bvector(t)

    #solving with RK45 module:
    def solve_rk45(A, bvector, y0, time_range, max_step = 0.1):
        ode_solver = RK45(fun = lambda t, y: ode_system(t, y, A, bvector), t0 = time_range[0], 
                          y0=y0, t_bound = time_range[1], max_step = max_step)

        # Lists to store the results
        t_values = [ode_solver.t]
        y_values = [ode_solver.y]

        while ode_solver.status == 'running':
            ode_solver.step()
            t_values.append(ode_solver.t)
            y_values.append(ode_solver.y)

        return np.array(t_values), np.array(y_values)
    
    #for a pendulum starting from rest:
    y0 = np.array([initial_angle, initial_ang_vel]) #[initial displacement angle, initial angular velocity]
    alpha = alphas(k, m, g, L)
    beta = betas(F, m, g)

    A = np.array([[0, 1], [-np.sin(np.pi/2), -alpha]]) 
    bvector = lambda t: np.array([0, beta * np.cos((1 - eta) * t)])

    t_values, theta_values = solve_rk45(A, bvector, y0, time_range, max_step = 0.05)

    return t_values, theta_values

def set_pi_ticks(ax, data):
    ax_ticks = np.arange(min(data[:, 0]), max(data[:, 0] + np.pi), np.pi)
    ax.set_yticks(ax_ticks)
    ax.set_yticklabels([f"{i//np.pi}$\\pi$" for i in ax_ticks])
    
def set_pi_ticks_x(ax, data):
    ax_ticks = np.arange(min(data[:, 0]), max(data[:, 0] + np.pi), np.pi)
    ax.set_xticks(ax_ticks)
    ax.set_xticklabels([f"{i//np.pi}$\\pi$" for i in ax_ticks])

def pendulum_position(angle, length):
    x = length * np.cos(angle - (np.pi/2))
    y = length * np.sin(angle - (np.pi/2))
    
    return x, y

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
    x, y = pendulum_position(angle, L)
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


# Plot the results
big_plot = False #plots a big plot containing results for particular simulations (NOT USED IN REPORT)
Animate = True #does an animation
plot_test = False #plots scenarios of different values of each parameters into one big plot
demo_plot = False #demonstration plots to compare time domain and phase domain plots
investigate = True #main code for investigating F~mg

if demo_plot == True:
    k = 1 #damping coefficient
    m = 1 #mass of pendulum (kg)
    g = 9.81 #gravitational constant (m * s^-2)
    L = 1 #length of string (meters)
    F = 10 #magnitude of external force (N, kg * m * s^-2)
    eta = 100 #small deviation in forcing frequency, Omega
    interval = [0, 50] #time interval (s)

    t_values, theta_values = pendulum_solver(m, L, k, g, F, eta, interval, 2 * np.pi, 10)

    fig_example, ax_example = plt.subplots(1, 2, figsize=(15, 5))
    ax_example[0].plot(t_values, theta_values[:, 0])      
    ax_example[0].set_xlabel('time, $t$')
    ax_example[0].set_ylabel('$\\theta$')
    ax_example[0].set_title('Time Domain Plot')
    ax_example[1].plot(theta_values[:, 0], theta_values[:, 1], marker = '>', markevery = 50, linestyle = '--')
    ax_example[1].set_xlabel('$\\theta$')
    ax_example[1].set_ylabel('$\\frac{d\\theta}{dt}$')
    ax_example[1].set_title('Phase Domain Plot')

    plt.show()
    
    filename = input('Save as:')
    
    plt.savefig(os.path.join(save_loc, filename + '.png'))
    
    fig, axs = plt.subplots(2, 2, figsize =(19.20, 10.80))

    animation = FuncAnimation(fig, update, frames=len(theta_values[:, 0]), interval=5)
    
    plt.show()
    
    animation.save(os.path.join(save_loc, filename + '.gif'))
    
    print('animation saved')

if plot_test == True:
    k = 0.5 #damping coefficient
    m = 1 #mass of pendulum (kg)
    g = 9.81 #gravitational constant (m * s^-2)
    L = 2 #length of string (meters)
    F = 5 #magnitude of external force (N, kg * m * s^-2)
    eta = 0.1 #small deviation in forcing frequency, Omega
    interval = [0, 50] #time interval (s)
    
    fig2, axs2 = plt.subplots(3, 2, figsize=(15, 17))
    for i in np.linspace(1, 21, 10):
        t_values, theta_values = pendulum_solver(i, L, k, g, F, eta, interval, np.pi, 0)

        axs2[0, 0].plot(t_values, theta_values[:, 0], label= f'm = {i:.1f}kg')

    axs2[0, 0].set_title('$\\theta$ against $t$ as mass $m$ increases')
    axs2[0, 0].set_xlabel('time, $t$')
    axs2[0, 0].set_ylabel('$\\theta$')
    axs2[0, 0].set_xlim(interval)
    axs2[0, 0].legend(loc='upper right')
    #set_pi_ticks(axs2[0, 0], data = theta_values)

    k = 0.5 #damping coefficient
    m = 1 #mass of pendulum (kg)
    g = 9.81 #gravitational constant (m * s^-2)
    L = 2 #length of string (meters)
    F = 20 #magnitude of external force (N, kg * m * s^-2)
    eta = 0.1 #small deviation in forcing frequency, Omega
    interval = [0, 50] #time interval (s)

    for j in np.linspace(0, 300, 10):
        t_values, theta_values = pendulum_solver(m, L, k, g, j, eta, interval, np.pi, 0)
        
        axs2[0, 1].plot(t_values, theta_values[:, 0], label= f'F = {j:.1f}N')
        
    axs2[0, 1].set_title('$\\theta$ against $t$ as force $F$ increases')
    axs2[0, 1].set_xlabel('time, $t$')
    axs2[0, 1].set_ylabel('$\\theta$')
    axs2[0, 1].set_xlim(interval)
    axs2[0, 1].legend(loc='upper right')
    #set_pi_ticks(axs2[0, 1], data = theta_values)
    
    k = 0.5 #damping coefficient
    m = 1 #mass of pendulum (kg)
    g = 9.81 #gravitational constant (m * s^-2)
    L = 2 #length of string (meters)
    F = 20 #magnitude of external force (N, kg * m * s^-2)
    eta = 0.1 #small deviation in forcing frequency, Omega
    interval = [0, 50] #time interval (s)

    for l in np.linspace(0.1, 10, 10):
        t_values, theta_values = pendulum_solver(m, l, k, g, F, eta, interval, np.pi, 0)
        
        axs2[1, 0].plot(t_values, theta_values[:, 0], label=f'L = {l:.1f}m')
        
    axs2[1, 0].set_title('$\\theta$ against $t$ as length $L$ increases')
    axs2[1, 0].set_xlabel('time, $t$')
    axs2[1, 0].set_ylabel('$\\theta$')
    axs2[1, 0].set_xlim(interval)
    axs2[1, 0].legend(loc='upper right')
    #set_pi_ticks(axs2[1, 0], data = theta_values)
    
    k = 0.5 #damping coefficient
    m = 1 #mass of pendulum (kg)
    g = 9.81 #gravitational constant (m * s^-2)
    L = 2 #length of string (meters)
    F = 20 #magnitude of external force (N, kg * m * s^-2)
    eta = 0.1 #small deviation in forcing frequency, Omega
    interval = [0, 50] #time interval (s)

    for p in np.linspace(0, 10, 10):
        t_values, theta_values = pendulum_solver(m, L, p, g, F, eta, interval, np.pi, 0)
        
        axs2[1, 1].plot(t_values, theta_values[:, 0], label = f'$k$ = {p:.1f}')
        
    axs2[1, 1].set_title('$\\theta$ against $t$ as coefficient $k$ increases')
    axs2[1, 1].set_xlabel('time, $t$')
    axs2[1, 1].set_ylabel('$\\theta$')
    axs2[1, 1].set_xlim(interval)
    axs2[1, 1].legend(loc='upper right')
    #set_pi_ticks(axs2[1, 1], data = theta_values)
    
    k = 0.5 #damping coefficient
    m = 1 #mass of pendulum (kg)
    g = 9.81 #gravitational constant (m * s^-2)
    L = 2 #length of string (meters)
    F = 20 #magnitude of external force (N, kg * m * s^-2)
    eta = 0.1 #small deviation in forcing frequency, Omega
    interval = [0, 50] #time interval (s)
    
    for q in np.linspace(0, 10, 10):
        t_values, theta_values = pendulum_solver(m, L, k, g, F, q, interval, np.pi, 0)
        
        axs2[2, 0].plot(t_values, theta_values[:, 0], label = f'$\\eta$ = {q:.1f}')
        #set_pi_ticks(axs2[2, 0], data = theta_values)
        
    axs2[2, 0].set_title('$\\theta$ against $t$ as coefficient $\\eta$ increases')
    axs2[2, 0].set_xlabel('time, $t$')
    axs2[2, 0].set_ylabel('$\\theta$')
    axs2[2, 0].set_xlim(interval)
    axs2[2, 0].legend(loc='upper right')
    
    axs2[2, 1].axis('off')

    plt.savefig(os.path.join(save_loc, "test_plots.png"))
    plt.show()
    
    ##phase plots
    k = 0.5 #damping coefficient
    m = 1 #mass of pendulum (kg)
    g = 9.81 #gravitational constant (m * s^-2)
    L = 2 #length of string (meters)
    F = 20 #magnitude of external force (N, kg * m * s^-2)
    eta = 0.1 #small deviation in forcing frequency, Omega
    interval = [0, 50] #time interval (s)
    
    fig3, axs3 = plt.subplots(3, 2, figsize=(15, 17))
    for i in np.linspace(1, 21, 10):
        t_values, theta_values = pendulum_solver(i, L, k, g, F, eta, interval, np.pi, 0)
        axs3[0, 0].plot(theta_values[:, 0], theta_values[:, 1], label=f'm = {i:.1f}kg')

    axs3[0, 0].set_title('Phase Plot:$\\frac{d\\theta}{dt}$ against $\\theta$ as mass $m$ increases')
    axs3[0, 0].set_xlabel('$\\theta$')
    axs3[0, 0].set_ylabel('$\\frac{d\\theta}{dt}$')
    axs3[0, 0].legend(loc='upper right')

    k = 0.5 #damping coefficient
    m = 1 #mass of pendulum (kg)
    g = 9.81 #gravitational constant (m * s^-2)
    L = 2 #length of string (meters)
    F = 20 #magnitude of external force (N, kg * m * s^-2)
    eta = 0.1 #small deviation in forcing frequency, Omega
    interval = [0, 50] #time interval (s)
    
    for j in np.linspace(0, 300, 10):
        t_values, theta_values = pendulum_solver(m, L, k, g, j, eta, interval, np.pi, 0)
        axs3[0, 1].plot(theta_values[:, 0], theta_values[:, 1], label=f'F = {j:.1f}N')
        
    axs3[0, 1].set_title('Phase Plot:$\\frac{d\\theta}{dt}$ against $\\theta$ as force $F$ increases')
    axs3[0, 1].set_xlabel('$\\theta$')
    axs3[0, 1].set_ylabel('$\\frac{d\\theta}{dt}$')
    axs3[0, 1].legend(loc='upper right')

    k = 0.5 #damping coefficient
    m = 1 #mass of pendulum (kg)
    g = 9.81 #gravitational constant (m * s^-2)
    L = 2 #length of string (meters)
    F = 20 #magnitude of external force (N, kg * m * s^-2)
    eta = 0.1 #small deviation in forcing frequency, Omega
    interval = [0, 50] #time interval (s)
    
    for l in np.linspace(0.1, 10, 10): #length
        t_values, theta_values = pendulum_solver(m, l, k, g, F, eta, interval, np.pi, 0)
        axs3[1, 0].plot(theta_values[:, 0], theta_values[:, 1], label=f'L = {l:.1f}m')
        
    axs3[1, 0].set_title('Phase Plot:$\\frac{d\\theta}{dt}$ against $\\theta$ as length $L$ increases')
    axs3[1, 0].set_xlabel('$\\theta$')
    axs3[1, 0].set_ylabel('$\\frac{d\\theta}{dt}$')
    axs3[1, 0].legend(loc='upper right')

    k = 0.5 #damping coefficient
    m = 1 #mass of pendulum (kg)
    g = 9.81 #gravitational constant (m * s^-2)
    L = 2 #length of string (meters)
    F = 20 #magnitude of external force (N, kg * m * s^-2)
    eta = 0.1 #small deviation in forcing frequency, Omega
    interval = [0, 50] #time interval (s)
    
    for p in np.linspace(0, 10, 10): #k
        t_values, theta_values = pendulum_solver(m, L, p, g, F, eta, interval, np.pi, 0)
        axs3[1, 1].plot(theta_values[:, 0], theta_values[:, 1], label=f'$k$ = {p:.1f}')
        
    axs3[1, 1].set_title('Phase Plot:$\\frac{d\\theta}{dt}$ against $\\theta$ as coefficient $k$ increases')
    axs3[1, 1].set_xlabel('$\\theta$')
    axs3[1, 1].set_ylabel('$\\frac{d\\theta}{dt}$')
    axs3[1, 1].legend(loc='upper right')

    k = 0.5 #damping coefficient
    m = 1 #mass of pendulum (kg)
    g = 9.81 #gravitational constant (m * s^-2)
    L = 2 #length of string (meters)
    F = 20 #magnitude of external force (N, kg * m * s^-2)
    eta = 0.1 #small deviation in forcing frequency, Omega
    interval = [0, 50] #time interval (s)
    
    for q in np.linspace(0, 10, 10): #eta
        t_values, theta_values = pendulum_solver(m, L, k, g, F, q, interval, np.pi, 0)
        
        axs3[2, 0].plot(theta_values[:, 0], theta_values[:, 1], label = f'$\\eta$ = {q:.1f}')
        
    axs3[2, 0].set_title('Phase Plot:$\\frac{d\\theta}{dt}$ against $\\theta$ as coefficient $\\eta$ increases')
    axs3[2, 0].set_xlabel('$\\theta$')
    axs3[2, 0].set_ylabel('$\\frac{d\\theta}{dt}$')
    axs3[2, 0].legend(loc='upper right')
    
    axs3[2, 1].axis('off')
    
    plt.savefig(os.path.join(save_loc, "test_plots_phase.png"))
    plt.show()
    
    print('test plots done')

if big_plot == True:
    
    # Using the RK45 module in scipy.integrate
    k = 10 #damping coefficient
    m = 1 #mass of pendulum (kg)
    g = 9.81 #gravitational constant (m * s^-2)
    L = 2 #length of string (meters)
    F = 20#magnitude of external force (N, kg * m * s^-2)
    eta = 0.05 #small deviation in forcing frequency, Omega
    interval = [0, 60] #time interval (s)

    fig1, axs1 = plt.subplots(2, 2)

    axs1[0, 0].set_xlim(0, interval[1])
    axs1[0, 0].plot(t_values, theta_values[:, 0], color='blue', label=f'Theta')
    axs1[0, 0].plot(t_values, theta_values[:, 1], color = 'green', label = f'Angular Velocity')
    axs1[0, 0].set_xlabel('Time, t, (s)')
    axs1[0, 0].set_title("Angular Displacement vs Time (blue), Angular Velocity vs Time (green)")
    axs1[0, 0].legend()
    set_pi_ticks(axs1[0, 0], data = theta_values)

    axs1[0, 1].clear()
    axs1[0, 1].set_xlim(0, interval[1])
    axs1[0, 1].plot(t_values, theta_values[:, 0] % (2 * np.pi), color='black')
    axs1[0, 1].set_title("Normalized Theta vs Time")
    axs1[0, 1].set_ylabel('Angle, (rad)')
    axs1[0, 1].set_xlabel('Time')
    set_pi_ticks(axs1[0, 1], data = theta_values % (2 * np.pi))

    axs1[1, 0].clear()
    axs1[1, 0].plot(theta_values[:, 0], theta_values[:, 1], color = 'orange', label = f'Phase')
    axs1[1, 0].set_xlabel('Angle $\\theta$ (rad)')
    axs1[1, 0].set_ylabel('Angular Velocity, $\\frac{d\\theta}{dt}$ ($rad s^{-1}$)')
    axs1[1, 0].set_title("Phase Portrait")
    axs1[1, 0].legend()
    set_pi_ticks(axs1[1, 0], data = theta_values)
    set_pi_ticks_x(axs1[1, 0], data = theta_values)

    axs1[1, 1].text(
        0.5, 0.5,
        f"Parameters: \n$k$ = {k}\n$m$ = {m}\n$L$ = {L}\n$g$ = {g}\n$F$ = {F}\n$\\eta$ = {eta}",
        ha ='center', va ='center', transform = axs1[1, 1].transAxes,
        bbox = dict(facecolor = 'white', alpha = 0.2, pad = 8), fontsize = 15
        )
    axs1[1, 1].axis('off')

    plt.show()

if Animate == True:
    k = 0.41 #damping coefficient
    m = 1 #mass of pendulum (kg)
    g = 9.81 #gravitational constant (m * s^-2)
    L = 2 #length of string (meters)
    F = 10 #magnitude of external force (N, kg * m * s^-2)
    eta = 0.1 #small deviation in forcing frequency, Omega
    interval = [50, 200] #time interval (s)
    
    t_values, theta_values = pendulum_solver(m, L, k, g, F, eta, interval, np.pi, 0)
    
    fig, axs = plt.subplots(2, 2, figsize =(19.20, 10.80))

    animation = FuncAnimation(fig, update, frames=len(theta_values[:, 0]), interval=5)
    
    animation_filename = input('Save animation as: (input n to skip save)')
    
    if animation_filename == 'n':
        plt.show()

    else:
        animation.save(os.path.join(save_loc, animation_filename + '.gif'))
        
        print('animation saved')
        
if investigate == True:
    #############################################################################################################################    
        ##investigating the breakdown of simple harmonic motion F ~ mg.

    F = 10 #force in N, F ~ mg in the case as m = 1 and g = 9.81, therefore F = 9.81
    m = 1 #mass of pendulum (kg)
    g = 9.81 #gravitational constant (m * s^-2)
    L = 2 #length of string (meters)
    eta = 0.1 #small deviation in forcing frequency, Omega
    interval = [0, 50] #time interval (s)

    fig_damping, ax_damping = plt.subplots(5, 2, figsize=(15, 17))

    k_start = 5
    k_end = 1
    k_range = np.linspace(k_end, k_start, 5)
    k_range = np.flip(k_range)

    for p in range(0, 5):  
        t_values, theta_values = pendulum_solver(m, L, k_range[p], g, F, eta, interval, np.pi, 0)
        ax_damping[p, 0].plot(t_values, theta_values[:, 0], label = f'$k$ = {k_range[p]:.5f}')
        ax_damping[p, 1].plot(t_values, theta_values[:, 0] % (2 * np.pi), label = f'$k$ = {k_range[p]:.5f}m')
        #set_pi_ticks(ax_damping[p, 0], data = theta_values)
        set_pi_ticks(ax_damping[p, 1], data = theta_values % (2 * np.pi))
        ax_damping[p, 0].legend(loc = 'upper left')
        ax_damping[p, 1].legend(loc = 'upper right')
        ax_damping[p, 0].set_xlabel("time $t$")
        ax_damping[p, 0].set_ylabel("$\\theta$")
        ax_damping[p, 1].set_xlabel("time $t$")
        ax_damping[p, 1].set_ylabel("$\\theta$")
        
        ax_damping[p, 0].set_xlim(interval)
        ax_damping[p, 1].set_xlim(interval)

    fig_damping.suptitle("F ~ mg, as damping coefficient k increases (Right: Normalised by 2$\\pi$)")
    plt.savefig(os.path.join(save_loc, f"F~mg as damping coefficient k increases from {k_start} to {k_end}.png"))

    fig_damping_phase, ax_damping_phase = plt.subplots(3, 2, figsize=(15, 17))

    for p in range(0, 3):
        t_values, theta_values = pendulum_solver(m, L, k_range[p], g, F, eta, interval, np.pi, 0)
        ax_damping_phase[p, 0].plot(theta_values[:, 0], theta_values[:, 1], marker='>', markevery=50, label = f'$k$ = {k_range[p]:.5f}')
        ax_damping_phase[p, 0].legend(loc = 'upper left')
        ax_damping_phase[p, 0].set_xlabel('Angle $\\theta$ (rad)')
        ax_damping_phase[p, 0].set_ylabel('Angular Velocity, $\\frac{d\\theta}{dt}$ ($rad s^{-1}$)')
        
    for p in range(0, 2):
        t_values, theta_values = pendulum_solver(m, L, k_range[p + 3], g, F, eta, interval, np.pi, 0)
        ax_damping_phase[p, 1].plot(theta_values[:, 0], theta_values[:, 1], marker='>', markevery=50, label = f'$k$ = {k_range[p + 3]:.5f}')
        ax_damping_phase[p, 1].legend(loc = 'upper left')
        ax_damping_phase[p, 1].set_xlabel('Angle $\\theta$ (rad)')
        ax_damping_phase[p, 1].set_ylabel('Angular Velocity, $\\frac{d\\theta}{dt}$ ($rad s^{-1}$)')
        
    ax_damping_phase[2, 1].axis("off")
    fig_damping_phase.suptitle('Phase Plot: F ~ mg, as damping coefficient k increases')


    plt.savefig(os.path.join(save_loc, f"Phase plot of F~mg as damping coefficient k increases from {k_start} to {k_end}.png"))

    plt.show()
        
    k = 0.41 #damping coefficient

    fig_damping_compare, ax_damping_compare = plt.subplots(3, 1, figsize = (6, 15))
    t_values, theta_values = pendulum_solver(m, L, k, g, F, eta, interval, np.pi, 0)

    ax_damping_compare[0].plot(t_values, theta_values[:, 0], label = f'$k$ = {k}')
    ax_damping_compare[0].legend(loc = 'upper right')
    ax_damping_compare[0].set_xlabel("time $t$")
    ax_damping_compare[0].set_ylabel("$\\theta$")
    ax_damping_compare[0].set_title('Time Domain')

    ax_damping_compare[1].plot(t_values, theta_values[:, 0] % (2 * np.pi), label = f'$k$ = {k}')
    ax_damping_compare[1].legend(loc = 'upper right')
    ax_damping_compare[1].set_xlabel("time $t$")
    ax_damping_compare[1].set_ylabel("$\\theta$")
    ax_damping_compare[1].set_title('Time Domain Normalised')

    ax_damping_compare[2].plot(theta_values[:, 0], theta_values[:, 1], marker='>', markevery=50, label = f'$k$ = {k}')
    ax_damping_compare[2].legend(loc = 'upper left')
    ax_damping_compare[2].set_xlabel('Angle $\\theta$ (rad)')
    ax_damping_compare[2].set_ylabel('Angular Velocity, $\\frac{d\\theta}{dt}$ ($rad s^{-1}$)')
    ax_damping_compare[2].set_title('Phase Domain')

    fig_damping_compare.suptitle(f'F ~ mg, k = {k}')

    plt.savefig(os.path.join(save_loc, f"Plot of F~mg, k is {k} (med).png"))
    plt.show()
