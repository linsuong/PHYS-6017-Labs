
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import RK45

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
    y0 = np.array([initial_angle, initial_ang_vel]) #[intial displacement angle, intial angular velocity]
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

def pendulum_position(angle, length):
    x = length * np.sin(angle)
    y = length * np.cos(angle)
    
    return x, y

def update(frame):
    for ax in axs.flatten():
        ax.clear()
    
    axs[0, 0].set_xlim(0, interval[1])
    axs[0, 0].plot(t_values, theta_values[:, 0], color = 'blue', label=f'Theta')
    axs[0, 0].plot(t_values[frame], theta_values[frame, 0], 'ro')  # Red tracer ball    
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Theta')
    axs[0, 0].set_title("Theta vs Time")
    axs[0, 0].legend(loc = 'upper right')
    set_pi_ticks(axs[0, 0], theta_values)

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
    axs[1, 0].plot(t_values, theta_values[:, 1], color='green', label=f'Angular Velocity')
    axs[1, 0].plot(t_values[frame], theta_values[frame, 1], 'ro')  # Red tracer ball
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Angular Velocity')
    axs[1, 0].set_title("Angular Velocity vs Time")
    axs[1, 0].legend(loc='upper right')
    set_pi_ticks(axs[1, 0], theta_values)

    axs[1, 1].clear()
    #axs[1, 1].set_xlim(0, 2 * np.pi)
    #axs[1, 1].set_ylim(-1.5 * L, 1.5 * L)
    
    axs[1, 1].plot(t_values, (theta_values[:, 0] % (2 * np.pi)), color='black')
    axs[1, 1].plot(t_values[frame], (theta_values[frame, 0] % (2 * np.pi)), color='red', marker='o')
    axs[1, 1].set_title("Normalized Theta")
    set_pi_ticks(axs[1, 1], theta_values % (2 * np.pi))
    
    for angle_label in [3/2, 0, 1, 1/2]:
        if angle_label == 0:
            angle_label_text = '0/2$\\pi$'
        else:
            angle_label_text = f'{angle_label}$\\pi$'
        
        axs[0, 1].text((L + 0.2) * np.sin(angle_label * np.pi), (L + 0.1) * np.cos(angle_label * np.pi), 
                    angle_label_text, color='black', ha='center', va='center')


# Using the RK45 module in scipy.integrate
k = 1.5 #damping coefficient
m = 10 #mass of pendulum (kg)
g = 9.81 #gravitational constant (m * s^-1)
L = 0.5 #length of string (meters)
F = 100 #magnitude of external force (N)
eta = 0.05 #small deviation in forcing frequency, Omega
interval = [0, 25] #time interval (s)

fig2, axs2 = plt.subplots(2, 2)
for i in np.linspace(1, 21, 10):
    t_values, theta_values = pendulum_solver(i, L, k, g, F, eta, interval, np.pi, np.pi)

    axs2[0, 0].plot(t_values, theta_values[:, 0], label= f'm = {i:.1f}kg')

axs2[0, 0].set_title('$\\theta$ against $t$ as mass $m$ increases')
axs2[0, 0].set_xlabel('time, $t$')
axs2[0, 0].set_ylabel('$\\theta$')
axs2[0, 0].legend(loc='upper right')

for j in np.linspace(0, 300, 10):
    t_values, theta_values = pendulum_solver(m, L, k, g, j, eta, interval, np.pi, np.pi)
    
    axs2[0, 1].plot(t_values, theta_values[:, 0], label= f'F = {j:.1f}N')
    
axs2[0, 1].set_title('$\\theta$ against $t$ as force $F$ increases')
axs2[0, 1].set_xlabel('time, $t$')
axs2[0, 1].set_ylabel('$\\theta$')
axs2[0, 1].legend(loc='upper right')


for l in np.linspace(0.1, 2.1, 10):
    t_values, theta_values = pendulum_solver(m, l, k, g, F, eta, interval, np.pi, np.pi)
    
    axs2[1, 0].plot(t_values, theta_values[:, 0], label=f'L = {l:.1f}m')
    
axs2[1, 0].set_title('$\\theta$ against $t$ as length $L$ increases')
axs2[1, 0].set_xlabel('time, $t$')
axs2[1, 0].set_ylabel('$\\theta$')
axs2[1, 0].legend(loc='upper right')

for p in np.linspace(0, 1, 10):
    t_values, theta_values = pendulum_solver(m, L, p, g, F, eta, interval, np.pi, np.pi)
    
    axs2[1, 1].plot(t_values, theta_values[:, 0], label = f'$\\eta$ = {p:.1f}m')
    
axs2[1, 1].set_title('$\\theta$ against $t$ as coeffecient $\\eta$ increases')
axs2[1, 1].set_xlabel('time, $t$')
axs2[1, 1].set_ylabel('$\\theta$')
axs2[1, 1].legend(loc='upper right')

plt.show()

# Plot the results
big_plot = False
Animate = True

if big_plot == True:
    fig1, axs1 = plt.subplots(2, 2)

    axs1[0, 0].set_xlim(0, interval[1])
    axs1[0, 0].plot(t_values, theta_values[:, 0], color='blue', label=f'Theta')
    axs1[0, 0].set_xlabel('Time')
    axs1[0, 0].set_ylabel('Theta')
    axs1[0, 0].set_title("Theta vs Time")
    axs1[0, 0].legend()
    set_pi_ticks(axs1[0, 0], data = theta_values)

    axs1[0, 1].clear()
    axs1[0, 1].set_xlim(0, interval[1])
    axs1[0, 1].plot(t_values, theta_values[:, 0] % (2 * np.pi), color='black')
    axs1[0, 1].set_title("Normalized Theta")
    axs1[0, 1].set_ylabel('Angle')
    axs1[0, 1].set_xlabel('Time')
    set_pi_ticks(axs1[0, 1], data = theta_values % (2 * np.pi))

    axs1[1, 0].clear()
    axs1[1, 0].set_xlim(0, interval[1])
    axs1[1, 0].plot(t_values, theta_values[:, 1], color = 'green', label = f'Angular Velocity')
    axs1[1, 0].set_xlabel('Time')
    axs1[1, 0].set_ylabel('Angular Velocity,  $rad s^{-1}$')
    axs1[1, 0].set_title("Angular Velocity vs Time")
    axs1[1, 0].legend()
    set_pi_ticks(axs1[1, 0], data = theta_values)


    axs1[1, 1].text(
        0.5, 0.5,
        f"Parameters: \n$k$ = {k}\n$m$ = {m}\n$L$ = {L}\n$g$ = {g}\n$F$ = {F}\n$\\eta$ = {eta}",
        ha ='center', va ='center', transform = axs1[1, 1].transAxes,
        bbox = dict(facecolor = 'white', alpha = 0.2, pad = 8), fontsize = 15
        )
    axs1[1, 1].axis('off')

    plt.show()

if Animate == True:
    fig, axs = plt.subplots(2, 2)

    animation = FuncAnimation(fig, update, frames=len(theta_values[:, 0]), interval=5)

    plt.show()