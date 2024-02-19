import math
import random 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def distance(n, step_size, zero_start = False):
    # Computes distance moved after n random steps
    x = y = 0
    mx = [0] * n
    my = [0] * n
    
    if zero_start:
        mx[0] = 0
        my[0] = 0
    
        for i in range(1, n):
            angle = 2.0 * math.pi * random.uniform(0, 1)
            x += step_size * math.cos(angle) 
            y += step_size * math.sin(angle)
            mx[i] = x
            my[i] = y
            
    else:
        for i in range(n):
            angle = 2.0 * math.pi * random.uniform(0, 1)
            x += step_size * math.cos(angle) 
            y += step_size * math.sin(angle)
            mx[i] = x
            my[i] = y
            
    return mx, my

n = 100

mx, my = distance(n, 2)
mx2, my2 = distance(n, 1)

# Set up the figure and axis
fig, ax = plt.subplots()

# Plot initial position
line, = ax.plot(mx, my, label = "step size of 2")
line2, = ax.plot(mx2, my2, label = "step size of 1")

def update(frame):
    x = mx[:frame]
    y = my[:frame]
    
    x2 = mx2[:frame]
    y2 = my2[:frame]    
        
    line.set_data(x, y)
    line2.set_data(x2, y2)

    return (line, line2)

num_frames = n

fig.suptitle("Random Walk Animation")
ax.legend(loc = 'upper left')
ax.grid()
ax.scatter(mx[0], my[0], color = 'red')
ax.scatter(mx2[0], my2[0], color = 'green')
ani = animation.FuncAnimation(fig, func = update, frames=num_frames, interval=0.05)

# Show the animation
plt.show()
