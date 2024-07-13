# PHYS 6017 - Computer Techniques in Physics
Here you will find my work done during the 2 hour lab sessions hosted twice a week for the module in Semester 2 of the academic year of Oct 2023 - May 2024.

## Labs
- Lab 1: Plotting graphs, estimating integrals
- Lab 2: Creating functions from scratch
- Lab 3: Working with numbers
- Lab 4: Animating a random walk
- Lab 5: Working with random numbers

## Projects
### Project 1: Forced Simple Pendulum
The equation of motion of a pendulum can be expressed as:
$$mL^2 \frac{d^2\theta}{dt^2} + k \frac{d\theta}{dt} + mgL\sin({\theta}) = FL\cos({\Omega}t)$$

Which can be written as a system of equations in the form $\vec{Y'} = A\vec{Y} - \vec{b}$, where $A$ is a matrix, $\vec{Y'}$, $\vec{Y}$ and $\vec{b}$ are column vectors.

By solving the ODEs with the Runge–Kutta–Fehlberg (RK4(5)) method, the behaviour of the pendulum can be investigated.

### Project 2: Cryptocurrency Time Series Analysis

Using simple time series analysis methods, such as Pearson's (and Spearman's) Correlation Coefficient, Cross-correlation and Q-Q plots, we analyse various BTC-X parings and see if there is any time-lag or correlation between them.
