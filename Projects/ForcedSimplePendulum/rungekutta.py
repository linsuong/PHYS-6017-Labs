import numpy as np
def rk3(A, bvector, y0, interval, N):
    """
    Solve the IVP y' = A y + b, y(0) = y0, in the interval,
    using N steps of RK3.

    Parameters
    ----------
    A : matrix
        Partially defines ODE according to (4)
    bvector : function name returning vector
        Completes definition of ODE according to (4)
    y0 : vector
        Initial data
    interval : vector
        Interval on which solution is required
    N : int
        Number of steps

    Returns
    -------
    x : array of float
        Coordinate locations of the approximate solution
    y : array of float
        Values of approximate solution at locations x
    """

    # Add RK3 algorithm implementation here according to Task 1

    # The first output argument x is the locations x_j at which the solution is evaluated;
    # this should be a real vector of length N + 1 covering the required interval.
    # The second output argument y should be the numerical solution approximated at the
    # locations x_j, which will be an array of size n × (N + 1).

    a, b = interval

    h = (b - a)/N
    x = np.linspace(a, b, N + 1)
    y = np.zeros((N + 1, len(y0))).T
    y[:, 0] = y0
    
    for j in range(N):
        x_n = x[j]
        y_n = y[:, j]

        b1 = bvector(x_n + h)

        y_1 = y_n + h * (A @ y_n + bvector(x_n))

        y_2 = ((3/4) * y_n) + ((1/4) * y_1) + (1/4) * h * (A @ y_1 + b1)

        y[:, j + 1] = (y_n / 3) + ((2 / 3) * y_2) + (2 / 3) * h * (A @ y_2 + b1)

    return h, x, y

'''def rk45(A, b_vector, y0, hrange, epsi, *args):
    """
    Solve the IVP y' = A y + b, y(0) = y0, using the Runge–Kutta–Fehlberg method (RKF 4(5))

    Args:
        ode_function (callable): 
            Function representing the ODE.
        x (float): 
            Current value of the independent variable.
        y (float): 
            Current value of the dependent variable.
        hrange (list): 
            Min and max values of h range, format [h_min, h_max].
        epsi (float): 
            Allowed value of the error of the solution to the ODE.
        *args: 
            Additional parameters needed by the ODE function.
    """
    
    # Coefficients from Hairer, Norsett & Wanner 1993
    B21 = 2.500000000000000e-01
    B31 = 9.375000000000000e-02
    B32 = 2.812500000000000e-01
    B41 = 8.793809740555303e-01
    B42 = -3.277196176604461e+00
    B43 = 3.320892125625853e+00
    B51 = 2.032407407407407e+00
    B52 = -8.000000000000000e+00
    B53 = 7.173489278752436e+00
    B54 = -2.058966861598441e-01
    B61 = -2.962962962962963e-01
    B62 = 2.000000000000000e+00
    B63 = -1.381676413255361e+00
    B64 = 4.529727095516569e-01
    B65 = -2.750000000000000e-01

    CT1 = 2.777777777777778e-03
    CT2 = 0.000000000000000e+00
    CT3 = -2.994152046783626e-02
    CT4 = -2.919989367357789e-02
    CT5 = 2.000000000000000e-02
    CT6 = 3.636363636363636e-02

    CH1 = 1.157407407407407e-01
    CH2 = 0.000000000000000e+00
    CH3 = 5.489278752436647e-01
    CH4 = 5.353313840155945e-01
    CH5 = -2.000000000000000e-01
    CH6 = 0.000000000000000e+00
    
    def ode_function(t, y):
        return A @ y + b

    if not isinstance(hrange, list) or len(hrange) != 2:
        raise ValueError("hrange must be a list of format [h start, h finish].")
    
    h_start, h_finish = hrange
    
    TE_step = epsi + 1
    
    while TE_step > epsi:

        for h in range(h_start, h_finish):

            k1 = h * ode_function(x, y, *args)
            k2 = h * ode_function(x + B21 * k1, y + B21 * h, *args)
            k3 = h * ode_function(x + B31 * k1 + B32 * k2, y + B32 * h, *args)
            k4 = h * ode_function(x + B41 * k1 + B42 * k2 + B43 * k3, y + B43 * h, *args)
            k5 = h * ode_function(x + B51 * k1 + B52 * k2 + B53 * k3 + B54 * k4, y + B54 * h, *args)
            k6 = h * ode_function(x + B61 * k1 + B62 * k2 + B63 * k3 + B64 * k4 + B65 * k5, y + B65 * h, *args)

            a = CH1 * k1 + CH2 * k2 + CH3 * k3 + CH4 * k4 + CH5 * k5 + CH6 * k6  # Estimate is just the weighted average

            TE_step = np.abs(CT1 * k1 + CT2 * k2 + CT3 * k3 + CT4 * k4 + CT5 * k5 + CT6 * k6)
            TE_step = np.max(TE_step)  # Error is always the maximum error

            if TE_step == 0:  # If error is zero, accept the estimate
                x = x + h
                return TE_step, x, a, h

            h = 0.9 * h * (epsi / TE_step) ** (1/5)
            h = np.min(np.array(h))

        x = x + h

    return TE_step, x, y, h'''

def rk45_solver(A, b, y0, t_span, h, error_allowance):
    """
    Solve the ODE Y' = AY + b using the RK45 method.

    Parameters:
    - A: Square matrix in the ODE.
    - b: Column vector in the ODE.
    - y0: Initial condition column vector.
    - t_span: Tuple (t0, t_end) specifying the time span.
    - h: Step size.

    Returns:
    - t_values: Array of time points.
    - y_values: Array of corresponding solutions.
    """

    def ode_func(t, y):
        return A @ y + b

    t0, t_end = t_span
    t_values = [t0]
    y_values = [y0]

    while t_values[-1] < t_end:
        t_current = t_values[-1]
        y_current = y_values[-1]

        k1 = h * ode_func(t_current, y_current)
        k2 = h * ode_func(t_current + 0.25 * h, y_current + 0.25 * k1)
        k3 = h * ode_func(t_current + 3/8 * h, y_current + 3/32 * k1 + 9/32 * k2)
        k4 = h * ode_func(t_current + 12/13 * h, y_current + 1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
        k5 = h * ode_func(t_current + h, y_current + 439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
        k6 = h * ode_func(t_current + 0.5 * h, y_current - 8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)

        y_new = y_current + 25/216 * k1 + 1408/2565 * k3 + 2197/4104 * k4 - 1/5 * k5
        error = np.max(np.abs(1/360 * k1 - 128/4275 * k3 - 2197/75240 * k4 + 1/50 * k5 + 2/55 * k6))

        if error < error_allowance:
            h *= 2              #h = h * 2

        t_values.append(t_current + h)
        y_values.append(y_new)

        if t_values[-1] + h > t_end:
            h = t_end - t_values[-1]

    return np.array(t_values), np.array(y_values)

# Example usage:
# Define A, b, y0, t_span, and h
# A = np.array([[1, -2], [3, -4]])  # Example 2x2 matrix
# b = np.array([1, 2])  # Example 2x1 column vector
# y0 = np.array([0, 0])  # Example initial condition
# t_span = (0, 5)  # Example time span
# h = 0.1  # Example step size

# t_values, y_values = rk45_solver(A, b, y0, t_span, h)
# print("Time values:", t_values)
# print("Solution values:", y_values)

'''
# Define a simple ODE function for testing purposes
def simple_ode(x, y, a=1):
    return (a * y[1], 0)

x_val = np.linspace(0, 100, 1000)
y_0 = [0, 0]

TE_step, x, y, h = rk45(simple_ode, x_val, y_0, [0, 10], 0.01)

plt.plot(x, y)
plt.show()
'''