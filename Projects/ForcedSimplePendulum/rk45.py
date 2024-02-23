import numpy as np

def rk45(function, t, pos_i, vel_i, h, epsi, hrange, mu):
    """_summary_

    Args:
        function (_type_): 
            function representing the ODE 
        t (_type_): _description_
        pos_i (_type_): _description_
        vel_i (_type_): _description_
        h (int): 
            step size
        epsi (int): 
            allowed value of the error of the solution to the ODE
        hrange (list):
            min max value of h range, format [h_min, h_max]

        mu (_type_): _description_
    """
    
    # Coeficients from Hairer, Norsett & Wanner 1993

    # Coeffients for each k:
    B21 =   2.500000000000000e-01  #  1/4
    B31 =   9.375000000000000e-02  #  3/32
    B32 =   2.812500000000000e-01  #  9/32
    B41 =   8.793809740555303e-01  #  1932/2197
    B42 =  -3.277196176604461e+00  # -7200/2197
    B43 =   3.320892125625853e+00  #  7296/2197
    B51 =   2.032407407407407e+00  #  439/216
    B52 =  -8.000000000000000e+00  # -8
    B53 =   7.173489278752436e+00  #  3680/513
    B54 =  -2.058966861598441e-01  # -845/4104
    B61 =  -2.962962962962963e-01  # -8/27
    B62 =   2.000000000000000e+00  #  2
    B63 =  -1.381676413255361e+00  # -3544/2565
    B64 =   4.529727095516569e-01  #  1859/4104
    B65 =  -2.750000000000000e-01  # -11/40

# Coefficients for the Truncation error (of the taylor expansion)
    CT1  =   2.777777777777778e-03  #  1/360
    CT2  =   0.000000000000000e+00  #  0
    CT3  =  -2.994152046783626e-02  # -128/4275
    CT4  =  -2.919989367357789e-02  # -2197/75240
    CT5  =   2.000000000000000e-02  #  1/50
    CT6  =   3.636363636363636e-02  #  2/55

# Coefficients for the weighted average (4th order)
# 4th order is used as it is the order to which the error is calculated (Note CH6 is 0)
    CH1  =   1.157407407407407e-01  #  25/216
    CH2  =   0.000000000000000e+00  #  0
    CH3  =   5.489278752436647e-01  #  1408/2565
    CH4  =   5.353313840155945e-01  #  2197/4104
    CH5  =  -2.000000000000000e-01  # -1/5
    CH6  =   0.000000000000000e+00  #  0

    
    TE_step = epsi + 1
    while TE_step > epsi:
        
        for h in range(hrange):

            k1 = function(pos_i, mu)     

            pos_k2 = pos_i + vel_i * B21 * h 
            k2 = function(pos_k2, mu)
            vel_k2 = vel_i + k1 * B21 * h

            pos_k3 = pos_i + vel_i * B31 * h + vel_k2 * B32 * h
            k3 = function(pos_k3, mu)
            vel_k3 = vel_i + k1 * B31 * h + k2 * B32 * h

            pos_k4 = pos_i + vel_i * B41 * h + vel_k2 * B42 * h + vel_k3 * B43 * h
            k4 = function(pos_k4, mu)
            vel_k4 = vel_i + k1 * B41 * h + k2 * B42 * h + k3 * B43 * h

            pos_k5 = pos_i + vel_i * B51 * h + vel_k2 * B52 * h + vel_k3 * B53 * h + vel_k4 * B54 * h
            k5 = function(pos_k5, mu)
            vel_k5 = vel_i + k1 * B51 * h + k2 * B52 * h + k3 * B53 * h + k4 * B54 * h

            pos_k6 = pos_i + vel_i * B61 * h + vel_k2 * B62 * h + vel_k3 * B63 * h + vel_k4 * B64 * h + vel_k5 * B65 * h
            k6 = function(pos_k6, mu)
            vel_k6 = vel_i + k1 * B61 * h + k2 * B62 * h + k3 * B63 * h + k4 * B64 * h + k5 * B65 * h

            a = CH1 * k1 + CH2 * k2 + CH3 * k3 + CH4 * k4 + CH5 * k5 + CH6 * k6  # Estimate is just the weighted average
            
            TE_step = np.abs(CT1 * k1 + CT2 * k2 + CT3 * k3 + CT4 * k4 + CT5 * k5 + CT6 * k6)
            TE_step = np.max(TE_step)  # Error is always the maximum error
            
            if TE_step == 0:  # If error is zero, accept the estimate
                t = t + h
                
                return TE_step, t, a, h
                
            h = 0.9 * h * (epsi / TE_step) ** (1/5)
            h = np.min(np.array(h))
                    
        t = t + h
    
    return TE_step, t, a, h