thank you for taking the time to review my code.
you would start off with "forced simple pendulum.py" for the first section of code - F~mg and increasing k until something happens.

you are looking at a simple ode solver followed by some simple functions that help with plotting an update function for the animations.
you will find some statements:
big_plot     #plots a big plot containing results for particular simulations (NOT USED IN REPORT)
Animate      #does an animation
plot_test    #plots scenarios of different values of each parameters into one big plot
demo_plot    #demonstration plots to compare time domain and phase domain plots
investigate  #main code for investigating F~mg

set whichever variable to True to run the plots

in the 2nd code, "rayleigh lorentz pendulum.py", it is more straightforward.
there is another ode solver, but now for the case where the length, L is changed based on a random walk and there is now no more forcing frequency omega. following that are the same simple functions for plotting and update function, followed by a ratio calculator that calculates the value of E/f.

Animate is set to False, but if you set to True it will plot an animation as well. the bulk of the code after is just running simulations over and over again and plotting averages.