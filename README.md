Lorenz63
========

A Python class that implements the forced dissipative model described in 
Lorenz, Edward Norton (1963) "Deterministic Nonperiodic Flow" (L63). The model 
can be integrated using the double approximation method described in L63 or 
Runge-Kutta 4 (RK4).

## Usage

The model can be run as follows:
<pre><code>from Lorenz63 import *

rho = 28; sigma = 10; beta = 8.0 / 3.0   # set parameters
x0, y0, z0 = 0, 0, 0;                    # set initial conditions
dt = 0.01                                # set timestep

model = Lorenz63(x0, y0, z0, sigma, rho, beta, dt, solver = "DBAX")

model.integrate(n = 160)  # Integrate
h = model.get_history()   # Retrieve history (dictionary) 
</code></pre>

Running the Lorenz63.py directly will create the model using parameters 
described in the L63 and print out results in the format of Table 1 of L63.
