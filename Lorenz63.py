#!/usr/bin/env python
################################################################################
# lorenz63.py
# Ken Dixon (github: kendixon)
################################################################################
# An implementation of Lorenz, Edward Norton (1963). "Deterministic nonperiodic 
# flow".
# dx/dt = sigma * (y-x)
# dy/dt = x * ( rho - z ) - y
# dz/dt = x * y - beta * z
# The model can be integrated using the double approx. method described in L63 
# or Runge-Kutta 4 (RK4).
################################################################################

import numpy as np

class Lorenz63(object):
    """
    Object represents atmospheric convection using the non-linear, 
    deterministic Lorenz 63 model
    """

    def __init__(self, x, y, z, sigma, rho, beta, dt, solver = "DBAX"):
        """
        Initialize Lorenz63 model state (x,y,z), timestep (dt), and parameters 
        (sigma, rho, and beta).
        Options for time-differencing are "RK4" and "DBAX"
        Starts at time zero.
        """
        self._x = x 
        self._y = y 
        self._z = z
        self._sigma = sigma
        self._rho = rho
        self._beta = beta
        self._dt = dt
        self._t = 0.0
        if solver == "RK4":
            self._solver = self._rk4
        else:
            self._solver = self._dbax
        self._history = {'time':[self._t],
                         'state':np.array([self._x, self._y, self._z])
                        }

    def integrate(self, n = 1):
        """
        Integrate the model state by n timesteps (default is 1 step).
        """
        for i in range(n):
            self._solver()
            self._save_state()

    def update_state(self, x, y, z):
        """
        Update state to posterior and save time and updated state to history so
        they can be retrieved later. 
        """
        self._x = x
        self._y = y
        self._z = z
        self._save_state()

    def get_history(self):
        """
        Get the dictionary containing the list of times and corresponding 
        states (x, y, z)
        """
        return self._history

    def get_current_state(self):
        """
        Return the current time and state
        """
        return self._t, self._x, self._y, self._z

    ############################################################################
    # Solvers (Time-Differencing)
    ############################################################################
    def _save_state(self):
        """
        Save time and state to history
        """
        self._history['time'].append(self._t)
        self._history['state'] = np.vstack([self._history['state'],
                                 np.array([self._x, self._y, self._z])])

    ############################################################################
    # Solvers (Time-Differencing)
    ############################################################################
    def _rk4(self): 
        '''
        Runge-Kutta 4-stage time differencing method
        '''
        # Stage 1
        k1x = self._dxdt(self._sigma, self._x0, self._y0)
        k1y = self._dydt(self._rho, self._x0, self._y0, self._z0)
        k1z = self._dzdt(self._beta, self._x0, self._y0, self._z0)

        # Stage 2
        k2x = self._dxdt(self._sigma, self._x0 + self._dt*(k1x/2.0), 
                         self._y0 + self._dt*(k1y/2.0))
        k2y = self._dydt(self._rho, self._x0 + self._dt*(k1x/2.0), 
                         self._y0 + self._dt*(k1y/2.0), 
                         self._z0 + self._dt*(k1z/2.0))
        k2z = self._dzdt(self._beta, self._x0 + self._dt*(k1x/2.0),
                         self._y0 + self._dt*(k1y/2.0),
                         self._z0 + self._dt*(k1z/2.0))

        # Stage 3
        k3x = self._dxdt(self._sigma, self._x + self._dt*(k2x/2.0), 
                         self._y + self._dt*(k2y/2.0))
        k3y = self._dydt(self._rho, self._x + self._dt*(k2x/2.0),
                         self._y + self._dt*(k2y/2.0), 
                         self._z + self._dt*(k2z/2.0))
        k3z = self._dzdt(self._beta,  self._x + self._dt*(k2x/2.0),
                         self._y + self._dt*(k2y/2.0),
                         self._z + self._dt*(k2z/2.0))

        # Stage 4
        k4x = dxdt(self._sigma, self._x + dt*(k3x/6.0),
                   self._y + self._dt*(k3y/6.0))
        k4y = dydt(self._rho, self._x + dt*(k3x/6.0), 
                   self._y + self._dt*(k3y/6.0), 
                   self._z + self._dt*(k3z/6.0))
        k4z = dzdt(self._beta, self._x + dt*(k3x/6.0), 
                   self._y + self._dt*(k3y/6.0),
                   self._z + self._dt*(k3z/6.0))

        self._x = self._x + self._dt * (k1x/6.0 + k2x/3.0 + k3x/3.0 + k4x/6.0)
        self._y = self._y + self._dt * (k1y/6.0 + k2y/3.0 + k3y/3.0 + k4y/6.0)
        self._z = self._z + self._dt * (k1z/6.0 + k2z/3.0 + k3z/3.0 + k4z/6.0)

        self._t += self._dt

    def _dbax(self):
        '''
        Double Approximation time-differencing method described in L63
        '''
        kx1 = self._dxdt(self._sigma, self._x, self._y)
        ky1 = self._dydt(self._rho, self._x, self._y, self._z)
        kz1 = self._dzdt(self._beta, self._x, self._y, self._z)

        x1g = self._x + self._dt * kx1
        y1g = self._y + self._dt * ky1
        z1g = self._z + self._dt * kz1

        kx2 = self._dxdt(self._sigma, x1g, y1g)
        ky2 = self._dydt(self._rho, x1g, y1g, z1g)
        kz2 = self._dzdt(self._beta, x1g, y1g, z1g)

        self._x = self._x + self._dt * (kx1/2.0 + kx2/2.0)
        self._y = self._y + self._dt * (ky1/2.0 + ky2/2.0)
        self._z = self._z + self._dt * (kz1/2.0 + kz2/2.0)

        self._t += self._dt

    ############################################################################
    # Differentials
    ############################################################################
    def _dxdt(self, s, x, y):
        return s * (y - x)

    def _dydt(self, r, x, y, z):
        return x * (r - z) - y

    def _dzdt(self, b, x, y, z):
        return (x * y) - (b * z)


################################################################################
    def get_state(self):
        """
        Return x, y, z, t
        """
        return (self._x, self._y, self._z, self._t)

    def get_parameters(self):
        """
        Return system parameters
        """
        return (self._sigma, self._rho, self._beta)        

    def get_timestep(self):
        """
        Return dt (timestep)
        """
        return self._dt

################################################################################
################################################################################
if __name__ == '__main__':
    '''
    Test case for the Lorenz63 class that sets system parameters, time settings,
    and initial conditions to match the original Lorenz (1963) paper. Results 
    are also printed to match published results.
    '''
    # System parameters
    rho = 28; sigma = 10; beta = 8.0 / 3.0

    # Initial conditions
    x0, y0, z0 = 0.0, 1.0, 0.0

    # Time settings
    dt = 0.01
    N = 160

    # Initialize solver
    model = Lorenz63(x0, y0, z0, sigma, rho, beta, dt, solver = "DBAX")

    # Integrate
    model.integrate(n = N)

    # Plot Data
    h = model.get_history()
    for t, s in zip(h['time'][::5], h['state'][::5]):
        print("%04d % 05d % 05d % 05d" %(round(t*1e2), 10*s[0], 10*s[1], 10*s[2]))
