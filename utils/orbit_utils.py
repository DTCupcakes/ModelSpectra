import numpy as np

'''
Constants
- All units are in cgs
'''
pi = np.pi

c = 3e10 # Speed of light
cms_to_kms = 1e-5

G = 6.67e-8 # Gravitational constant
R_sol = 6.9634e10 # Solar radius in cgs units
M_sol = 2.e33 # Solar mass in cgs units
WD_mass_sol = 0.705 # White dwarf mass in solar units
WD_mass = WD_mass_sol*M_sol # White dwarf mass in cgs units

'''
Orbit integration functions
'''
def acc(x):
    # Get acc. from pos.
    r = np.sqrt(x[0]**2+x[1]**2)
    a = (-G*WD_mass/r**3)*x
    return a

def integrate_orbit(semia, e):
    # Integrate over an orbit with eccentricity e and return velocities
    n_points = 1000
    period = 2*pi*np.sqrt(semia**3/(G*WD_mass))
    dt = period/n_points # timestep
    x = np.array([semia*(1+e),0]) # Initial position
    v = np.array([0,np.sqrt(G*WD_mass*(1-e)/(semia*(1+e)))]) # Initial velocity
    vx_n = np.zeros(n_points)
    vy_n = np.zeros(n_points)
    for n in range(n_points):
        # Integrate using the leapfrog method
        a = acc(x)
        v = v + 0.5*dt*a
        x = x + dt*v
        a = acc(x)
        v = v + 0.5*dt*a
        vx_n[n] = v[0]*cms_to_kms
        vy_n[n] = v[1]*cms_to_kms
    return vx_n, vy_n

def get_model(alpha, semia, e):
    # Return velocity magnitude for a set of angles (alpha) in velocity space and a given semia and e
    semia = semia*R_sol
    vx, vy = integrate_orbit(semia, e)
    v_mag = np.sqrt(vx**2 + vy**2)
    v_angle = np.arctan2(vy, vx)
    return np.interp(alpha, v_angle, v_mag, period=2*pi)

def get_model_with_phase(alpha, semia, e, phase):
    semia = semia*R_sol
    vx, vy = integrate_orbit(semia, e)
    v_mag = np.sqrt(vx**2 + vy**2)
    v_angle = np.arctan2(vy, vx)
    alpha = alpha + phase
    return np.interp(alpha, v_angle, v_mag, period=2*pi)

