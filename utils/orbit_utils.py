import numpy as np
import utils.img_utils as img

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

def GR_to_kms(v):
    # Takes an array of velocities in GR units and converts it to kms
    v *= cms_to_kms
    return v

'''
Orbit integration functions
'''
def acc(x):
    # Get acc. from pos.
    r = np.sqrt(x[0]**2+x[1]**2)
    a = (-G*WD_mass/r**3)*x
    return a

def acc_3D(x):
    # Get acc. from pos.
    r = np.sqrt(x[0]**2+x[1]**2+x[2]**2)
    a = (-G*WD_mass/r**3)*x
    return a

def integrate_orbit(semia, e, phase=0):
    '''Add phase to initial x and v'''
    # Integrate over an orbit with parameters semia, e and return velocities
    semia = semia*R_sol
    n_points = 1000
    period = 2*pi*np.sqrt(semia**3/(G*WD_mass))
    dt = period/n_points # timestep
    r0 = semia*(1-e**2)/(1+e*np.cos(pi-phase)) # Initial value of r
    x = np.array([r0,0]) # Initial position
    v = np.array([0,np.sqrt(G*WD_mass*(2/r0-1/semia))]) # Initial velocity
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

def integrate_orbit_3D(semia, e, i, O, w, f):
    #Campbell elements
    #semia should be in R_Sun
    #Angles should be in degrees 
    semia = semia*R_sol
    n_points = 1000
    period = 2*pi*np.sqrt(semia**3/(G*WD_mass))
    dt = period/n_points # timestep
    a = semia
    ecc = e
    omega = w*np.pi/180.
    # our conventions here are Omega is measured East of North
    big_omega = O*np.pi/180. + 0.5*np.pi
    inc = i*np.pi/180.
    # get eccentric anomaly from true anomaly
    # (https://en.wikipedia.org/wiki/Eccentric_anomaly#From_the_true_anomaly)
    theta = f*np.pi/180.
    E = np.arctan2(np.sqrt(1. - ecc**2)*np.sin(theta),(ecc + np.cos(theta)))
    # Positions in plane (Thiele-Innes elements)
    P = np.zeros(3)
    Q = np.zeros(3)
    P[0] = np.cos(omega)*np.cos(big_omega) - np.sin(omega)*np.cos(inc)*np.sin(big_omega)
    P[1] = np.cos(omega)*np.sin(big_omega) + np.sin(omega)*np.cos(inc)*np.cos(big_omega)
    P[2] = np.sin(omega)*np.sin(inc)
    Q[0] = -np.sin(omega)*np.cos(big_omega) - np.cos(omega)*np.cos(inc)*np.sin(big_omega)
    Q[1] = -np.sin(omega)*np.sin(big_omega) + np.cos(omega)*np.cos(inc)*np.cos(big_omega)
    Q[2] = np.sin(inc)*np.cos(omega)
    term1 = np.cos(E)-ecc
    term2 = np.sqrt(1.-(ecc*ecc))*np.sin(E)
    E_dot = np.sqrt(G*WD_mass/(a**3))/(1.-ecc*np.cos(E))
    # Rotating everything
    # Set the positions for the primary and the secondary
    x = a*(term1*P + term2*Q)
    # Set the velocities
    v = -a*np.sin(E)*E_dot*P + a*np.sqrt(1.-(ecc*ecc))*np.cos(E)*E_dot*Q
    vx_n = np.zeros(n_points)
    vy_n = np.zeros(n_points)
    a = acc_3D(x)
    for n in range(n_points):
        # Integrate using the leapfrog method
        v = v + 0.5*dt*a
        x = x + dt*v
        a = acc_3D(x)
        v = v + 0.5*dt*a
        vx_n[n] = v[0]*cms_to_kms
        vy_n[n] = v[1]*cms_to_kms
    return vx_n, vy_n

def get_model_3D(alpha, semia, e, i, O, w, f):
    vx, vy = integrate_orbit_3D(semia, e, i, O, w, f)
    v_mag = np.sqrt(vx**2 + vy**2)
    v_angle = np.arctan2(vy, vx)
    #alpha = alpha + f
    return np.interp(alpha, v_angle, v_mag, period=2*pi)

def get_model_with_phase(alpha, semia, e, phase):
    vx, vy = integrate_orbit(semia, e)
    # vx, vy = integrate_orbit_with_inc(semia, e)
    v_mag = np.sqrt(vx**2 + vy**2)
    v_angle = np.arctan2(vy, vx)
    alpha = alpha + phase
    return np.interp(alpha, v_angle, v_mag, period=2*pi)

'''
Functions to plot orbits
'''
def plot_orbit_params(ax, semia, e, phase=0):
    # Plot orbit with parameters semia and e in Cartesian coordinates
    vx_n, vy_n = integrate_orbit(semia, e, phase=phase)
    ax.plot(vx_n, vy_n, 'r', label="orbit params")

def plot_orbit_params_3D(ax, semia, e, i, O, w, f):
    # Plot 3D orbit in Cartesian coordinates
    vx_n, vy_n = integrate_orbit_3D(semia, e, i, O, w, f)
    ax.plot(vx_n, vy_n, 'r', label="orbit params")
    
def plot_orbit_params_polar(ax, semia, e, phase=0):
    # Plot orbit with parameters semia and e in polar coordinates
    vx_n, vy_n = integrate_orbit(semia, e, phase=phase)
    v_mag_n, alpha_n = img.cart2polar(vx_n, vy_n)
    inds = alpha_n.argsort()
    alpha_n = alpha_n[inds]
    v_mag_n = v_mag_n[inds]
    ax.plot(alpha_n, v_mag_n, 'r', label="orbit params")

def plot_orbit_params_polar_3D(ax, semia, e, i, O, w, f):
    # Plot 3D orbit in polar coordinates
    vx_n, vy_n = integrate_orbit_3D(semia, e, i, O, w, f)
    v_mag_n, alpha_n = img.cart2polar(vx_n, vy_n)
    inds = alpha_n.argsort()
    alpha_n = alpha_n[inds]
    v_mag_n = v_mag_n[inds]
    ax.plot(alpha_n, v_mag_n, 'r', label="orbit params")
        
def plot_Kep_v(ax):
    # Plot circles of Keplerian velocity at particular physical radii
    r = np.array([0.2, 0.64, 1.2, 2]) # Radii in solar radii
    v_mag_array = np.sqrt(G*WD_mass/(r*R_sol))*cms_to_kms # Convert r to kms
    # Add circles to the tomogram
    step = 0
    linestyles = ['--', '-.', ':' , '-']
    alpha = np.linspace(0, 2*pi, 100)
    for v_mag in v_mag_array:
        vx = v_mag*np.cos(alpha)
        vy = v_mag*np.sin(alpha)
        label = str(r[step]) + r' $R_\odot$'
        ax.plot(vx, vy, linestyle=linestyles[step], color='w', label=label)
        step += 1

