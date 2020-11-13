import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize, curve_fit
import emcee
import corner

import utils.read_utils as rd
import utils.orbit_utils as orb
import utils.img_utils as img
import utils.emcee_utils as prms

def val_to_edges(x):
    # Takes an evenly spaced 1D array and outputs 1D grid edges for the values
    x_edges = np.array([])
    x_diff = (x[1] - x[0])/2
    for n in range(len(x)):
        x_edges = np.append(x_edges, x[n] - x_diff)
    x_edges = np.append(x_edges, x[n] + x_diff)
    return x_edges

def edges_to_val(x_edges):
    # Take the grid edges for a 1D array and return the values
    x = np.array([])
    for n in range(len(x_edges)-1):
        x = np.append(x, 0.5*(x_edges[n]+x_edges[n+1]))
    return x

class Spectral_Line:
    # Convert particle data into spectral line data
    def __init__(self, data):
        self.data = data # rd.particle_data object
        self.n_bins = 256
    
    def plot_angle(self, ax, angle):
        # Produce spectral line angle clockwise from +ve y-axis
        print('Creating spectral line at ',angle,' degrees') # Status message
        self.data.rotate(angle) # Rotate velocities anticlockwise by angle
        v_max = 1000
        hist_fig, hist_ax = plt.subplots() # Plot the histogram separately
        hist, v_bins, patches = hist_ax.hist(self.data.vy, bins=self.n_bins, range=[-v_max, v_max])
        plt.close(hist_fig) # Make sure only a single plot is shown
        v_hist = edges_to_val(v_bins)
        hist /= 100000
        
        ax.plot(v_hist, hist)
        plt.xlim(-v_max, v_max)
        ax.axvline(x=0, linestyle='--')
        max_WHT = 560 # Maximum velocity from WHT observations in 2006
        ax.axvline(x=max_WHT, linestyle='-.')
        ax.axvline(x=-max_WHT, linestyle='-.')
        
class Variability_Plot:
    # Plot variability of Ca II spectral lines
    def __init__(self, data):
        self.data = data
        self.n_bins = 256
        v_max = 1000
        
        shift = np.array([])
        for t in range(data.vx):
            hist, v_bins, patches = plt.hist(data.vx[t], bins=self.n_bins, range=[-v_max, v_max])
            plt.close() # Make sure only a single plot is shown
            v_hist = edges_to_val(v_bins)
            shift.append(np.sum(v_hist*hist))
            
        self.shift = shift
        
    def plot_variability(self, ax):
        time = np.linspace(0, 1, num=len(data.vx))
        ax.plot(time, self.shift, marker='o', linestyle='None')
        ax.xlabel('Orbital Phase')
        ax.ylabel('Blue-to-red ratio')

class Tomogram:
    # Convert particle data into 2D histogram data
    def __init__(self, data, obs=False):
        self.data = data # rd.particle_data object
        
        if obs == False: # Use simulated particle data
            n_bins = 256
            
            # Create histogram data in Cartesian coordinates
            vx_max = 1500 # Max vx (and vy)
            hist2d_cart, vx_bins, vy_bins, mesh = plt.hist2d(data.vx, data.vy, bins=n_bins, range=[[-vx_max,vx_max],[-vx_max,vx_max]])
            # Get the data to line up with the right axes
            hist2d_cart = np.flip(hist2d_cart,1)
            hist2d_cart = np.transpose(hist2d_cart)
            
            # Create histogram data in polar coordinates
            v_mag_max = np.amax(self.data.v_mag)
            hist2d_polar, alpha_bins, v_mag_bins, mesh = plt.hist2d(data.alpha, data.v_mag, bins=n_bins, range=[[-np.pi,np.pi],[0,v_mag_max]])
            plt.close() # Remove histogram plots
            
        else: # Use observational data
            vx_bins = val_to_edges(data.vx)
            vy_bins = val_to_edges(data.vy)
            hist2d_cart = data.data_cart
            
            # Initialise polar coordinates
            v_mag_bins = val_to_edges(data.v_mag)
            alpha_bins = val_to_edges(data.alpha)
            hist2d_polar = data.data_polar
            
        self.vx_bins, self.vy_bins = vx_bins, vy_bins
        self.hist2d_cart = hist2d_cart
        self.alpha_bins, self.v_mag_bins = alpha_bins, v_mag_bins
        self.hist2d_polar = hist2d_polar
        
    
    def plot_data(self, ax, blur_hist=False):
        hist2d_cart = self.hist2d_cart
        if blur_hist == True:
            # Blur 2D histogram
            hist2d_cart = img.blur(hist2d_cart)
        ax.pcolormesh(self.vx_bins, self.vy_bins, hist2d_cart)
        
    def plot_data_polar(self, ax, blur_hist=False):
        hist2d_polar = self.hist2d_polar
        if blur_hist == True:
            # Blur 2D histogram
            hist2d_polar = img.blur(hist2d_polar)
        ax.pcolormesh(self.alpha_bins, self.v_mag_bins, hist2d_polar)                

def read_data(obs=False):
    if obs == False: # Import particle data from filenames on command line
        parser = argparse.ArgumentParser(description='Some files.',formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('files',nargs='+',help='files with the appropriate particle data')
        args = parser.parse_args()
        data = rd.particle_data(args.files, sep_tstep=sep_tstep)
    else: # Import observational data from Manser et al. (2016)
        obs_filename = 'map10000_2.fits'
        scale_per_pixel = 5 #km/s per pixel
        data = rd.obs_2Dhist(filename=obs_filename, scale_per_pixel=scale_per_pixel)
    return data

'''
Plotting functions
'''
def plot_specline_single(data, angle):
    # Plot a single spectral line from a particular angle
    spec_line = Spectral_Line(data)
    fig, axs = plt.subplots(1, figsize=(10,10))
    spec_line.plot_angle(axs, angle)
    filename = 'spec_line_' + str(angle) + '.png'
    fig.tight_layout()
    print('Writing to', filename) # Status message
    #plt.savefig(outpath + filename)
    plt.show()
    plt.close()
    
def plot_tom_single(data, semia, e, obs=False, polar=False, blur_hist=False):
    # Plot a single tomogram
    tom = Tomogram(data, obs=obs)
    fig, axs = plt.subplots(1, figsize=(10,10))
    if polar == True: # Plot in polar coordinates
        tom.plot_data_polar(axs, blur_hist=blur_hist)
        orb.plot_orbit_params_polar(axs, semia, e)
    else: # Plot in Cartesian coordinates
        tom.plot_data(axs, blur_hist=blur_hist)
        orb.plot_Kep_v(axs)
        orb.plot_orbit_params(axs, semia, e)
    plt.legend(fancybox=True, framealpha=0.4, loc='upper left')
    #filename = 'tomogram_' + str_n + '.png'
    #print('Writing to', filename) # Status message
    #plt.savefig(outpath + filename)
    plt.show()
    plt.close()
    
def plot_variability(data):
    # Plot variability of Ca II spectral lines
    var_plot = Variability_Plot(data)
    fig, axs = plt.subplots(1, figsize=(10,10))
    var_plot.plot_variability(axs)
    #filename = 'variability_plot_' + str_n + '.png'
    #print('Writing to', filename) # Status message
    #plt.savefig(outpath + filename)
    plt.show()
    plt.close()

'''
Functions for ellipse fitting
'''
def find_ellipse(data, obs=False, plot=False):
    tom = Tomogram(data, obs=obs)
    hist2d_polar = tom.hist2d_polar
    alpha, v_mag = edges_to_val(tom.alpha_bins), edges_to_val(tom.v_mag_bins)
    
    # Find maximum v_mag for each alpha
    v_max = np.array([])
    v_max_err = np.array([])
    for n in range(len(hist2d_polar)):
        #if n % 100 == 0:
            #img.plt_Gaussian_n(n, v_mag, hist2d_polar) # Plot Gaussian
        hist_max = np.amax(hist2d_polar[n,:]) # Maximum histogram value
        hist_max_arg = np.argmax(hist2d_polar[n,:]) # Index of max histogram value
        v_max_n = v_mag[-1]*hist_max_arg/len(hist2d_polar[n,:]) # v_mag value of max hist val
        v_max = np.append(v_max, v_max_n)
        mean = np.sum(v_mag*hist2d_polar[n,:])/np.sum(hist2d_polar[n,:])
        sigma = np.sqrt(np.sum(hist2d_polar[n,:]*(v_mag - mean)**2)/np.sum(hist2d_polar[n,:]))
        popt, pcov = curve_fit(prms.Gauss, v_mag, hist2d_polar[n,:], p0=[hist_max, mean, sigma])
        v_max_err = np.append(v_max_err, sigma) # Uncertainty in v_mag_max
    
    if plot == True: # Plot maximum v_mag
        fig, axs = plt.subplots(1, figsize=(10,10))    
        tom.plot_data_polar(axs)
        axs.errorbar(alpha, v_max, yerr=v_max_err, label='plot err')
        plt.legend()
        plt.show()
    
    return alpha, v_max, v_max_err

def get_ellipse_parameters(alpha, v_mag, v_mag_err):
    # Get values for the parameters of the ellipse
    nll = lambda *args: -prms.log_likelihood(*args) # Log likelihood function
    initial_guess = np.array([0.73, 0.54, np.pi]) # Initial guess for parameters
    bnds = ((0, 10),(0, 0.999),(0, 2*np.pi)) # Bounds on the parameters
    params = minimize(nll, initial_guess, bounds=bnds, args=(alpha, v_mag, v_mag_err))
    semia, e, phase = params.x
    print("Maximum likelihood estimates:")
    print("semia = {0:.3f}".format(semia))
    print("e = {0:.3f}".format(e))
    print("phase = {0:.3f}".format(phase))
    return params

def get_ellipse_uncertainties(params, alpha, v_mag, v_mag_err, plot_sampler_steps=False, corner_plot=False):
    pos = params.x + 1e-4*np.random.randn(32,3)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, prms.log_probability, args=(alpha, v_mag, v_mag_err))
    sampler.run_mcmc(pos, 5000, progress=True);
    
    if plot_sampler_steps == True:
        # Plot steps of emcee sampler
        fig, axes = plt.subplots(3, figsize=(10,7), sharex=True)
        samples = sampler.get_chain()
        labels = ["semia", "e", "phase"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:,:,i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
        axes[-1].set_xlabel("step number")
        plt.show()

    tau = sampler.get_autocorr_time()
    print("tau =",tau)

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    print("Shape of flat_samples:",flat_samples.shape)
    # Create corner plot
    if corner_plot == True:
        fig = corner.corner(flat_samples, labels=labels);
        plt.show()

'''
Commands
- All angles should be in degrees
'''
# Switch options on/off
obs = False
sep_tstep = False # Separate particles by timestep (for simulated data)
blur_hist = False # Histogram blurring
polar = False # Switch between Cartesian and polar plotting

# Set destination for output plots
outpath = './plots/'
emcee_outpath = './emcee_plots/'
str_n = '90-99'

# Read in data
data = read_data(obs=obs)

# Fit ellipse and plot
alpha, v_mag, v_mag_err = find_ellipse(data, obs=obs)
params = get_ellipse_parameters(alpha, v_mag, v_mag_err)
semia, e, phase = params.x
get_ellipse_uncertainties(params, alpha, v_mag, v_mag_err, plot_sampler_steps=True, corner_plot=True)
plot_tom_single(data, semia, e, obs=obs, polar=polar, blur_hist=blur_hist)

# Plot spectral line
angle = 90
plot_specline_single(data, angle)

# Plot variability (make sure sep_tstep=True)
plot_variability(data)
