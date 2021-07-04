import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import signal
from scipy.optimize import minimize, curve_fit
import emcee
import corner

# Import external modules
import utils.read_utils as rd # Module for reading data
import utils.orbit_utils as orb # Module for finding and plotting orbits
import utils.img_utils as img # Module for processing images
import utils.emcee_utils as prms # Module for MCMC fits


def val_to_edges(x):
    '''Output histogram bin edges for an evenly-spaced 1D array.
    
        Parameters:
        x -> 1D evenly-spaced array 
        
        Returns:
        x_edges -> Histogram bin edges for x
    '''
    
    x_edges = np.array([]) # Histogram bin edges
    x_diff = (x[1] - x[0])/2 # Distance between bin value and bin edge
    
    # Append edge values to the left of the bin values
    for n in range(len(x)):
        x_edges = np.append(x_edges, x[n] - x_diff)
        
    # Append edge values to the right of the final bin value
    x_edges = np.append(x_edges, x[n] + x_diff)
    
    return x_edges

def edges_to_val(x_edges):
    '''Output evenly-spaced 1D array from the histogram bin edges.
        
        Parameters:
        x_edges -> Histogram bin edges
        
        Returns:
        x -> Evenly-spaced 1D array
    '''
    
    x = np.array([]) # Evenly-spaced 1D array values
    
    # Append bin values to the right of the bin values
    for n in range(len(x_edges)-1):
        x = np.append(x, 0.5*(x_edges[n]+x_edges[n+1]))
        
    return x

def plot_Gaussian(n, v_mag, img):
    '''Plot 1D cross-section of 2D histogram
    
        Parameters:
        n -> number of x-axis 2D histogram bin
        v_mag -> y-axis bin values
        img -> 2D histogram 
    '''
    
    plt.plot(v_mag, img[n,:], label=str(n)) # Plot histogram cross-section
    #plt.plot(v_mag, prms.Gauss(v_mag, *popt), label='Gaussian') # Plot best-fitting Gaussian for the histogram data
    plt.legend()
    #plt.savefig('obs_data_angle_'+str(n)+'.png')
    plt.show()

    
''' Classes for different types of plots created from particle velocity data '''

class Spectral_Line:
    '''Simulated spectral line with a 1D histogram'''
    
    def __init__(self, data):
        self.data = data # rd.particle_data object (particle velocity data)
        self.n_bins = 256 # Number of histogram bins
    
    def plot_angle(self, ax, angle):
        '''Produce spectral line from projecting velocity data along one axis
            
            Parameters:
            ax -> Figure axis (for plotting)
            angle -> Angle of projection axis clockwise from the positive y-axis
        '''
        
        print('Creating spectral line at',angle,' degrees') # Status message
        
        self.data.rotate(angle) # Rotate velocities anticlockwise by angle
        
        # Create histogram of projected velocities
        v_max = 1000 # Maximum range on histogram
        hist_fig, hist_ax = plt.subplots() # Plot the histogram separately
        hist, v_bins, patches = hist_ax.hist(self.data.vy, bins=self.n_bins, range=[-v_max, v_max])
        plt.close(hist_fig) # Make sure only a single plot is shown
        v_hist = edges_to_val(v_bins) # Get histogram bin values from bin edge values
        hist /= 100000 # Scale y-axis of histogram
        
        # Plot histogram on ax
        ax.plot(v_hist, hist)
        plt.xlim(-v_max, v_max)
        ax.axvline(x=0, linestyle='--') # Create dashed line at y=0
        max_WHT = 560 # Maximum velocity from WHT observations in 2006 (Manser et al., 2016)
        ax.axvline(x=max_WHT, linestyle='-.') # Create vertical lines at max_WHT
        ax.axvline(x=-max_WHT, linestyle='-.')
        ax.set_xlabel('Projected Velocity (km/s)')
        ax.set_ylabel('No. of particles (x$10^5$)')
        ax.set_title('Projection angle = ' + str(angle) + ' degrees')
        
class Variability_Plot:
    '''Plot ratio of redshift to blueshift in simulated spectral line'''
    
    def __init__(self, data):
        self.data = data # rd.particle_data object (particle velocity data)
        self.n_bins = 256 # Number of histogram bins
        v_max = 1000 # Maximum range on histogram
        
        # Determine y-axis values on redshift-blueshift plot
        shift = np.array([]) # y-axis values on redshift-blueshift plot
        for t in range(len(data.vx)): # Get values at each timestep
            fig_hist, axs_hist = plt.subplots(1, figsize=(10,10))
            hist, v_bins, patches = plt.hist(data.vx[t], bins=self.n_bins, range=[-v_max, v_max])
            plt.close(fig_hist) # Make sure only a single plot is shown
            v_hist = edges_to_val(v_bins) # Get histogram bin values from bin edge values
            shift = np.append(shift, np.sum(v_hist*hist))
            
        self.shift = shift
        
    def plot_variability(self, ax):
        '''Produce plot of redshift/blueshift of spectral line over time
            
            Parameters:
            ax -> Figure axis (for plotting)
        '''
        
        # Plot redshift/blueshift with phase (assuming data covers a single orbit)
        time = np.linspace(0, 1, num=len(data.vx)) # Phase
        ax.plot(time, self.shift, marker='o', linestyle='None')
        ax.set_xlabel('Orbital Phase')
        ax.set_ylabel('Blue-to-red ratio')

class Tomogram:
    '''Plot particle velocities in the x and y direction in a 2D histogram (tomogram)'''
    
    def __init__(self, data, obs=False):
        self.data = data # rd.particle_data object (particle velocity data)
        
        if obs == False: # Use simulated particle data
            # Create tomogram from array of particle velocities
            
            n_bins = 256 # Number of histogram bins
            
            # Create tomogram data (Cartesian)
            vx_max = 1000 # Max vx (and vy)
            fig_hist, axs_hist = plt.subplots(1, figsize=(10,10))
            hist2d_cart, vx_bins, vy_bins, mesh = plt.hist2d(data.vx, data.vy, bins=n_bins, range=[[-vx_max,vx_max],[-vx_max,vx_max]])
            plt.close(fig_hist) # Make sure only a single plot is shown
            hist2d_cart = np.flip(hist2d_cart)

            # Create tomogram data (polar)
            # v_mag -> Velocity magnitude
            # alpha -> Angle (in velocity space)
            self.v_mag_max = np.amax(self.data.v_mag) # Max v_mag
            fig_hist, axs_hist = plt.subplots(1, figsize=(10,10))
            hist2d_polar, alpha_bins, v_mag_bins, mesh = plt.hist2d(data.alpha, data.v_mag, bins=n_bins, range=[[-np.pi,np.pi],[0,self.v_mag_max]])
            plt.close(fig_hist) # Make sure only a single plot is shown
            hist2d_polar = np.flip(hist2d_polar)
            hist2d_polar = np.flip(hist2d_polar, axis=1)

        else: # Use observational data
            # Obtain bin edges from existing 2D histogram
            
            # Create tomogram data (Cartesian)
            vx_bins = val_to_edges(data.vx)
            vy_bins = val_to_edges(data.vy)
            hist2d_cart = data.data_cart
            
            # Create tomogram data (polar)
            v_mag_bins = val_to_edges(data.v_mag)
            alpha_bins = val_to_edges(data.alpha)
            hist2d_polar = data.data_polar
            
        # Define attributes for tomogram object
        self.vx_bins, self.vy_bins = vx_bins, vy_bins
        self.hist2d_cart = hist2d_cart
        self.alpha_bins, self.v_mag_bins = alpha_bins, v_mag_bins
        self.hist2d_polar = hist2d_polar
    
    def plot_data(self, ax, blur_hist=False):
        '''Plot Cartesian tomogram of particle velocity data
            
            Parameters:
            ax -> Figure axis (for plotting)
            blur_hist -> Add Gaussian convolution (blur) to reduce resolution (for simulated data)
        '''
        
        # Tomogram data (Cartesian)
        hist2d_cart = self.hist2d_cart
        hist2d_cart = np.transpose(hist2d_cart)
        
        # Add blur
        if blur_hist == True: 
            hist2d_cart = img.blur(hist2d_cart)
            
        # Plot tomogram on ax
        ax.pcolormesh(self.vx_bins, self.vy_bins, hist2d_cart)
        
    def plot_data_polar(self, ax, blur_hist=False):
        '''Plot polar tomogram of particle velocity data
            
            Parameters:
            ax -> Figure axis (for plotting)
            blur_hist -> Add Gaussian convolution (blur) to reduce resolution (for simulated data)
        '''
        
        # Tomogram data (polar)
        hist2d_polar = self.hist2d_polar
        hist2d_polar = np.transpose(hist2d_polar)
        
        # Add blur
        if blur_hist == True:
            hist2d_polar = img.blur(hist2d_polar)
        
        # Plot tomogram on ax
        ax.pcolormesh(self.alpha_bins, self.v_mag_bins, hist2d_polar)                


'''Load files with particle velocity data from command line'''
parser = argparse.ArgumentParser(description='Some files.',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('files',nargs='+',help='files with the appropriate particle data')
args = parser.parse_args() # Files with particle velocity data

def read_data(file_list, obs=False, sep_tstep=False):
    '''Read data from files in file_list and convert into appropriate format
        
        Parameters:
        file_list -> List of files from which to read data
        obs -> Read observational data files
        sep_tstep -> Separate particle data by timestep (i.e. by file)
        
        Returns:
        data -> Particle velocity data, either simulated (arrays) or observational (2D histogram)
        run_name -> Partial filename for saving plots of data
    '''
    
    # Import simulated particle data from filenames on command line
    if obs == False: 
        data = rd.particle_data(file_list, sep_tstep=sep_tstep) # Create arrays of particle velocity data
        run_name = file_list[0].split('/')[2] # Partial filename for creating plot filenames
       
    # Import observational data
    else:
        obs_filename = 'map10000_2.fits' # Observational data from Manser et al. (2016)
        scale_per_pixel = 5 # Scale of observational 2D histogram in km/s per pixel from Manser et al. (2016)
        data = rd.obs_2Dhist(filename=obs_filename, scale_per_pixel=scale_per_pixel) # Read in 2D histogram data from file
        run_name = 'obs' # Partial filename for creating plot filenames
        
    return data, run_name


''' Functions for fitting eccentric orbits to velocity data'''

def find_ellipse(data, obs=False, plot=False):
    '''Find centre of gas disc on 2D histogram in polar coordinates
        
        Parameters:
        data -> Particle velocity data
        obs -> Read observational data files
        plot -> Plot location of disc centre on tomogram
        
        Returns:
        alpha -> Array of angles in velocity space
        v_max -> Array of velocity magnitude with the highest tomogram value (for each alpha)
        v_max_err -> Array of uncertainties in v_max
        hist_max -> Array of highest tomogram value (for each alpha)
    '''
    
    # Create polar tomogram from particle velocity data
    tom = Tomogram(data, obs=obs)
    hist2d_polar = tom.hist2d_polar # 
    
    # Get velocity magnitudes (v_mag) and angles in velocity space (alpha) from particle data
    alpha, v_mag = edges_to_val(tom.alpha_bins), edges_to_val(tom.v_mag_bins)
    
    # Determine value of v_mag containing greatest number particles for each alpha
    v_max = np.array([]) # v_mag value with the highest tomogram value for each value of alpha
    v_max_err = np.array([]) # Uncertainty in v_max
    hist_max = np.array([]) # Highest tomogram value for each alpha
    for n in range(len(hist2d_polar)): # For each value of alpha
        alpha_col = hist2d_polar[n,:] # Tomogram values for single alpha value
        
        # Find max tomogram value
        hist_max_n = np.amax(alpha_col) # Max tomogram value
        hist_max = np.append(hist_max, hist_max_n)
        
        # Find associated v_mag value
        hist_max_arg = np.argmax(alpha_col) # Index of max tomogram value
        v_max_n = v_mag[-1]*hist_max_arg/len(alpha_col) # v_mag value of max tomogram value
        v_max = np.append(v_max, v_max_n)
        
        # Find mean and std for a Gaussian distribution fit to alpha_col
        mean = np.sum(v_mag*alpha_col)/np.sum(alpha_col)
        sigma = np.sqrt(np.sum((v_mag - v_max_n)**2)/np.sum(v_mag))
        #if n % 10 == 0: # Plot alpha_col
            #print(sigma)
            #plot_Gaussian(n, v_mag, hist2d_polar) # Plot data with Gaussian fit
        v_max_err = np.append(v_max_err, sigma) # Uncertainty in v_max
    
    if plot == True: # Plot v_max with uncertainty
        fig, axs = plt.subplots(1, figsize=(10,10))    
        tom.plot_data_polar(axs) # Plot tomogram data
        axs.errorbar(alpha, v_max, yerr=v_max_err, label='plot err') # Plot v_max
        plt.legend()
        plt.show()
    
    return alpha, v_max, v_max_err, hist_max

def get_ellipse_parameters(alpha, v_mag, v_mag_err):
    # Get values for the parameters of the ellipse
    nll = lambda *args: -prms.log_likelihood(*args) # Log likelihood function
    initial_guess = np.array([0.73, 0.54, 17, 0, 0, 0]) # Initial guess for parameters
    bnds = ((1e-8, 10),(0, 0.999),(17,17),(-360,360),(-360,360),(-360, 360)) # Bounds on the parameters
    params = minimize(nll, initial_guess, bounds=bnds, args=(alpha, v_mag, v_mag_err))
    semia, e, i, O, w, f = params.x
    print("Maximum likelihood estimates:")
    print("semia = {0:.3f}".format(semia))
    print("e = {0:.3f}".format(e))
    print("i = {0:.3f}".format(i))
    print("O = {0:.3f}".format(O))
    print("w = {0:.3f}".format(w))
    print("f = {0:.3f}".format(f))
    return params

def get_ellipse_uncertainties(params, alpha, v_mag, v_mag_err, plot_sampler_steps=False, corner_plot=False):
    pos = params.x + 1e-4*np.random.randn(32,6)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, prms.log_probability, args=(alpha, v_mag, v_mag_err))
    sampler.run_mcmc(pos, 5000, progress=True);
    
    if plot_sampler_steps == True:
        # Plot steps of emcee sampler
        fig, axes = plt.subplots(6, figsize=(10,7), sharex=True)
        samples = sampler.get_chain()
        labels = ["semia", "e", "i", "O", "w", "f"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:,:,i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
        axes[-1].set_xlabel("step number")
        print("Plot sampler steps")
        plt.savefig('./emcee_plots/WD_obs_setei_sampler_steps.png')
        plt.show()

    tau = sampler.get_autocorr_time()
    print("tau =",tau)

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    print("Shape of flat_samples:",flat_samples.shape)
    # Create corner plot
    if corner_plot == True:
        fig = corner.corner(flat_samples, labels=labels);
        plt.savefig('./emcee_plots/WD_obs_setei_corner_plot.png')
        plt.show()

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(labels[i],'=',params.x[i],'(+',q[1],', -',q[0],')')

'''
Commands
'''
# Switch options on/off
obs = True
blur_hist = True # Histogram blurring
polar = False # Switch between Cartesian and polar plotting
plot_fit = False # Plot polar maxima (for tomogram)
plot_orbit = False # Plot orbit with parameters semia, e, i, O, w, f
angle = 90 # Angle (clockwise) from positive y axis at which spectral line is plotted
plots = [4]
# 1 - Plot tomogram
# 2 - Plot spectral line
# 3 - Variability plot
# 4 - Plot tomogram comparison (observational, model)
# 5 - Plot tomogram eccentricity comparison

# Set the font (size) for plots
font = {'size' : 22}
matplotlib.rc('font', **font)

# Set destination for output plots
outpath = './plots/'
emcee_outpath = './emcee_plots/'

# Read in data
data, run_name = read_data(args.files, obs=obs)
#data.rotate(-0.25*np.pi*180/np.pi)

# Fit ellipse and plot
alpha, v_mag, v_mag_err, hist_max = find_ellipse(data, obs=obs)
params = get_ellipse_parameters(alpha, v_mag, v_mag_err)
semia, e, i, O, w, f = params.x

# Get reduced chi-squared for fit
v_mag_mod = orb.get_model_3D(alpha, semia, e , i, O, w, f) # Velocity magnitudes using the model
chisq = prms.redchisqg(v_mag, v_mag_mod, deg=6, sd=v_mag_err)
print('Chi_sq =',chisq)

#get_ellipse_uncertainties(params, alpha, v_mag, v_mag_err, plot_sampler_steps=True, corner_plot=True)

if 1 in plots: # Plot tomogram
    tom = Tomogram(data, obs=obs)
    fig, axs = plt.subplots(1, figsize=(10,10))
    if polar == True: # Plot in polar coordinates
        tom.plot_data_polar(axs, blur_hist=blur_hist)
        if plot_orbit == True:
            #orb.plot_orbit_params_polar(axs, semia, e, phase=phase)
            orb.plot_orbit_params_polar_3D(axs, semia, e, i, O, w, f) # Plot in 3D
        plt.xlim(-np.pi, np.pi)
        if plot_fit == True:
            axs.errorbar(alpha, v_mag, yerr=v_mag_err)
        plt.ylim(0, 800)
    else: # Plot in Cartesian coordinates
        tom.plot_data(axs, blur_hist=blur_hist)
        orb.plot_Kep_v(axs)
        if plot_orbit == True:
            #orb.plot_orbit_params(axs, semia, e, phase=phase)
            orb.plot_orbit_params_3D(axs, semia, e, i, O, w, f) # Plot in 3D
        plt.xlim(-950, 950)
        plt.ylim(-950, 950)
        axs.set_xlabel(r'$v_x$ (km/s)')
        axs.set_ylabel(r'$v_y$ (km/s)')
    axs.legend(fancybox=True, framealpha=0.4, loc='upper right', fontsize=16)
    #axs.axvline(x=135*np.pi/180, linestyle='-.', color='r')
    fname_suf = rd.create_filename_suffix(run_name, tomogram=True, polar=polar, plot_orbit=plot_orbit)
    #filename = 'tom_5_' + run_name[-11:-7] + '.png' #
    filename = 'tom_1.png'
    print('Writing tomogram to', filename) # Status message
    plt.savefig('./tom_movie/' + filename) #
    plt.show()
    plt.close()

if 2 in plots: # Plot spectral line
    spec_line = Spectral_Line(data)
    fig, axs = plt.subplots(1, figsize=(10,10))
    spec_line.plot_angle(axs, angle)
    fig.tight_layout()
    fname_suf = rd.create_filename_suffix(run_name, tomogram=False)
    filename = 'spec_line_' + fname_suf + '_' + str(angle) + '.png'
    print('Writing spectral line to', filename) # Status message
    plt.savefig(outpath + filename)
    plt.show()
    plt.close()

if 3 in plots: # Plot variability of Ca II emission lines
    data, run_name = read_data(args.files, obs=obs, sep_tstep=True)
    fig, axs = plt.subplots(1, figsize=(10,10))
    var_plot = Variability_Plot(data)
    var_plot.plot_variability(axs)
    fname_suf = rd.create_filename_suffix(run_name, tomogram=False)
    filename = 'var_' + fname_suf + '.png'
    print('Writing variability plot to', filename) # Status message
    plt.savefig(outpath + filename)
    plt.show()
    plt.close()

if 4 in plots: # Plot tomogram comparison
    # Plotted orbits will only be from a fit to a single set of data
    data_obs, run_name = read_data(args.files, obs=True)
    data_sim, run_name = read_data(args.files, obs=False)
    data_sim.rotate(-0.25*np.pi*180/np.pi)
    tom_obs = Tomogram(data_obs, obs=True)
    tom_sim = Tomogram(data_sim, obs=False)
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(24, 12), dpi=50)
    plt.subplots_adjust(wspace=0.05, hspace=0.05) # Set the spacing between axes
    if polar == True: # Plot in polar coordinates
        deg = np.arange(-120,181,60) # x-axis tick positions (degrees)
        deg_locs = deg*np.pi/180 # x-axis tick positions (radians)
        deg_labels = [] # x-axis tick labels (degrees)
        for d in deg:
            deg_labels.append(str(d))
        for ax in axs:
            ax.xaxis.set_major_locator(ticker.FixedLocator(deg_locs))
            ax.xaxis.set_major_formatter(ticker.FixedFormatter(deg_labels))
        tom_obs.plot_data_polar(axs[0])
        tom_sim.plot_data_polar(axs[1], blur_hist=blur_hist)
        if plot_orbit == True:
            orb.plot_orbit_params_polar(axs[0], semia, e, phase=phase)
            orb.plot_orbit_params_polar(axs[1], semia, e, phase=phase)
        plt.xlim(-np.pi, np.pi)
        if plot_fit == True:
            axs[0].errorbar(alpha, v_mag, yerr=v_mag_err)
        plt.ylim(0, 950)
        axs[0].set_xlabel(r'Angle $\alpha$ (degrees)')
        axs[1].set_xlabel(r'Angle $\alpha$ (degrees)')
        axs[0].set_ylabel(r'Velocity $|\mathbf{v}|$ (km/s)')
        #for ax in axs:
            #plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
    else: # Plot in Cartesian coordinates
        for ax in axs:
            ax.xaxis.set_major_locator(ticker.IndexLocator(base=250, offset=250))
        tom_obs.plot_data(axs[0])
        tom_sim.plot_data(axs[1], blur_hist=blur_hist)
        #orb.plot_Kep_v(axs[0])
        #orb.plot_Kep_v(axs[1])
        orb.plot_Kep_v_3D(axs[0])
        orb.plot_Kep_v_3D(axs[1])
        if plot_orbit == True:
            orb.plot_orbit_params(axs[0], semia, e, phase=phase)
            orb.plot_orbit_params(axs[1], semia, e, phase=phase)
        for ax in axs:
            ax.set_xlim(-950, 950)
            ax.set_ylim(-950, 950)
            ax.set_xlabel(r'$v_x$ (km/s)')
        axs[0].set_ylabel(r'$v_y$ (km/s)')
        plt.legend(fancybox=True, framealpha=0.4, ncol=2, loc='upper right')
    filename = 'tom_comp.png'
    print('Writing tomogram comparison to', filename) # Status message
    plt.savefig(outpath + filename, dpi=600)
    plt.show()
    plt.close()

if 5 in plots: # Plot tomogram comparison
    # Plotted orbits will only be from a fit to a single set of data
    data, tom = [], []
    ecc = [0.1, 0.3, 0.5, 0.7]
    for n in range(4):
        file_list = rd.find_files(args.files, ecc[n], 99)
        data_n, run_name = read_data(file_list, obs=False)
        tom_n = Tomogram(data_n, obs=False)
        data.append(data_n)
        tom.append(tom_n)
    fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(40, 10), dpi=50)
    plt.subplots_adjust(wspace=0.05, hspace=0.05) # Set the spacing between axes
    if polar == True: # Plot in polar coordinates
        for n in range(axs.size):
            axs.flat[n].xaxis.set_major_locator(ticker.FixedLocator([-3,-2,-1,0,1,2,3]))
            tom[n].plot_data_polar(ax, blur_hist=blur_hist)
            axs.flat[n].set_title('e =' + str(ecc[n]))
        plt.xlim(-np.pi, np.pi)
        plt.ylim(0, 950)
        for ax in axs:
            ax.set_xlabel(r'Angle $\alpha$ (rad)')
        axs[0].set_ylabel(r'Velocity $|\mathbf{v}|$ (km/s)')
        plt.setp(axs.get_xticklabels(), rotation=30, horizontalalignment='right')
    else: # Plot in Cartesian coordinates
        for n in range(axs.size):
            #axs.flat[n].xaxis.set_major_locator(ticker.IndexLocator(base=250, offset=250))
            tom[n].plot_data(axs.flat[n], blur_hist=blur_hist)
            orb.plot_Kep_v(axs.flat[n])
            axs.flat[n].set_xlim(-950, 950)
            axs.flat[n].set_ylim(-950, 950)
            axs.flat[n].set_title('e =' + str(ecc[n]))
        for ax in axs:
            ax.set_xlabel(r'$v_x$ (km/s)')
        axs[0].set_ylabel(r'$v_y$ (km/s)')
        plt.legend(fancybox=True, framealpha=0.4, ncol=2, loc='upper right')
    filename = 'tom_ecc_comp.png'
    print('Writing tomogram comparison to', filename) # Status message
    plt.savefig(outpath + filename, dpi=600)
    plt.show()
    plt.close()
