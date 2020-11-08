import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize, curve_fit
from scipy.stats import shapiro
import astropy.io.fits as fits
import emcee
import corner

import utils.read_utils as rd
import utils.orbit_utils as orb
import utils.img_utils as img_util
import utils.emcee_utils as mc_util 

# Get filenames from command line
parser = argparse.ArgumentParser(description='Some files.',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('files',nargs='+',help='files with the appropriate particle data')
args = parser.parse_args()

# Import observational data from Manser et al. (2016)
inpath = './obs_data/'
hdulist_map = fits.open(inpath + 'map10000_2.fits')
velocity_data = hdulist_map[1].data
#plt.imshow(velocity_data)
#plt.show()

# Set the font (size) for plots
#font = {'size' : 28}
#matplotlib.rc('font', **font)

outpath = './plots/'
emcee_outpath = './emcee_plots/'
str_n = '90-99'

'''
Constants and conversions
'''
R_sol = orb.R_sol
G = orb.G
WD_mass = orb.WD_mass
cms_to_kms = orb.cms_to_kms

semia = 0.73*R_sol # Semi-major axis in cgs units
e = 0.54 # Eccentricity
Manser_2016_angle = 95

'''
Functions to print error messages
'''
def err_units(v):
    wrong_units = False
    if np.amax(v) > 10000:
        print("Particle velocities are larger than plot limits.")
        wrong_units = True
    if np.amax(v) < 10:
        print("Particle velocities are much smaller than plot limits.")
        wrong_units = True
    if wrong_units == True:
        print("Are you sure the velocities are in the right units?")

'''
Plot Classes
'''
class Hist:
    # General histogram (1D or 2D)
    def __init__(self, vx_array, vy_array):
        self.vx = np.hstack(vx_array) # hstack flattens arrays into 1D
        self.vy = np.hstack(vy_array)
        err_units(self.vx) # Check if velocities are in the right units
        self.v_mag = np.sqrt(self.vx**2 + self.vy**2)
        self.v_angle = np.arctan2(self.vy, self.vx)
        
        self.n_bins = 256

    def rotate(self, angle):
       # Rotate velocities by angle (degrees) anticlockwise
       angle = angle*np.pi/180
       v_mag = self.v_mag
       v_angle = self.v_angle + angle
       vx = v_mag*np.cos(v_angle)
       vy = v_mag*np.sin(v_angle)
       return vx, vy
   
class Hist1D(Hist):
    # General 1D histogram
    def __init__(self, vx_array, vy_array):
        super().__init__(vx_array, vy_array)
        # Set the edges of the histogram
        self.vmax = 1000
        self.vmin = -self.vmax
        self.range = [self.vmin, self.vmax]
    
    def plt_hist(self, v):
        # Generate histogram of velocities
        hist, v_bins, patches = plt.hist(v, bins=self.n_bins, range=self.range)
        plt.close() # Make sure only a single plot is shown
        v_hist = []
        for i in range(0, len(v_bins)-1):
            v_average = (v_bins[i+1] + v_bins[i])/2
            v_hist = np.append(v_hist, v_average)
            v_hist = np.array(v_hist)
        return hist, v_hist
        
class SpecLines (Hist1D):
    # A set of spectral lines (1D histograms)
    def __init__(self, vx_array, vy_array):
        super().__init__(vx_array,vy_array)
        self.max_WHT = 560 # Maximum velocity from WHT observations in 2006
        self.min_WHT = -self.max_WHT # Minimum velocity from WHT observations in 2006
    
    def plt_angle(self, angle, ax):
        # Produce spectral line angle clockwise from +ve y-axis
        print('Creating spectral line at ',angle,' degrees.') # Status message
        vx, vy = self.rotate(angle) # Rotate velocities anticlockwise by angle
        hist, v_hist = self.plt_hist(vy)
        hist = hist/100000

        plt.plot(v_hist, hist)
        plt.xlim(self.vmin, self.vmax)
        ax.axvline(x=0, linestyle='--')
        ax.axvline(x=self.max_WHT, linestyle='-.')
        ax.axvline(x=self.min_WHT, linestyle='-.')

class Hist2D (Hist):
    # Plot a 2D histogram
    def __init__(self, vx_array, vy_array):
        super().__init__(vx_array, vy_array)
        vx_min = -1500
        vx_max = -vx_min
        vy_min = vx_min
        vy_max = -vx_min
        self.v_range = [[vx_min,vx_max],[vy_min,vy_max]]
    
    def blur(self, img):
        # Create a 1D Gaussian to blur the histogram
        tom_pixels = 65 # Number of pixels on each side of the tomogram
        t = np.linspace(-10, 10, tom_pixels)
        sigma = 0.25 # Increasing this increases the blurring of the tomogram
        bump = np.exp(-0.5*t**2/sigma**2)
        bump /= np.trapz(bump) # Normalize the integral to 1
        # Make a 2D kernal out of it
        kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
        img_convolve = signal.fftconvolve(img, kernel[:, :], mode='same') # Convolve the tomogram
        return img_convolve
    
    def plt_hist2D(self):
        # Plot the 2D histogram
        self.vx, self.vy = self.rotate(Manser_2016_angle) # Rotate velocities to match Manser (2016)
        vx = self.vx
        vy = self.vy
        img, self.vx_bins, self.vy_bins, mesh = plt.hist2d(vx, vy, bins=self.n_bins, range=self.v_range)
        plt.close() # This stops the histogram from plotting as a separate figure
        img = np.flip(img,1)
        img = np.transpose(img)
        img = self.blur(img)
        return img
    
class Tomogram(Hist2D):
    def __init__(self, vx_array, vy_array):
        super().__init__(vx_array, vy_array)
        self.img = self.plt_hist2D()
        self.tom_xran = self.vx_bins[-1] - self.vx_bins[0]
        self.tom_yran = self.vy_bins[-1] - self.vy_bins[0]
        self.vx_min = self.vx_bins[0]
        self.vy_min = self.vy_bins[0]
        # Establish plot limits (from image pixels)
        self.xmin = -0.5
        self.xmax = self.xmin + len(self.img[0])
        self.ymax = self.xmin
        self.ymin = self.ymax + len(self.img[:,0])
        self.xran = self.xmax - self.xmin
        self.yran = self.ymax - self.ymin
        self.vx_per_pixel = self.tom_xran/self.xran
        self.vy_per_pixel = self.tom_yran/self.yran
        
    def x_scale(self, v):
        # Scale a velocity value into number of pixels
        x = (v - self.vx_min)/self.vx_per_pixel + self.xmin
        return x
    
    def y_scale(self, v):
        # Scale a velocity value into number of pixels
        y = (v - self.vy_min)/self.vy_per_pixel + self.ymin
        return y
    
    def plt_orbit(self, e):
        # Plot an orbit with a particular eccentricity (in velocity space)
        vx_n, vy_n = orb.integrate_orbit(semia, e)
        orbit = Hist(vx_n, vy_n)
        vx_n, vy_n = orbit.rotate(Manser_2016_angle) # Rotate plotted orbit
        vx_n = self.x_scale(vx_n)
        vy_n = self.y_scale(vy_n)
        return vx_n, vy_n
    
    def plt_Kep_r(self, ax):
        # Plot circles of Keplerian velocity at particular physical radii
        radii = np.array([0.2, 0.64, 1.2, 2]) # Radii in solar radii
        v_radii = np.sqrt(G*WD_mass/(radii*R_sol))*cms_to_kms # Convert radius to kms
        # Add circles to the tomogram
        step = 0
        linestyles = ['--', '-.', ':' , '-']
        theta = np.linspace(0, 2*np.pi, 100)
        for r in v_radii:
            x1 = self.x_scale(r*np.cos(theta))
            x2 = self.y_scale(r*np.sin(theta))
            label = str(radii[step]) + r' $R_\odot$'
            ax.plot(x1, x2, linestyle=linestyles[step], color='w', label=label)
            step += 1
        
    def plot(self, e, ax):
        # Plot the tomogram
        labels = np.arange(-1500, 1501, 250)
        self.x_locs = self.x_scale(labels)
        self.y_locs = self.y_scale(labels)
        self.vx_plot_max = 850
        self.vx_plot_min = -self.vx_plot_max
        self.vy_plot_max = self.vx_plot_max
        self.vy_plot_min = -self.vx_plot_max
        
        plt.xticks(self.x_locs, labels)
        plt.yticks(self.y_locs, labels)
        plt.xlim(self.x_scale(self.vx_plot_min), self.x_scale(self.vx_plot_max))
        plt.ylim(self.y_scale(self.vy_plot_min), self.y_scale(self.vy_plot_max))
        plt.xlabel(r'$v_x$ (km/s)')
        plt.ylabel(r'$v_y$ (km/s)')
        
        vx_orbit, vy_orbit = self.plt_orbit(e)
        ax.plot(vx_orbit, vy_orbit, 'r', label='e='+str(e))
        self.plt_Kep_r(ax)
        ax.imshow(self.img)        

'''
Plotting functions
'''
def finalise_plot(fig, filename):
    # Save figure and show
    fig.tight_layout()
    print('Writing to', filename) # Status message
    #plt.savefig(outpath + filename)
    plt.show()
    plt.close()

def plt_spec_single(angle):
    # Plot a single spectral line from a particular angle
    vx, vy = rd.read_ascii(args.files)
    spec = SpecLines(vx, vy)
    fig = plt.figure(1, figsize=[10,10])
    axs = fig.subplots()
    spec.plt_angle(angle, axs)
    filename = 'spec_line_' + str_n + '_' + str(angle) + '.png'
    finalise_plot(fig, filename)

def plt_tom_single():
    # Plot a single tomogram (matching SDSS J1228+1040)
    vx, vy = rd.read_ascii(args.files)
    tom = Tomogram(vx, vy)
    fig = plt.figure(1, figsize=[10,10])
    axs = fig.subplots()
    tom.plot(e, axs)
    plt.legend(fancybox=True, framealpha=0.4, loc='upper left')
    filename = 'tomogram_' + str_n + '.png'
    finalise_plot(fig, filename)

def plt_ecc_comp():
    # Plot tomogram comparisons of 4 different eccentricities
    fig = plt.figure(1, figsize=[10,10])
    axs = fig.subplots(4, 4, sharex=True, sharey=True)
    e_array = [0.1, 0.3, 0.5, 0.7]
    n_orbit_array = [10, 20, 50, 60]
    e_step = 0
    for ax in axs.flat:
        e = e_array[e_step]
        n_orbit_step = 0
        for sub_ax in ax:
            n_orbit = n_orbit_array[n_orbit_step]
            file_list = rd.find_files(args.files, e, n_orbit)
            vx, vy = rd.read_ascii(file_list)
            tom = Tomogram(vx, vy)
            tom.plot(e, sub_ax)
            n_orbit_step += 1
        e_step += 1
    plt.legend(fancybox=True, framealpha=0.4, loc='upper left')
    filename = 'tom_comp.png'
    finalise_plot(fig, filename)
    
def plt_spec_angles(angle_diff, spec, ax):
    # Produce a set of spectral lines separated by angle_diff (in degrees)
    nsteps = int(360/angle_diff + 1)
    angle_y_deg = np.linspace(0, 360, nsteps)
    angle_step = 0
    for sub_ax in ax:
        spec.plt_angle(angle_y_deg[angle_step], sub_ax)
        angle_step +=1
    
def plt_spec_comp():
    # Plot tomogram comparisons of 4 different eccentricities
    fig = plt.figure(1, figsize=[10,10])
    axs = fig.subplots(4, 6, sharex=True, sharey=True)
    e_array = [0.1, 0.3, 0.5, 0.7]
    step = 0
    for ax in axs.flat:
        e = e_array[step]
        file_list = rd.find_files(args.files, e)
        vx, vy = rd.read_ascii(file_list)
        Ca_II = SpecLines(vx, vy)
        plt_spec_angles(90, Ca_II, ax)
        step += 1
    plt.xlabel('Projected velocity (km/s)')
    plt.ylabel(r'Particles ($\times 10^5$)')
    filename = 'spec_line_comp.png'
    finalise_plot(fig, filename)
    
def plt_var():
    # Reproduce variability of Ca II spectral lines
    vx, vy = rd.read_ascii(args.files)
    file_no = len(args.files)
    shift = []
    for n in range(len(vx)):
        vel = Hist1D(vx[n], vy[n])
        hist, v_hist = vel.plt_hist(vx[n])
        shift.append(np.sum(v_hist*hist))
    shift = np.array(shift)

    # Plot the spectrum variability
    fig = plt.figure(1, figsize=[10,10])
    time = np.linspace(0, 2, num=file_no)
    plt.plot(time, shift, marker='o', linestyle='None')
    plt.xlabel('Orbital Phase')
    plt.ylabel('Blue-to-red ratio')
    filename = 'var_90-99.pdf'
    finalise_plot(fig, filename)

def plt_hist2D_polar():
    # Plot 2D histogram of vx, vy in polar coordinates and find radial maxima
    vx, vy = rd.read_ascii(args.files)
    data = Hist(vx, vy)
    img, v_angle_bins, v_mag_bins, mesh = plt.hist2d(data.v_angle, data.v_mag, bins=data.n_bins, range=[[-np.pi, np.pi],[0, 1500]]) # Plot 2D histogram
    plt.close()

    alpha = np.array([])
    v_mag = np.array([])
    for n in range(len(v_angle_bins)-1):
        alpha = np.append(alpha, 0.5*(v_angle_bins[n]+v_angle_bins[n+1])) # Angle list
        v_mag = np.append(v_mag, 0.5*(v_mag_bins[n]+v_mag_bins[n+1])) # Velocity magnitude list
    return img, alpha, v_mag
   
def find_ellipse(obs_data=False):
    if obs_data == True: # Determines if image is from simulated or observational data
        img, v_mag, alpha = img_util.reproject_image_into_polar(velocity_data)
        img = np.transpose(img)
    else:
        img, alpha, v_mag = plt_hist2D_polar()
    
    v_max = np.array([])
    v_max_err = np.array([])
    for n in range(len(img)):
        #if n % 100 == 0:
            #img_util.plt_Gaussian_n(n, v_mag, img) # Plot Gaussian
        hist_max = np.amax(img[n,:]) # Maximum histogram value
        hist_max_arg = np.argmax(img[n,:]) # Index of max histogram value
        v_max_n = v_mag[-1]*hist_max_arg/len(img[n,:]) # v_mag value of max hist val
        v_max = np.append(v_max, v_max_n)
        mean = np.sum(v_mag*img[n,:])/np.sum(img[n,:])
        sigma = np.sqrt(np.sum(img[n,:]*(v_mag - mean)**2)/np.sum(img[n,:]))
        popt, pcov = curve_fit(mc_util.Gauss, v_mag, img[n,:], p0=[hist_max, mean, sigma])
        v_max_err = np.append(v_max_err, sigma) # Uncertainty in v_mag_max

    # Plot v_max on top of the histogram
    if obs_data == True:
        img = np.transpose(img)
    plt.imshow(img)
    if obs_data == True:
        plt.ylim(np.flip(plt.ylim()))
    plt.errorbar(alpha, v_max, yerr=v_max_err, label='plot err')
    plt.legend()
    plt.show()
    return alpha, v_max, v_max_err

def get_ellipse_parameters(obs_data=False):
    # Get values for the parameters of the ellipse
    alpha, v_mag, v_mag_err = find_ellipse(obs_data=obs_data) # Data
    
    nll = lambda *args: -mc_util.log_likelihood(*args) # Log likelihood function
    initial_guess = np.array([0.73, 0.54]) # Initial guess for parameters
    bnds = ((0.1, None), (0, 0.999)) # Bounds on the parameters (semia, e, logf)
    params = minimize(nll, initial_guess, bounds=bnds, args=(alpha, v_mag, v_mag_err))
    semia, e = params.x
    print("Maximum likelihood estimates:")
    print("semia = {0:.3f}".format(semia))
    print("e = {0:.3f}".format(e))

    pos = params.x + 1e-4*np.random.randn(32,2)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, mc_util.log_probability, args=(alpha, v_mag, v_mag_err))
    sampler.run_mcmc(pos, 5000, progress=True);
    
    fig, axes = plt.subplots(2, figsize=(10,7), sharex=True)
    samples = sampler.get_chain()
    labels = ["semia", "e"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:,:,i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        #ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.show()

    tau = sampler.get_autocorr_time()
    print(tau)

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    print(flat_samples.shape)

    fig = corner.corner(flat_samples, labels=labels);
    plt.show()
    
    angle = np.linspace(-np.pi, np.pi)
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        plt.plot(angle, np.dot(np.vander(angle, 2), sample[:2]), "C1", alpha=0.1)
    plt.errorbar(alpha, v_mag, yerr=v_mag_err, fmt=".k", capsize=0)
    plt.legend(fontsize=14)
    plt.show()

'''
Commands
- All angles should be in degrees
- obs_data=True if using observational data (False if not)
'''
obs_data = True
#plt_spec_single(90)
#plt_tom_single()
#plt_ecc_comp()
#plt_spec_comp()
#plt_var()
#find_ellipse(obs_data=obs_data)
get_ellipse_parameters(obs_data=obs_data)
