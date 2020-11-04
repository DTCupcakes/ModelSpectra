import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal
from scipy.optimize import minimize, curve_fit
import emcee
import corner

# Get filenames from command line
parser = argparse.ArgumentParser(description='Some files.',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('files',nargs='+',help='files with the appropriate particle data')
args = parser.parse_args()

# Set the font (size) for plots
#font = {'size' : 28}
#matplotlib.rc('font', **font)

'''
Constants and conversions
sol -> Solar units
'''
c_cgs = 3e10 # Speed of light
cms_to_kms = 1e-5

G_cgs = 6.67e-8 # Gravitational constant
R_sol_cgs = 6.9634e10 # Solar radius in cgs units
M_sol_cgs = 2.e33 # Solar mass in cgs units
WD_mass_sol = 0.705 # White dwarf mass in solar units
WD_mass_cgs = WD_mass_sol*M_sol_cgs # White dwarf mass in cgs units

semia_sol = 0.73 # Semi-major axis in solar units
semia_cgs = semia_sol*R_sol_cgs # Semi-major axis in cgs units

Manser_2016_angle = 95

e = 0.54 # Eccentricity
str_n = '90-99'

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

def err_no_files(file_array):
    if len(file_array) == 0:
        print("File array is empty.")

'''
Orbit integration functions
'''
def acc(x):
    # Get acc. from pos.
    r = np.sqrt(x[0]**2+x[1]**2)
    a = (-G_cgs*WD_mass_cgs/r**3)*x
    return a

def integrate_orbit(semia, e):
    # Integrate over an orbit with eccentricity e and return velocities
    n_points = 1000
    period = 2*np.pi*np.sqrt(semia**3/(G_cgs*WD_mass_cgs))
    dt = period/n_points # timestep
    x = np.array([semia*(1+e),0]) # Initial position
    v = np.array([0,np.sqrt(G_cgs*WD_mass_cgs*(1-e)/(semia*(1+e)))]) # Initial velocity
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
        vx_n, vy_n = integrate_orbit(semia_cgs, e)
        orbit = Hist(vx_n, vy_n)
        vx_n, vy_n = orbit.rotate(Manser_2016_angle) # Rotate plotted orbit
        vx_n = self.x_scale(vx_n)
        vy_n = self.y_scale(vy_n)
        return vx_n, vy_n
        
    def plt_tom(self, e, ax):
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
    
class Tomogram_Data (Tomogram):
    # Produce the tomogram from the 2D histogram of velocities
    def __init__(self, vx_array, vy_array):
        super().__init__(vx_array, vy_array)
        
    def plt_Kep_r(self, ax):
        # Plot circles of Keplerian velocity at particular physical radii
        radii = np.array([0.2, 0.64, 1.2, 2]) # Radii in solar radii
        v_radii = np.sqrt(G_cgs*WD_mass_cgs/(radii*R_sol_cgs))*cms_to_kms # Convert radius to kms
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
        self.plt_tom(e, ax)
        self.plt_Kep_r(ax)
        ax.imshow(self.img)

class Tomogram_PNG (Tomogram):
    # Produce tomogram from a PNG image
    def __init__(self, vx_array, vy_array, filename):
        self.tom_xran = 1700
        self.tom_yran = self.tom_xran
        self.vx_min = -850
        self.vy_min = self.vx_min
        self.img = mpimg.imread(filename)
        # Establish plot limits (from image pixels)
        self.xmin = -0.5
        self.xmax = self.xmin + len(self.img[0])
        self.ymax = self.xmin
        self.ymin = self.ymax + len(self.img[:,0])
        self.xran = self.xmax - self.xmin
        self.yran = self.ymax - self.ymin
        self.vx_per_pixel = self.tom_xran/self.xran
        self.vy_per_pixel = self.tom_yran/self.yran
        
    def plot(self, e, ax):
        self.plt_tom(e, ax)
        plt.imshow(self.img)

class ascii_file:
    # A class for ascii files
    def __init__(self, filename):
        self.filename = filename
        self.f = open(filename, 'r')
        self.lines = self.f.readlines()
        
    def read_v(self):
    # Read in the velocity data
        vx_i = []
        vy_i = []
        column_number = 8 # Column with the velocity data
        line_step = 0
        for x in self.lines:
            line_step += 1
            if line_step > 14:
                row = x.split()
                vx_i.append(float(row[column_number-2]))
                vy_i.append(float(row[column_number-1]))
        print('Reading ', self.filename) # Status message
        return vx_i, vy_i

def read_ascii(file_list):
    # Read ascii file for velocities
    vx = []
    vy = []
    for filename in file_list:
        data = ascii_file(filename)
        vx_i, vy_i = data.read_v()
        data.f.close()
        vx.append(vx_i)
        vy.append(vy_i)
    vx = np.array(vx)
    vy = np.array(vy)
    return vx, vy

def find_files(ascii_files, e, n_orbit):
    # Sorts files by eccentricity of asteroid orbit
    e_str = str(e)[2]
    n_orbit_str = str(n_orbit)
    print('Finding files for orbit',n_orbit_str,'at e=',e_str)
    file_list = []
    for f in file_list:
        if f[-13] == e_str and f[-9:-7] == n_orbit_str:
            file_list.append(f)
    err_no_files(file_list) # Return an error if the file list is empty
    return file_list

def finalise_plot(fig, filename):
    # Save figure and show
    fig.tight_layout()
    print('Writing to', filename) # Status message
    plt.savefig(filename)
    plt.show()
    plt.close()

def plt_spec_single(angle):
    # Plot a single spectral line from a particular angle
    vx, vy = read_ascii(args.files)
    spec = SpecLines(vx, vy)
    fig = plt.figure(1, figsize=[10,10])
    axs = fig.subplots()
    spec.plt_angle(angle, axs)
    filename = 'spec_line_' + str_n + '_' + str(angle) + '.png'
    finalise_plot(fig, filename)

def plt_tom_single():
    # Plot a single tomogram (matching SDSS J1228+1040)
    vx, vy = read_ascii(args.files)
    tom = Tomogram_Data(vx, vy)
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
            file_list = find_files(args.files, e, n_orbit)
            vx, vy = read_ascii(file_list)
            tom = Tomogram_Data(vx, vy)
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
        spec.plt_angle(angle_y[angle_step], sub_ax)
        angle_step +=1
    
def plt_spec_comp():
    # Plot tomogram comparisons of 4 different eccentricities
    fig = plt.figure(1, figsize=[10,10])
    axs = fig.subplots(4, 6, sharex=True, sharey=True)
    e_array = [0.1, 0.3, 0.5, 0.7]
    step = 0
    for ax in axs.flat:
        e = e_array[step]
        file_list = find_files(args.files, e)
        vx, vy = read_ascii(file_list)
        Ca_II = SpecLines(vx, vy)
        plt_spec_angles(90, Ca_II, ax)
        step += 1
    plt.xlabel('Projected velocity (km/s)')
    plt.ylabel(r'Particles ($\times 10^5$)')
    filename = 'spec_line_comp.png'
    finalise_plot(fig, filename)

def plt_tom_png():
    # Plot tomogram with data input as a PNG screenshot
    filename = '../Manser_2016_tomogram.PNG'
    fig = plt.figure(1, figsize=[10,10])
    axs = fig.subplots()
    tom = Tomogram_PNG(1, 1, filename)
    tom.plot(e, axs)
    plt.legend(fancybox=True, framealpha=0.4, loc='upper right')
    save_f_name = 'tomogram_Manser_2016.png'
    finalise_plot(fig, save_f_name)
    
def plt_var():
    # Reproduce variability of Ca II spectral lines
    vx, vy = read_ascii(args.files)
    file_no = len(args.files)
    shift = []
    for n in range(len(vx)):
        vel = Hist1D(vx[n], vy[n])
        hist, v_hist = vel.plt_hist(vx[n])
        shift.append(np.sum(v_hist*hist))
    shift = np.array(shift)

    # Plot the spectrum variability
    fig = plt.figure(1, figsize=[10,10])
    axs = fig.subplots()
    time = np.linspace(0, 2, num=file_no)
    plt.plot(time, shift, marker='o', linestyle='None')
    plt.xlabel('Orbital Phase')
    plt.ylabel('Blue-to-red ratio')
    filename = 'var_90-99.pdf'
    finalise_plot(fig, filename)

def exp_hist2D():
    vx, vy = read_ascii(args.files)
    hist2D = Hist2D(vx, vy)
    img = hist2D.plt_hist2D()
    filename = 'hist2D_WD_14_90-99.txt'
    print('Writing 2D histogram to',filename)
    np.save(filename, img)

def Gauss(x, a, x0, sigma):
    return a* np.exp(-(x - x0)**2/(2*sigma**2))

def get_model(alpha, semia, e):
    # Return velocity magnitude for a set of angles (alpha) in velocity space and a given semia and e
    semia_cgs = semia*R_sol_cgs
    vx, vy = integrate_orbit(semia_cgs, e)
    v_mag = np.sqrt(vx**2 + vy**2)
    v_angle = np.arctan2(vy, vx)
    return np.interp(alpha, v_angle, v_mag, period=2*np.pi)

def plt_hist2D_polar():
    # Plot 2D histogram of vx, vy in polar coordinates and find radial maxima
    vx, vy = read_ascii(args.files)
    data = Hist(vx, vy)
    img, v_angle_bins, v_mag_bins, mesh = plt.hist2d(data.v_angle, data.v_mag, bins=data.n_bins, range=[[-np.pi, np.pi],[0, 1500]]) # Plot 2D histogram
    #plt.show()
    plt.close()

    alpha = np.array([])
    v_mag = np.array([])
    for n in range(len(v_angle_bins)-1):
        alpha = np.append(alpha, 0.5*(v_angle_bins[n]+v_angle_bins[n+1])) # Angle list
        v_mag = np.append(v_mag, 0.5*(v_mag_bins[n]+v_mag_bins[n+1])) # Velocity magnitude list
    
    hist_max = np.amax(img, axis=1) # Maximum histogram value for each alpha
    hist_max_arg = np.argmax(img, axis=1) # Index of max hist val for each alpha
    v_max = v_mag_bins[-1]*hist_max_arg/len(img) # v_mag value of max hist val

    v_max_err = np.array([])
    for n in range(len(img)):
        sigma = np.sqrt(np.sum(img[n,:]*(v_mag - v_max[n])**2)/np.sum(img[n,:]))
        popt, pcov = curve_fit(Gauss, v_mag, img[n,:], p0=[hist_max[n],v_max[n],sigma])
        v_max_err = np.append(v_max_err, sigma) # Uncertainty in v_mag_max
        '''
        # Plot the Gaussian for a particular alpha with index n
        if n == 10:
            plt.plot(v_mag, img[n,:], label='data')
            plt.plot(v_mag, Gauss(v_mag, *popt), label='Gaussian')
            plt.legend()
            plt.show()
        '''
    # Plot v_max on top of the histogram    
    #plt.errorbar(alpha, v_max, yerr=v_max_err, label='plot err')
    #plt.legend()
    #plt.show()
    return alpha, v_max, v_max_err

def log_likelihood(sample_params, alpha, v_mag, v_mag_err):
    semia, e, log_f = sample_params
    model = get_model(alpha, semia, e)
    sigma2 = v_mag_err**2 + model**2 * np.exp(2*log_f)
    return -0.5*np.sum((v_mag - model)**2/sigma2 + np.log(sigma2))

def log_prior(sample_params):
    semia, e, log_f = sample_params
    if 0.1 < semia < 10.0 and 0 < e < 1.0 and -15.0 < log_f < 1.0:
        return 0.0
    return -np.inf

def log_probability(sample_params, alpha, v_mag, v_mag_err):
    lp = log_prior(sample_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(sample_params, alpha, v_mag, v_mag_err)

def get_ellipse_parameters():
    # Get values for the parameters of the ellipse
    alpha, v_mag, v_mag_err = plt_hist2D_polar() # Data
    
    nll = lambda *args: -log_likelihood(*args) # Log likelihood function
    initial_guess = np.array([0.73, 0.54, 0.0]) # Initial guess for parameters
    bnds = ((0.1, None), (0, 0.999), (None, None)) # Bounds on the parameters (semia, e, logf)
    params = minimize(nll, initial_guess, bounds=bnds, args=(alpha, v_mag, v_mag_err))
    semia, e, log_f = params.x
    print("Maximum likelihood estimates:")
    print("semia = {0:.3f}".format(semia))
    print("e = {0:.3f}".format(e))
    print("f = {0:.3f}".format(np.exp(log_f)))

    pos = params.x + 1e-4*np.random.randn(32,3)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(alpha, v_mag, v_mag_err))
    sampler.run_mcmc(pos, 5000, progress=True);
    
    fig, axes = plt.subplots(3, figsize=(10,7), sharex=True)
    samples = sampler.get_chain()
    labels = ["semia", "e", "log(f)"]
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
'''
# Any angles should be in radians
#plt_spec_single(90)
#plt_tom_single()
#plt_tom_png()
#plt_ecc_comp()
#plt_spec_comp()
#plt_var()
#exp_hist2D()
#plt_hist2D_polar()
get_ellipse_parameters()
