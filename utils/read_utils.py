import numpy as np
import astropy.io.fits as fits

import utils.img_utils as img
import utils.orbit_utils as orb

''' Functions to print error messages '''

def err_units(v):
    # If particle velocities are read in with the wrong units (should be in km/s)
    wrong_units = False
    if np.amax(v) > 10000:
        print("Particle velocities are larger than plot limits.")
        wrong_units = True
    if np.amax(v) < 10:
        print("Particle velocities are much smaller than plot limits.")
        wrong_units = True
    if wrong_units == True:
        print("Are you sure the velocities are in the right units?")
    return wrong_units

def err_no_files(file_array):
    # If there are no files to be read
    if len(file_array) == 0:
        print("File array is empty.")

''' Classes for simulated and observational data '''

class ascii_file:
    # A class for ascii files
    def __init__(self, filename):
        self.filename = filename
        self.f = open(filename, 'r')
        self.lines = self.f.readlines()
        
    def read_v(self):
    '''Read in the velocity data for each particle
    
        Returns:
        vx_i -> Particle velocities in the x direction
        vy_i -> Particle velocities in the y direction
    '''
    
        vx_i, vy_i = [], [] # Velocities in the x and y direction for each particle
        column_number = 8 # Column with the velocity data
        
        line_step = 0
        for x in self.lines:
            line_step += 1
            if line_step > 14:
                row = x.split()
                vx_i.append(float(row[column_number-2]))
                vy_i.append(float(row[column_number-1]))
                
        print('Reading ', self.filename) # Status message
        
        vx_i, vy_i = vx_i, vy_i
        vx_i, vy_i = np.array(vx_i), np.array(vy_i)
        
        return vx_i, vy_i

class particle_data:
    # Velocity data for a set of particles
    def __init__(self, file_list, sep_tstep=False):
    '''Parameters:
        file_list -> List of file names
        sep_tstep -> Separate particle data by timestep
    '''
    
        vx, vy = [], [] # Particle velocity data for all files (x and y direction)
        v_mag, alpha = [], [] # Particle velocity data for all files (angles and magnitudes)
        for filename in file_list:
            data = ascii_file(filename) # Create ascii file object
            vx_i, vy_i = data.read_v() # Read velocity data from ascii file
            data.f.close() # Close ascii file
            
            vx_i, vy_i = np.array(vx_i), np.array(vy_i)
            wrong_units = err_units(vx_i) # Check if velocities are in the right units
            if wrong_units == True: # Convert velocities from GR to kms
                vx_i = orb.GR_to_kms(vx_i)
                vy_i = orb.GR_to_kms(vy_i)
                
            # Convert particle velocities to polar coordinates
            v_mag_i, alpha_i = img.cart2polar(vx_i, vy_i)
            
            vx.append(vx_i)
            vy.append(vy_i)
            
            v_mag.append(v_mag_i)
            alpha.append(alpha_i)
            
        vx, vy = np.array(vx), np.array(vy)
        v_mag, alpha = np.array(v_mag), np.array(alpha)
        
        if sep_tstep == False: # Don't separate particles by timestep
            vx, vy = np.hstack(vx), np.hstack(vy)
            v_mag, alpha = np.hstack(v_mag), np.hstack(alpha)
        
        self.vx, self.vy = vx, vy
        self.v_mag, self.alpha = v_mag, alpha
        
        self.rot_angle = 0 # Angle particle velocities have been rotated by
        
    def rotate(self, angle):
    ''' Rotate particle velocities anticlockwise
    
        Parameters:
        angle -> Angle (degrees) to rotate particle velocities by
    '''
   
       print("Rotating velocities by", angle, "degrees")
    
       angle = angle*np.pi/180 # Convert to radians
        
       self.alpha = self.alpha + angle
       
       # Keep particle velocity angle alpha between -pi and pi
       step = 0
       for alpha in self.alpha:
           if alpha > np.pi:
               self.alpha[step] -= 2*np.pi
           step += 1
       
       # Get new particle velocities in Cartesian coordinates
       self.vx, self.vy = img.polar2cart(self.v_mag, self.alpha)
        
       # Update total particle rotation angle
       self.rot_angle += angle
       
    def undo_rotate(self):
       # Undo all previous rotation of particles
       print("Undoing all previous rotation")
       self.rotate(-self.rot_angle)     
        
class obs_2Dhist:
    # Read in a 2D histogram of observational particle velocity data
    
    def __init__(self, filename, scale_per_pixel):
        # scale_per_pixel -> Velocity per pixel (in km/s)
        
        # Read in file
        inpath = './obs_data/'
        print("Reading", filename)
        hdulist_map = fits.open(inpath + filename)
        v_data = hdulist_map[1].data
        self.data_cart = v_data
        
        # Determine velocities in 2D histogram bins
        origin = np.array([(len(v_data)-1)/2, (len(v_data[0])-1)/2])
        vx = np.linspace(0, len(v_data), num=len(v_data)) - origin[0]
        vy = np.linspace(0, len(v_data[0]), num=len(v_data[0])) - origin[1]
        vx *= scale_per_pixel # Velocities in the x direction
        vy *= scale_per_pixel # Velocities in the y direction
        self.vx, self.vy = vx, vy
        
        # Convert data to polar coordinates
        self.data_polar, self.v_mag, self.alpha = img.obs_hist2D_to_polar(v_data, vx, vy)

''' Functions for selecting files '''

def find_files(ascii_files, e, n_orbit):
    '''Sort files by eccentricity of asteroid orbit in simulation and orbit number
        
        Parameters:
        ascii_files -> List of file names
        e -> Eccentricity of asteroid orbit
        n_orbit -> Orbit number
    '''
    
    e_str = str(e)[2]
    n_orbit_str = str(n_orbit)
    
    print('Finding files for orbit',n_orbit_str,'at e=',e)
    
    # Find files
    file_list = []
    for f in ascii_files:
        if f[-13] == e_str and f[-9:-7] == n_orbit_str:
            file_list.append(f)
    err_no_files(file_list) # Return an error if the file list is empty
    
    return file_list

''' Function for creating filenames (for plots) '''

def create_filename_suffix(run_name, tomogram, polar=False, plot_orbit=False):
    filename_suffix = run_name # Number of simulated orbit (or 'obs')
    if tomogram == True: # Plot is a tomogram
        if polar == True: # Plot is in polar coordinates
            filename_suffix += '_polar'
        if plot_orbit == True: # Best fit orbit is plot over the top of tomogram
            filename_suffix += '_fit'
            
    return filename_suffix
