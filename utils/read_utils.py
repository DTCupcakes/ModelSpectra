import numpy as np
import astropy.io.fits as fits

import utils.img_utils as img

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
Classes for simulated and observational data
'''
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

class particle_data:
    # Velocity data for a set of particles
    def __init__(self, file_list, sep_tstep=False):
        vx, vy = np.array([[],[]])
        for filename in file_list:
            data = ascii_file(filename)
            vx_i, vy_i = data.read_v()
            data.f.close()
            err_units(vx_i) # Check if velocities are in the right units
            vx = np.append(vx, vx_i)
            vy = np.append(vy, vy_i)
        if sep_tstep == False:
            # Don't separate particles by timstep
            vx, vy = np.hstack(vx), np.hstack(vy)
        self.vx, self.vy = vx, vy
        
        # Convert particle velocities to polar coordinates
        self.v_mag, self.alpha = img.cart2polar(vx, vy)
        
        self.rot_angle = 0 # Angle particle velocities have been rotated by
        
    def rotate(self, angle):
       # Rotate velocities by angle (degrees) anticlockwise
       print("Rotating velocities by", angle, "degrees")
       angle = angle*np.pi/180 # Convert to rad
       self.alpha = self.alpha + angle
       self.vx, self.vy = img.polar2cart(self.v_mag, self.alpha)
       self.rot_angle += angle
       
    def undo_rotate(self):
       print("Undoing all previous rotation")
       self.rotate(-self.rot_angle)     
        
class obs_2Dhist:
    # Read in a 2D histogram of observational data
    def __init__(self, filename, scale_per_pixel):
        # scale_per_pixel -> Velocity per pixel (in km/s)
        inpath = './obs_data/'
        print("Reading", filename)
        hdulist_map = fits.open(inpath + filename)
        v_data = hdulist_map[1].data
        self.data_cart = v_data
        origin = np.array([(len(v_data)-1)/2, (len(v_data[0])-1)/2])
        vx = np.linspace(0, len(v_data), num=len(v_data)) - origin[0]
        vy = np.linspace(0, len(v_data[0]), num=len(v_data[0])) - origin[1]
        vx *= scale_per_pixel
        vy *= scale_per_pixel
        self.vx, self.vy = vx, vy
        
        # Convert data to polar coordinates
        self.data_polar, self.vata.vy, bins=self.n_bins, ran_mag, self.alpha = img.obs_hist2D_to_polar(v_data, vx, vy)

'''
Functions for selecting files
'''
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
