import numpy as np

'''
Functions to print error messages
'''
def err_no_files(file_array):
    if len(file_array) == 0:
        print("File array is empty.")

'''
Classes and functions to read files
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
