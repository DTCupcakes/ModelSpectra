import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import map_coordinates

'''
Convert images to polar coordinates
'''
def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def reproject_image_into_polar(data, origin=None):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)
    r, theta = cart2polar(x, y)

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nx)
    theta_i = np.linspace(theta.min(), theta.max(), ny)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = polar2cart(r_grid, theta_grid)
    xi += origin[0] # We need to shift the origin back to 
    yi += origin[1] # back to the lower-left corner...
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)

    # Reproject the data
    zi = map_coordinates(data, coords, order=1)
    output = zi.reshape((nx, ny))
    return output, r_i, theta_i

def obs_hist2D_to_polar(data, x, y):
    data_polar, r_i, theta_i = reproject_image_into_polar(data)
    r_ran = np.sqrt(np.amax(x)**2 + np.amax(y)**2) # Range of real r values
    r = r_i * r_ran/np.amax(r_i)
    theta = theta_i
    data_polar = np.transpose(data_polar)
    return data_polar, r, theta

'''
Functions for altering 2D histogram data
'''
def blur(img):
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

'''
Find and plot histogram Gaussians
'''
def plt_Gaussian_n(n, v_mag, img):
    plt.plot(v_mag, img[n,:], label=str(n))
    #plt.plot(v_mag, mc_util.Gauss(v_mag, *popt), label='Gaussian')
    plt.legend()
    #plt.savefig('obs_data_angle_'+str(n)+'.png')
    plt.show()

