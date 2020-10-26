import numpy as np
import os
from os.path import isdir
from C_2020_10_spiral_fitting.trace_and_fit_one_spiral import fit_one_spiral_arm

# VIP routines # download VIP at: https://github.com/VaChristiaens/VIP
import vip_hci
from vip_hci.fits import open_fits, write_fits
from vip_hci.var import frame_center

# plots
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        #'weight' : 'bold',
                'size'   : 16}
matplotlib.rc('font', **font)
plt.rcParams['image.cmap'] = 'CMRmap'
cmap_imgs = 'gist_heat' #or CMRmap

# misc.
pi = np.pi

# files and paths
filename = "hist2D_WD_14_90-99.txt"
inpath_data = './'
outpath = "products/" # define absolute or relative path where you want products to be written

# create folder if doesn't exist
if not isdir(outpath):
        os.system("mkdir "+outpath)  

# load image containing spirals
ori_img = np.loadtxt(inpath_data+filename)

# define the routine to convert from cartesian to polar - necessary to find radial maxima
from skimage.transform import warp
def polar2cart_warp(coords,center):
    theta = coords[:,0]*360./np.max(coords[:,0])
    rr = coords[:,1]
    cen_y, cen_x = center[0],center[1]
    x = rr * np.cos(np.deg2rad(theta)) + cen_x
    y = rr * np.sin(np.deg2rad(theta)) + cen_y
    # Now we write the new coords in the same array
    coords[:,0] = x
    coords[:,1] = y
    return coords

## first normalize before using warp:
max_img = np.amax(ori_img)
min_img = np.amin(ori_img)
tmp = ori_img-min_img
tmp = 2*tmp/(max_img-min_img)
tmp = tmp-1
print(np.amin(tmp))
print(np.amax(tmp))

## infer max r after warping
nr = int(tmp.shape[1]*np.sqrt(2)/2. + 1)
ntheta = 360

## warp
final_img = warp(tmp, polar2cart_warp, map_args={'center':frame_center(tmp)},output_shape=(nr,ntheta))
final_img = np.roll(final_img, -90, axis=1)

std1 = np.std(final_img)
med1 = np.median(final_img)
print(std1)
print("Median of the whole image: ", med1)

polar_coords = (np.arange(final_img.shape[0]),np.arange(final_img.shape[1])) # in this case the step is 1 pix and 1 deg along r and theta, making this definition trivial
print("number of radial steps in polar img: ",polar_coords[0].shape[0])

fwhm = 4             # beam size in px
plsc_ori = 0.049     # plate scale, in arcsec/px
dist = 157.3         # distance in pc; used to plot scale in au

# Global params for spiral tracing and deprojection (params for individual spirals are set before each relevant cell)

## spiral params
search_mode = 'simplex'                 # how is the best fit found? Choice between {'simplex','linsearch'}
clockwise = False                       # whether the spirals are clockwise or counter-clockwise

## Plot options
plot_format_paper = True # whether format of figures should be paper-ready
log_plot = False         # log scale or not? 
cbar_label = 'Tb (K)'    # Label for color bar
y_ax_lab= 'd'            # label for x axis
x_ax_lab= y_ax_lab       # label for y axis
scale_au = 200           # au (size of the scale shown in the image)
scale_as = 0.5           # arcsec (size of the scale shown in the image)

# PARAMETERS FOR SPIRAL TRACING AND FITTING - to be adapted for each spiral!

## Spiral parameters
tot_range_spi = 230            # total angle range subtended by the spiral in deg (max. allowed: 360)
th_spi = 220                   # rough PA of the root of the spiral (measured from positive y axis, counterclockwise)
A_in = 1.3                     # in arcsec     - Inner spiral guide parameter A of equation r = A*exp(sign*B*theta)
B_in = 0.28                    # in arcsec/rad - Inner spiral guide parameter B of equation r = A*exp(sign*B*theta)
A_out = 1.5                    # in arcsec     - Outer spiral guide parameter A of equation r = A*exp(sign*B*theta)
B_out = 0.28                   # in arcsec/rad - Outer spiral guide parameter B of equation r = A*exp(sign*B*theta)
sep_ang = 100                  # PA where the outer spiral guide fits better the spiral trace than the inner spiral
rmin_trace = 1.0               # in arcsec - the algo starts to report local radial max starting at this radius. Should be set to a bit less than radius of the root of the spiral 
rmax_trace = 3.5               # in arcsec - max radius to report local maxima. If None, it is set automatically to max size of the frame.
bad_angs = None                #e.g. [[0.89*pi,0.91*pi]] # Set this to a list of pairs of min and max angle values where the inferred trace should not be considered (e.g. spirals broken by shadows). Set to None or empty list if all points should be considered.

## Tracing options
iterate_eq_params = False      # Whether to stop the code after plotting the test spiral models. Allows to stop earlier and re-run if not satisfied with params used to isolate the local maxima of interest. 
thresh = med1                  # Min threshold to consider a local maximum as potential spiral trace. med1 = median of the whole image.

## Spiral fitting
fit_eq = 'poly'                # Spiral equation for the fit, choice between {'gen_archi', 'log', 'lin_archi', 'muto12','poly'}, and below, corresponding parameter estimate
param_estimate = np.array([1, 1, 1, 1, 1]) # first guess of parameters for simplex search (usually values do not matter - simplex finds its way, but number of params matter). For polynomial fits, the number of params provided here determined poly order.
npt_model = None               # Number of points in best-fit spiral model. If not provided, it will automatically pick 5x the number of azimuthal sections of the input simulation (i.e. typically 5*360=1800)
symmetric_plot = False         # Whether to plot the point-symmetric best-fit spiral 
find_uncertainty=True          # whether to compute uncertainties on best-fit params (a bit slower to set to True)
ratio_trace=None               # plot the spiral model for how much of the whole spiral trace (1=same length as spiral trace)

## Plot options
label_fig='a)'                  # label printed in top left corner
vmin = np.percentile(ori_img,5) # lower cut for image
vmax = np.percentile(ori_img,99)# upper cut for image
plot_fig = {1,2,3,4,5}          # Index of the figures to be plotted:
# 1 -> just show the density map;
# 2 -> show all local radial max;
# 3 -> show inner/outer spiral models used to isolate the good spiral traces
# 4 -> show the isolated trace of the spiral(s)
# 5 -> show best fit to the selected equation

# Formatting of input parameters provided above - do not modify
tot_range_spi = tot_range_spi*np.pi/180
th_spi = th_spi*np.pi/180
sep_ang = sep_ang*np.pi/180
A_in = int(A_in/plsc_ori)
A_out = int(A_out/plsc_ori) 
spiral_in_params = (int(A_in/plsc_ori), B_in)
spiral_out_params = (int(A_out/plsc_ori) , B_out)
rmin_trace = int(rmin_trace/plsc_ori)
rmax_trace = int(rmax_trace/plsc_ori)
ang_offset = 90                # offset between trigonometric and PA angle
weight_type = 'gaus_fit'       # Definition of error bars (weight of each point for the fit), choice between {'more_weight_in', 'uniform', 'individual', 'gaus_fit'}
pix_to_dist_factor = plsc_ori*dist  # pixel scale in au/px (arcsec/px and au/arcsec)
## only relevant in case of muto+12 fit (i.e. rafikov equation fit):
alpha = 1.5  
beta = 0.45 # Andrews+11 (referenced in Benisty+15)

# actual tracing and fitting
best_params, trace, best_model, best_model_pol = fit_one_spiral_arm(final_img, polar_coords, fwhm, clockwise, tot_range_spi, th_spi, spiral_in_params, spiral_out_params=(A_out,B_out), sep_ang=sep_ang, bad_angs=bad_angs, ori_img=ori_img, rmin_trace=rmin_trace, rmax_trace=rmax_trace, thresh=thresh, r_square=None, ang_offset=ang_offset, iterate_eq_params=iterate_eq_params, fit_eq=fit_eq, param_estimate=param_estimate, npt_model=npt_model, weight_type=weight_type, symmetric_plot=symmetric_plot, log_plot=log_plot, plot_fig=plot_fig, cbar_label=cbar_label, y_ax_lab=y_ax_lab, x_ax_lab=x_ax_lab, pix_to_dist_factor=pix_to_dist_factor, label=fit_eq, search_mode = 'simplex', dist=dist, outpath = outpath, find_uncertainty=find_uncertainty, label_fig=label_fig, font_sz=16, scale_as=scale_as, scale_au=scale_au, color_trace='c+', color_trace2='r+', ratio_trace=ratio_trace, vmin=vmin, vmax=vmax, deproj=False)

