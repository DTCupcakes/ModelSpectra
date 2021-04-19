# -*- coding: utf-8 -*-
import numpy as np
import utils.orbit_utils as orb

def Gauss(x, a, x0, sigma):
    return a* np.exp(-(x - x0)**2/(2*sigma**2))

def log_likelihood(sample_params, alpha, v_mag, v_mag_err):
    semia, e, i, O, w, f = sample_params
    #model = orb.get_model_with_phase(alpha, semia, e, phase)
    model = orb.get_model_3D(alpha, semia, e, i, O, w, f)
    sigma2 = v_mag_err**2
    return -0.5*np.sum((v_mag - model)**2/sigma2 + np.log(sigma2))

def log_prior(sample_params):
    semia, e, phase = sample_params
    if 0 < semia < 10 and 0 < e < 1.0 and 0 < phase < 2*np.pi:
        return 0.0
    return -np.inf

def log_probability(sample_params, alpha, v_mag, v_mag_err):
    lp = log_prior(sample_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(sample_params, alpha, v_mag, v_mag_err)
