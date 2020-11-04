# -*- coding: utf-8 -*-
import numpy as np
import orbit_utils as orb

def Gauss(x, a, x0, sigma):
    return a* np.exp(-(x - x0)**2/(2*sigma**2))

def log_likelihood(sample_params, alpha, v_mag, v_mag_err):
    semia, e, log_f = sample_params
    model = orb.get_model(alpha, semia, e)
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
