# -*- coding: utf-8 -*-
import numpy as np
import utils.orbit_utils as orb

def Gauss(x, a, x0, sigma):
    '''Gaussian function
        
        Parameters:
        x -> x value
        a -> Height of Gaussian
        x0 -> Mean of Gaussian
        sigma -> Standard deviation of Gaussian
        
        Returns: y value of Gaussian
    '''
    
    return a* np.exp(-(x - x0)**2/(2*sigma**2))

def redchisqg(ydata, ymod, deg=6, sd=None):
    '''Return the reduced chi-squared error statistic for an arbitrary model
        
        Parameters:
        ydata -> Data
        ymod -> Model outputs
        deg -> Number of free parameters
        sd -> Uncertainties
        
        Returns: Chi-squared statistic
    '''
    
    chisq = np.sum(((ydata-ymod)/sd)**2)
    nu=ydata.size-1-deg
    
    return chisq/nu


def log_likelihood(sample_params, alpha, v_mag, v_mag_err):
    '''Find the log likelihood for an orbit fit to polar velocity data
        
        Parameters:
        sample_params -> Parameters for orbit fit
        alpha -> Angles in velocity space
        v_mag -> Velocity magnitude (for each alpha)
        v_mag_err -> Uncertainty on v_mag
        
        Returns: Log likelihood for orbit fit
    '''
    
    semia, e, i, O, w, f = sample_params # Orbit parameters
    
    # Get velocity magnitude values for given orbit parameters
    #model = orb.get_model_with_phase(alpha, semia, e, phase) # Without orbital angles
    model = orb.get_model_3D(alpha, semia, e, i, O, w, f) # With orbital angles
    
    sigma2 = v_mag_err**2 # Sigma squared
    
    return -0.5*np.sum((v_mag - model)**2/sigma2 + np.log(sigma2))

def log_prior(sample_params):
    '''Return uniform log prior for orbit parameters.
    
        Parameters: 
        sample_params -> Orbit parameters
        
        Returns:
        - Negative infinity if parameters are not within range of prior
        - Zero otherwise
    '''
    
    semia, e, i, O, w, f = sample_params # Orbit parameters
    
    # Set range of uniform prior
    if 0<semia<10 and 0<e<1 and -90<i<90 and -180<O<180 and -180<w<180 and -180<f<180:
        return 0.0
    return -np.inf

def log_probability(sample_params, alpha, v_mag, v_mag_err):
    '''Return log posterior probability from prior and likelihood.
        
        Parameters:
        sample_params -> Orbit parameters
        alpha -> Angles in velocity space
        v_mag -> Velocity magnitude (for each alpha)
        v_mag_err -> Uncertainty on v_mag
        
        Returns:
        - Negative infinity if log prior returns negative infinity
        - Log prior + log likelihood otherwise
    '''
    
    lp = log_prior(sample_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(sample_params, alpha, v_mag, v_mag_err)
