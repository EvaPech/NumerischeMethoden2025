import numpy as np
import pytest
from transit_model import create_flux_model, fit_transit

################################
# Test 1: Transit depth recovery
################################
#
# Test will be performed for different cases of transit
#
@pytest.mark.parametrize(
    "d_true, T_true, t1_true, sigma, tol",
    [
        # Baseline case 
        (0.01, 0.12, -0.06, 5e-4, 0.15),

        # Transit near left edge 
        (0.01, 0.12, -0.49, 5e-4, 0.15),

        # Transit near right edge 
        (0.01, 0.12,  0.37, 5e-4, 0.15),

        # Shorter transit duration 
        (0.01, 0.03, -0.06, 5e-4, 0.15),

        # Slightly higher noise 
        (0.01, 0.12, -0.06, 8e-4, 0.15),
    ],
)

def test_depth_recovery_within_tolerance(d_true, T_true, t1_true, sigma, tol):
    # This test checks whether the fitted transit depth is recovered
    # within a tolerance 'tol' of the true (injected) value
    # 
    #############################
    # 1) Create synthetic dataset
    #############################
    #
    # Define the time range around the transit
    t_min = -0.5
    t_max = 0.5
    dt = 0.01
    # Create a uniform time grid
    time = np.arange(t_min, t_max + 0.5*dt, dt)
 
    # Generate an ideal, noiseless transit light curve 
    flux_true = create_flux_model(time, T_true, d_true, t1_true)

    # Initialize a random number generator with a fixed seed
    # to make the test fully reproducible
    rng = np.random.default_rng(0)      
    # Generate Gaussian noise for each time point    
    noise = rng.normal(0.0, sigma, size=time.size)
    # Add noise to the ideal light curve to simulate observations
    flux = flux_true + noise

    ##############################
    # 2) Define trial grid for fitting
    ##############################
    # Grid should contain injected transit
    d_grid = np.linspace(0.0, 0.03, 61)       
    T_grid = np.linspace(0.01, 0.20, 61)    
    t1_step = dt                                # natural choice = sampling step
    
    ##############################
    # 3) Run fitting code
     ##############################
    best_d, best_T, best_t1, best_chi2, best_model, count = fit_transit(time, flux, sigma, d_grid, T_grid, t1_step)

    ##############################
    # 4) Assert depth accuracy
    ##############################
    
    # Compute the relative error between fitted and true depth
    rel_err = abs(best_d - d_true) / d_true
    
    # Assert that the fitted depth is accurate within 15%
    assert rel_err <= tol
    
################################
# Test 2: False positives with noise
################################
#
# Test will be performed for different levels of noise
# and different seeds for the random noise
#
@pytest.mark.parametrize(
    "seed, sigma",
    [
        (0, 5e-4),
        (1, 5e-4),
        (2, 5e-4),
        (0, 8e-4),
        (1, 8e-4),
        (2, 8e-4),
    ],
)
def test_pure_noise_no_deep_false_positive(seed, sigma):
    # This test checks whether the fitting functions find a transit in pure noise
    # The fitted transit depth must remain consistent with noise fluctuations.
    #
    #############################
    # 1) Create synthetic dataset
    #############################
    #
    # Define the time range around the transit
    # Time grid (same as in the other tests)
    t_min = -0.5
    t_max = 0.5
    dt = 0.01
    time = np.arange(t_min, t_max + 0.5 * dt, dt)

    # Initialize a random number generator with a changing seed in different tests
    rng = np.random.default_rng(seed)
    # Generate a flux with pure Gaussian noise around a flat baseline (no injected transit)
    flux = 1.0 + rng.normal(0.0, sigma, size=time.size)

    ##############################
    # 2) Define trial grid for fitting
    ##############################
    # Fitting grids must include d = 0 so "no transit" is possible
    d_grid = np.linspace(0.0, 0.03, 61)       
    T_grid = np.linspace(0.01, 0.20, 61)    
    t1_step = dt  # natural choice = sampling step

    ##############################
    # 3) Run fitting code
    ##############################
    best_d, best_T, best_t1, best_chi2, best_model, count = fit_transit(time, flux, sigma, d_grid, T_grid, t1_step)

    ##############################
    # 4) Assert: no false positives
    ##############################
    # Threshold scales with noise level
    # Everything deeper than about 6σ must not be claimed by the code if no real transit is present
    assert best_d <= 6.0 * sigma
    
################################
# Test 3: Physics sanity checks
################################
#
# The fiting function must not return unphysical parameters:
# - transit duration must be strictly positive
# - transit depth must be non-negative
#
@pytest.mark.parametrize(
    "seed, sigma, dmin, Tmin",
    [
        (0, 5e-4, -0.02, -0.10),   # includes negative depths and negative and zero durations
        (1, 5e-4, -0.10, -0.50),   # a more unphysical range
        (0, 8e-4, -0.02, -0.10),   # higher noise
        (2, 8e-4, -0.10, -0.50),   # higher noise
    ],
)
def test_best_fit_parameters_are_physical(seed, sigma, dmin, Tmin):

    #############################
    # 1) Create synthetic dataset
    #############################
    t_min = -0.5
    t_max = 0.5
    dt = 0.01
    time = np.arange(t_min, t_max + 0.5 * dt, dt)

    # Inject a real transit so the fitter has something to recover
    d_true = 0.01
    T_true = 0.12
    t1_true = -0.06


    # Generate an ideal, noiseless transit light curve 
    flux_true = create_flux_model(time, T_true, d_true, t1_true)

    # Initialize a random number generator with a fixed seed
    # to make the test fully reproducible
    rng = np.random.default_rng(seed)      
    # Generate Gaussian noise for each time point    
    noise = rng.normal(0.0, sigma, size=time.size)
    # Add noise to the ideal light curve to simulate observations
    flux = flux_true + noise

 

    ##############################
    # 2) Define trial grid for fitting
    ##############################
    # Intentionally include unphysical values in the grids
    d_grid = np.linspace(dmin, 0.03, 101)   # includes negative depths
    T_grid = np.linspace(Tmin, 0.20, 121)   # includes negative and zero durations
    t1_step = dt

    ##############################
    # 3) Run fitting code
    ##############################
    best_d, best_T, best_t1, best_chi2, best_model, count = fit_transit(time, flux, sigma, d_grid, T_grid, t1_step)
    
    ##############################
    # 4) Assert physical sanity
    ##############################
    assert best_d is not None # fitting function must have found a transit depth
    assert best_T is not None # fitting function must have found a transit time
    assert best_d >= 0.0 # must not be negative
    assert best_T > 0.0 # must be positive