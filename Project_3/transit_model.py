import numpy as np

##########################################################
# Transit model: fitting simple box-shaped transit on a normalized baseline (= 1)
# Parameters:
# - time: time array
# - t_transit: transit duration T (days)
# - d_transit: transit depth d (relative flux drop, e.g. 0.01 = 1%)
# - t1: transit start time (days)
# Returns:
# - flux_model: model flux values at each time stamp
#########################################################

def create_flux_model(time, t_transit,d_transit,t1):
    t2=t1+t_transit
    flux_model=np.ones_like(time)
    in_window = (time >= t1) & (time <= t2)
    flux_model[in_window] = 1.0 - d_transit
    return flux_model

#########################################################
# Chi-square statistic for model comparison:
# chi2 = sum( ((data - model) / sigma)**2 )
# sigma can be a float (constant uncertainty) or an array (per-point uncertainties)
#########################################################
def chi2_test(flux, flux_model, sigma):
    return np.sum(((flux - flux_model) / sigma)**2)

#########################################################
# Fitting function
# Scans over transit duration T, transit depth d, and transit start time t1
# Keeps the parameter set with the minimum chi2
#########################################################

def fit_transit(time, flux, sigma, d_grid, T_grid, t1_step):
    # Define the allowed time window from the data
    t_min = time.min()
    t_max = time.max()

   # Initialize "best-fit" storage
    best_chi2 = np.inf
    best_d = None
    best_T = None
    best_t1 = None
    best_model = None

    # Count the models that are tested
    count = 0
    # Loop over candidate transit durations 
    for T in T_grid:
        # Physics test: negative or zero duration makes no sense
        if T <= 0:
            continue

        # For a given duration, restrict t1 so that the transit fully fits into the data window
        t1_min = t_min
        t1_max = t_max - T
        
        # Physics test: End of transit must be after start of transit
        if t1_max <= t1_min:
            continue

        # Build the grid of candidate start times t1
        # The additional + 0.5*t1_step ensures that the upper boundary is included
        # despite floating-point rounding and the half-open nature of np.arange.
        t1_grid = np.arange(t1_min, t1_max + 0.5 * t1_step, t1_step)

        # Loop over candidate transit depths 
        for d in d_grid:
            
            # Physics test: negative depth makes no sense
            if d < 0:
                continue

            # Loop over candidate start times (shifting the transit along the timeline)
            for t1 in t1_grid:

                count = count + 1
                # Generate the model light curve for this parameter set
                model = create_flux_model(time, T, d, t1)
                
                # Compute chi-square between observed flux and model
                chi2 = chi2_test(flux, model, sigma)

                # Update best-fit if this chi2 is smaller
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_d = d
                    best_T = T
                    best_t1 = t1
                    best_model = model
    # Return best-fit parameters and the best-fit model curve
    return best_d, best_T, best_t1, best_chi2, best_model, count

