def create_lya_forest(params, skewers, fwhm, z=None, sampling=3.0, cgm_dict=None, seed=None):
    """
        Generate lya line forest at the specified metallicity, with the option to include CGM absorbers.

        Args:
            params (astropy table)
            skewers (astropy table)
            logZ (float)
            fwhm (float): in km/s
            z (float): redshift
            sampling (float): number of pixels per resolution element

            ::For incorporating CGM absorbers::
            cgm_dict (dictionary): containing parameters for the distribution of CGM absorbers.
            metal_dndz_func (function): used to compute dndz of CGM absorber and for drawing random W values. See create_metal_cgm().
            seed (int): random seed for drawing CGM absorbers.
    """

    # ~0.00014 sec to generate one skewer
    if z is None:
        z = params['z'][0]

    dvpix = fwhm/sampling

    # Size of skewers and pixel scale at sim resolution
    vside = params['VSIDE'][0] # box size in km/s
    Ng = params['Ng'][0]       # number of grids on each side of the box
    dvpix_hires = vside/Ng     # pixel scale of the box in km/s
    nskew =len(skewers)

    cosmo = FlatLambdaCDM(H0=100.0 * params['lit_h'][0], Om0=params['Om0'][0], Ob0=params['Ob0'][0])
    tau0, nh_bar = lya_tau0(z, cosmo=Planck15, X=0.76)
    # note that tau0 is obtained from the stronger blue line

    # Pad the skewer for the convolution
    npad = int(np.ceil((7.0*fwhm)/dvpix_hires)) ## v_metal = 0 and f_ratio = 1 is possible?
    v_pad = npad*dvpix_hires
    pad_tuple = ((0,0), (npad, npad))
    tau_igm = np.pad(tau0*skewers['TAU'].data, pad_tuple, 'wrap')

    oden_pad = np.pad(skewers['ODEN'].data, pad_tuple, 'wrap')
    T_pad = np.pad(skewers['T'].data, pad_tuple, 'wrap')
    v_los_pad = np.pad(skewers['VEL_Z'].data, pad_tuple, 'wrap')

    # Determine the velocity coverage including padding, etc.
    v_min = 0.0
    v_max = vside
    vel_pad = (v_min - v_pad) + np.arange(Ng + 2*npad)*dvpix_hires
    # For the high-resolution spectrum take the valid region and recenter about zero
    iobs_hires = (vel_pad >= v_min) & (vel_pad <= v_max)
    vel_hires = vel_pad[iobs_hires]
    nhires = vel_hires.size
    # For the observed spectrum take the valid region and recenter about zero
    vel_obs_pad = np.arange(vel_pad.min(),vel_pad.max(),dvpix)
    iobs = (vel_obs_pad >= v_min) & (vel_obs_pad <= v_max)
    vel_lores = vel_obs_pad[iobs]
    nlores = vel_lores.size

    # original tau array was the stronger blue transition. Now shift velocity array to get tau array for red transition
    tau_interp = interp1d(vel_pad, tau_igm, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)
    # tau_igm: total tau is the sum of the red and blue tau's

    # Now generate the CGM
    # if cgm_dict != None:
    #     # tau_cgm, logN_draws, b_draws, v_draws, W_2796_draws, iskew_abs, tau_draws = create_mgii_cgm(vel_pad, nskew, z, cgm_dict, rand=rand)
    #     tau_cgm, logN_draws, b_draws, v_draws, W_blue_draws, iskew_abs, tau_draws = create_metal_cgm(vel_pad, nskew, z, cgm_dict, metal_dndz_func, metal_ion=metal_ion, seed=seed)
    #
    #     # Only pass back the draws that reside in the final velocity coverage (as determined by vel_lores)
    #     ikeep = (v_draws > vel_lores.min()) & (v_draws < vel_lores.max())
    #     cgm_tuple = (logN_draws[ikeep], b_draws[ikeep], v_draws[ikeep], W_blue_draws[ikeep], iskew_abs[ikeep], tau_draws[ikeep, :])
    # else:
    tau_cgm = np.zeros_like(tau_igm)
    cgm_tuple = None

    # Compute the various fluxes with and without the CGM
    flux_tot_hires_pad = np.exp(-(tau_igm + tau_cgm))
    flux_igm_hires_pad = np.exp(-tau_igm)
    flux_cgm_hires_pad = np.exp(-tau_cgm)

    # Now smooth this and interpolate onto observational wavelength grid
    sigma_resolution = (fwhm/2.35483)/dvpix_hires  # fwhm = 2.3548 sigma
    flux_tot_sm = gaussian_filter1d(flux_tot_hires_pad, sigma_resolution, mode='mirror')
    flux_igm_sm = gaussian_filter1d(flux_igm_hires_pad, sigma_resolution, mode='mirror')
    flux_cgm_sm = gaussian_filter1d(flux_cgm_hires_pad, sigma_resolution, mode='mirror')

    flux_tot_interp = interp1d(vel_pad, flux_tot_sm, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)
    flux_igm_interp = interp1d(vel_pad, flux_igm_sm, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)
    flux_cgm_interp = interp1d(vel_pad, flux_cgm_sm, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)
    flux_tot_pad = flux_tot_interp(vel_obs_pad)
    flux_igm_pad = flux_igm_interp(vel_obs_pad)
    flux_cgm_pad = flux_cgm_interp(vel_obs_pad)

    # For the high-resolution spectrum take the valid region and recenter about zero
    flux_tot_hires = flux_tot_hires_pad[:,iobs_hires]
    flux_igm_hires = flux_igm_hires_pad[:,iobs_hires]
    flux_cgm_hires = flux_cgm_hires_pad[:,iobs_hires]

    oden = oden_pad[:,iobs_hires]
    T = T_pad[:,iobs_hires]
    v_los = v_los_pad[:,iobs_hires]

    # For the observed spectrum take the valid region and recenter about zero
    flux_tot_lores = flux_tot_pad[:,iobs]
    flux_igm_lores = flux_igm_pad[:,iobs]
    flux_cgm_lores = flux_cgm_pad[:,iobs]

    # Guarantee that nothing exceeds one due to roundoff error
    flux_tot_lores = np.clip(flux_tot_lores, None, 1.0)
    flux_igm_lores = np.clip(flux_igm_lores, None, 1.0)
    flux_cgm_lores = np.clip(flux_cgm_lores, None, 1.0)

    flux_tot_hires = np.clip(flux_tot_hires, None, 1.0)
    flux_igm_hires = np.clip(flux_igm_hires, None, 1.0)
    flux_cgm_hires = np.clip(flux_cgm_hires, None, 1.0)

    return vel_lores, (flux_tot_lores, flux_igm_lores, flux_cgm_lores), \
           vel_hires, (flux_tot_hires, flux_igm_hires, flux_cgm_hires), (oden, v_los, T), cgm_tuple
