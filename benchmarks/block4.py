
settings = {
    'width' :  500 * 1_000,
    'height' : 500 * 1_000,
    'j_res' :  51,
    'i_res' : 51,
    'gx_0' : 0,
    'gy_0' : 10,
    'p0cell' : 0,
    'pdensity': 10,
    'MaxT' : '10 Myr',
    'plot_step_interval' : 0,
    'stress_changes' : 'subgrid',
    'subgrid_relaxation' : 1.0,
    'advect_scheme' : 'Runge-Kutta 2nd order',
    'seed' : 0,

}

materials = [
    {'name': 'mobile wall', 'rho': 2520, 'eta': 10**27, 'mu': 8*10**10, 'C': 10, 'sinphi': 0},
    {'name': 'sand',        'rho': 1560, 'eta': 10**10, 'mu': 10**6, 'C': 10, 'sinphi': 0.58778525229247314},
]

boundaries = {
    'left_bound' : 'sleep',
    'top_bound' : 'sleep',
    'right_bound' : 'sleep',
    'bottom_bound' : 'sleep',
}

moving_cells = [
]
