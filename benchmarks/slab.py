
settings = {
    'width' :  1_000 * 1_000,
    'height' : 1_000 * 1_000,
    'j_res' :  91,
    'i_res' : 91,
    'gx_0' : 0,
    'gy_0' : 10,
    'p0cell' : 0,
    'pdensity': 10,
    'MaxT' : '100 Myr',
    'plot_step_interval' : 0,
    'stress_changes' : 'subgrid',
    'subgrid_relaxation' : 1.0,
    'advect_scheme' : 'Runge-Kutta 2nd order',
    'seed' : 0,
}

materials = [
    {'name': 'viso-elastic medium', 'rho': 1,  'eta': 1e24, 'mu': 1e20, 'C': 10, 'sinphi': 0.58778525229247314},
    {'name': 'viso-elastic slab', 'rho': 4000, 'eta': 1e27, 'mu': 1e10, 'C': 10, 'sinphi': 0.58778525229247314},
    {'name': 'viso-elastic slab', 'rho': 4000, 'eta': 1e27, 'mu': 1e10, 'C': 10, 'sinphi': 0.58778525229247314},
    {'name': 'viso-elastic medium', 'rho': 1,  'eta': 1e24, 'mu': 1e20, 'C': 10, 'sinphi': 0.58778525229247314},
]

boundaries = {
    'left_bound' : 'nosleep',
    'top_bound' : 'sleep',
    'right_bound' : 'sleep',
    'bottom_bound' : 'sleep',
}

moving_cells = [
]
