
settings = {
    'width' :  1_000 * 1_000,
    'height' : 1_000 * 1_000,
    'j_res' :  41,
    'i_res' : 41,
    'gx_0' : 0,
    'gy_0' : 0,
    'p0cell' : 0,
    'pdensity': 5,
    'MaxT' : '10 Myr',
    'plot_step_interval' : 0,
}

materials = [
    {'name': 'viso-elastic medium', 'rho': 1, 'eta': 1000000000000000000000000, 'mu': 100000000000000000000, 'C': 10, 'sinphi': 0.58778525229247314},
    {'name': 'viso-elastic medium', 'rho': 1, 'eta': 1000000000000000000000000, 'mu': 100000000000000000000, 'C': 10, 'sinphi': 0.58778525229247314},
    {'name': 'viso-elastic slab', 'rho': 4000, 'eta': 1000000000000000000000000000, 'mu': 10000000000, 'C': 10, 'sinphi': 0.58778525229247314},
    {'name': 'viso-elastic medium', 'rho': 1, 'eta': 1000000000000000000000000, 'mu': 100000000000000000000, 'C': 10, 'sinphi': 0.58778525229247314},
]

boundaries = {
    'left_bound' : 'nosleep',
    'top_bound' : 'sleep',
    'right_bound' : 'sleep',
    'bottom_bound' : 'sleep',
}

moving_cells = [
]

