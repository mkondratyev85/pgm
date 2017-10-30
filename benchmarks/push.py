
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
    {'name': 'mobile wall', 'rho': 2520, 'eta': 10000000000000000000000, 'mu': 10000000000000000, 'C': 10, 'sinphi': 0},
    {'name': 'sand', 'rho': 1560, 'eta': 1000000000, 'mu': 1000000, 'C': 10, 'sinphi': 0.58778525229247314},
    {'name': 'sticky air', 'rho': 1, 'eta': 100, 'mu': 1000000, 'C': 10, 'sinphi': 0},
]

boundaries = {
    'left_bound' : 'sleep',
    'top_bound' : 'sleep',
    'right_bound' : 'nosleep',
    'bottom_bound' : 'sleep',
}

moving_cells = [
]

