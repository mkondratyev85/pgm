import subprocess

from view import Matplot
from model import Model
from largetime import Time


class PGM:
    def __init__(self, problem, figname=''):
        self.problem = problem
        dx = problem['width']/(problem['j_res']-1)
        dy = problem['height']/(problem['i_res']-1)

        label = subprocess.check_output(["git", "describe", "--always"])[:-1].decode('utf-8')

        self.view = Matplot(width = problem['width'],
                            height = problem['height'],
                            i_res = problem['i_res'],
                            j_res = problem['j_res'],
                            dx = dx,
                            dy = dy,
                            git = label,
                            figname = figname )

        self.model = Model(problem)
        self.maxT = Time(problem['MaxT'])

    def run(self):
        plot_step_interval = self.problem['plot_step_interval']
        while True:
            for iteration in self.model.make_step(maxT = self.maxT):
                print(iteration['step'])
                # if iteration['step']>50:
                #     self.model.gx_0 = 0
                #     self.model.gy_0 = 0
                if plot_step_interval:
                    if iteration['step'] % plot_step_interval:
                        continue
                self.view.plot12(iteration)
