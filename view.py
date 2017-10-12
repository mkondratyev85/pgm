import numpy as np
import matplotlib.pylab as plt

class Matplot(object):
    """ Draw set of grid as plots and write to file via Matplotlib """

    def __init__(self, **kwargs):
        defaults = {
                   'figsize' : (30,20),
                   'width' : None,
                   'height' : None,
                   'j_res' : None,
                   'i_res' : None,
                   'dx' : None,
                   'dy' : None,
                   'git' : None,
                   'figname' : None,
        }
        self.__dict__.update(defaults)
        for key in kwargs:
            if key not in self.__dict__:
                raise KeyError( f"Wrong parameter: {key}")
        self.__dict__.update(kwargs)

        for key in self.__dict__:
            if self.__dict__[key] is None:
                raise ValueError (f'{key} parameter must be set')

        j = np.linspace(0,self.j_res-1,self.j_res).astype('int')
        i = np.linspace(0,self.i_res-1,self.i_res).astype('int')
        self.jj,self.ii  = np.meshgrid(j,i)

    def plot12(self, parameters):
        """ Make 12 plots on a list """
        plt.clf()
        fig = plt.figure(figsize = self.figsize)

        subtitle = f'Model size: {self.width/1000} km x {self.height/1000} ' +\
                   f'{self.j_res} x {self.i_res}, dx={self.dx}m, dy={self.dy}m.' +\
                   f'Current Time: {parameters["T"]}, Step: {parameters["step"]} ' +\
                   f'git: {self.git}'

        plt.suptitle(subtitle)

        plt.subplot(3,4,1)
        self._plot_visocity(parameters)

        things_to_plot = ['Vx', 'Vy', 'e_xx', 'e_xy', 's_xx', 's_xy', 'mu_n', 'w' ]
        for i,array in enumerate(things_to_plot):
            i+=2
            plt.subplot(3,4,i)
            self._plot_simple_w_colorbar(parameters, array)

        plt.subplot(3,4,10)
        self._plot_sigma(parameters)

        plt.subplot(3,4,11)
        self._plot_P(parameters)

        plt.subplot(3,4,12)
        self._plot_particals(parameters)


        plt.savefig('%s/%003d-%12.8f.png' % (self.figname,
                                             parameters['step'],
                                             parameters['T']))
        plt.close(fig)

    def _plot_particals(self, parameters, title = None):
        mxx, myy = parameters['mxx'], parameters['myy']
        m_cat = parameters['m_cat']

        if title:
            plt.title(title)
        plt.scatter(mxx,myy,c=m_cat,s=1,edgecolors='face',cmap='copper')
        plt.colorbar()
        plt.ylim([self.i_res-1,0])
        plt.xlim([0,self.j_res-1])


    def _plot_sigma(self, parameters, title = None):
        if not title:
            title = 'Sigma ii'
        plt.title(title)
        sii= parameters['sii']
        plt.imshow(sii[:-1,:-1],interpolation='none')
        plt.colorbar()

    def _plot_P(self, parameters):
        Vx, Vy = parameters['Vx'], parameters['Vy']
        P = parameters['P']

        Vy_average = 0.5*(Vx[1:-1,:-2]+Vx[:-2,:-2])
        Vx_average = 0.5*(Vy[ :-2,1:-1]+Vy[:-2,:-2])

        plt.title("P")
        plt.imshow(P[1:,1:],interpolation='none')
        plt.colorbar()
        plt.streamplot(self.jj[:-2,:-2]+.5,self.ii[:-2,:-2]+.5,Vy_average,Vx_average,color='white')
        plt.ylim([self.i_res-2,0])
        plt.xlim([0,self.j_res-2])


    def _plot_simple_w_colorbar(self, parameter, array, title=None):
        if not title:
            title=array
        plt.title(title)
        plt.imshow(parameter[array],
                   interpolation='none')
        plt.colorbar()

    def _plot_visocity(self, parameters ):
        plt.title("Viscosity")
        plt.imshow(parameters['eta_n'],interpolation='none',cmap='copper')
        plt.colorbar()

