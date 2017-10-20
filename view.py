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

        slice_n = lambda array: array[1:,1:]
        slice_v = lambda array: array[:-1,:-1]
        self.slices = { 'eta_n' :  slice_n,
                        'mu_n' : slice_n,
                        'Vx' : slice_v,
                        'Vy' : slice_v,
                        'e_xx' : slice_v,
                        'e_xy' : slice_v,
                        's_xx' : slice_v,
                        's_xy' : slice_v,
                        'default' : lambda array: array,
                        }

    def plot12(self, parameters):
        """ Make 12 plots on a list """
        plt.clf()
        fig = plt.figure(figsize = self.figsize)
        print(fig.get_size_inches())

        subtitle = f'Model size: {self.width/1000} km x {self.height/1000} ' +\
                   f'{self.j_res} x {self.i_res}, dx={self.dx}m, dy={self.dy}m.' +\
                   f'Current Time: {parameters["T"]}, Step: {parameters["step"]} ' +\
                   f'git: {self.git}'

        plt.suptitle(subtitle, fontsize=25)

        things_to_plot = ['eta_n', 'Vx', 'Vy', 'e_xx', 'e_xy', 's_xx', 's_xy', 'mu_n', 'w' ]
        titles =         [r'$\eta_n$', r'$V_x$', r'$V_y$', r'$e_{xx}$', r'$e_{xy}$', r'$\sigma_{xx}$', r'$\sigma_{xy}$', r'$\mu_{n}$', r'$\omega$' ]
        for i,(array, title) in enumerate(zip(things_to_plot, titles)):
            i+=1
            plt.subplot(3,4,i)
            self._plot_simple_w_colorbar(parameters, array, title=title)

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

        print(min(self.figsize))
        size = min(self.figsize)/m_cat.size*5000
        if title:
            plt.title(title, fontsize=15)
        plt.scatter(mxx,myy,c=m_cat,s=size,edgecolors='face',cmap='Blues')
        plt.colorbar()
        plt.ylim([self.i_res-1,0])
        plt.xlim([0,self.j_res-1])


    def _plot_sigma(self, parameters, title = r"$\sigma_{ii}$", fontsize=30):
        plt.title(title, fontsize=fontsize)
        sii= parameters['sii']
        plt.imshow(sii[:-1,:-1],interpolation='none',cmap='Reds')
        plt.colorbar()

    def _plot_P(self, parameters, fontsize=30):
        Vx, Vy = parameters['Vx'], parameters['Vy']
        P = parameters['P']

        Vy_average = 0.5*(Vx[1:-1,:-2]+Vx[:-2,:-2])
        Vx_average = 0.5*(Vy[ :-2,1:-1]+Vy[:-2,:-2])

        plt.title(r"$P$", fontsize=fontsize)
        plt.imshow(P[1:,1:],interpolation='none',cmap='seismic')
        plt.colorbar()
        plt.streamplot(self.jj[:-2,:-2]+.5,self.ii[:-2,:-2]+.5,Vy_average,Vx_average,color='black')
        plt.ylim([self.i_res-2,0])
        plt.xlim([0,self.j_res-2])


    def _plot_simple_w_colorbar(self, parameter, array, title=None, cmap='seismic', fontsize=30):
        if not title:
            title=array
        plt.title(title, fontsize=fontsize)
        A = parameter[array]
        try:
            slicer = self.slices[array]
        except KeyError:
            slicer = self.slices['default']
        plt.imshow(slicer(A),
                   interpolation='none',
                   cmap=cmap)
        plt.colorbar()
