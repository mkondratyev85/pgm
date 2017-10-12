import numpy as np
import matplotlib.pylab as plt

class Matplot(object):
    """ Draw set of grid as plots and write to file via Matplotlib """

    def __init__(self, **kwargs):
        defaults = {
                   'figsize' : (30,20),
                   'width' : None,
                   'height' : None,
                   'j_res' : 1,
                   'i_res' : 1,
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
        plt.savefig('%s/%003d-%12.8f.png' % (self.figname,
                                             parameters['step'],
                                             parameters['T']))
        plt.close(fig)


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
        plt.streamplot(self.jj[:-2,:-2]+0.5,self.ii[:-2,:-2]+0.5,Vx_average,Vy_average,color='white')
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

    def plot(self,T, Step, eta_n, mxx, myy, m_cat, sii, P, Vx, Vy, e_xx, e_xy, s_xx, s_xy, xelvis, mu_n, mu_s, w):
        Myr = lambda t: t/(365.25*24*3600*10**6) # Convert seconds to millions of year

        plt.clf()
        fig = plt.figure(figsize=(30,20))

        plt.suptitle("Model size: %s km x %s km (%s x %s cells, dx=%s km, dy=%s km). Current Time: %07.3f Myr. Step %s git verstion: %s" %
                                (self.width/1000, self.height/1000, self.j_res, self.i_res, self.dx/1000, self.dy/1000, Myr(T), Step, self.label))

        plt.subplot(3,4,1)
        plt.title("Viscosity")
        plt.imshow(eta_n,interpolation='none',cmap='copper')
        plt.colorbar()
        
        plt.subplot(3,4,2)
        plt.scatter(mxx,myy,c=m_cat,s=1,edgecolors='face',cmap='copper')
        plt.colorbar()
        plt.ylim([self.i_res-1,0])
        plt.xlim([0,self.j_res-1])

        plt.subplot(3,4,5)
        plt.title("Vx")
        plt.imshow(Vx,interpolation='none')
        plt.colorbar()

        plt.subplot(3,4,6)
        plt.title("Vy")
        plt.imshow(Vy,interpolation='none')
        plt.colorbar()

        plt.subplot(3,4,7)
        plt.title("e_xx")
        plt.imshow(e_xx,interpolation='none')
        plt.colorbar()

        plt.subplot(3,4,8)
        plt.title("e_xy")
        plt.imshow(e_xy,interpolation='none')
        plt.colorbar()

        plt.subplot(3,4,9)
        plt.title("s_xx")
        plt.imshow(s_xx,interpolation='none')
        plt.colorbar()

        plt.subplot(3,4,10)
        plt.title("s_xy")
        plt.imshow(s_xy,interpolation='none')
        plt.colorbar()

        plt.subplot(3,4,11)
        plt.title("mu_n")
        plt.imshow(mu_n,interpolation='none')
        plt.colorbar()

        plt.subplot(3,4,12)
        plt.title("w")
        plt.imshow(w,interpolation='none')
        plt.colorbar()

        
        plt.subplot(3,4,3)
        plt.title("Sigma II")
        plt.imshow(sii[:-1,:-1],interpolation='none')
        plt.colorbar()

        Vx_average = 0.5*(Vx[1:-1,:-2]+Vx[:-2,:-2])
        Vy_average = 0.5*(Vy[ :-2,1:-1]+Vy[:-2,:-2])
        plt.subplot(3,4,4)
        plt.title("P")
        plt.imshow(P[1:,1:],interpolation='none')
        plt.colorbar()
        plt.streamplot(self.jj[:-2,:-2],self.ii[:-2,:-2],Vx_average,Vy_average,color='white')
        plt.ylim([self.i_res-2,0])
        plt.xlim([0,self.j_res-2])

        plt.savefig('%s/%003d-%12.8f.png' % (self.figname, Step, Myr(T)))
        plt.close(fig)
