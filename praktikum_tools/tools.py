import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager

class ErrValue:
    __array_ufunc__ = None
    def __init__(self, value, error = 0):
        if not type(value) in (int, float, np.ndarray, np.float64, np.float32, np.int64, np.int32):
            raise TypeError(f'Value of type {type(value)} not supported!')
        if not type(error) in (int, float, np.ndarray, np.float64, np.float32, np.int64, np.int32):
            raise TypeError(f'Value of type {type(error)} not supported!')
        self.value = value
        self.error = error
    def __add__(self, other):
        if type(other) in (int, float, np.ndarray, np.float64, np.float32, np.int64, np.int32):
            return ErrValue(self.value + other, self.error)
        elif type(other) is ErrValue:
            return ErrValue(self.value + other.value, (self.error**2 + other.error**2)**.5)
        else: raise TypeError(f'Type {type(other)} is not supported!')
    def __radd__(self, other):
        return self + other
    def __sub__(self, other):
        if type(other) in (int, float, np.ndarray, np.float64, np.float32, np.int64, np.int32):
            return ErrValue(self.value - other, self.error)
        elif type(other) is ErrValue:
            return ErrValue(self.value - other.value, (self.error**2 + other.error**2)**.5)
        else: raise TypeError(f'Type {type(other)} is not supported!')
    def __rsub__(self, other):
        return self*(-1) + other
    def __mul__(self, other):
        if type(other) in (int, float, np.ndarray, np.float64, np.float32, np.int64, np.int32):
            return ErrValue(self.value*other, np.abs(self.error * other))
        elif type(other) is ErrValue:
            return ErrValue(self.value*other.value, ((self.error*other.value)**2 + (self.value*other.error)**2)**.5)
        else: raise TypeError(f'Type {type(other)} is not supported!')
    def __rmul__(self, other):
        return self*other
    def __truediv__(self, other):
        if type(other) in (int, float, np.ndarray, np.float64, np.float32, np.int64, np.int32):
            return ErrValue(self.value/other, self.error/other)
        elif type(other) is ErrValue:
            return ErrValue(self.value/other.value, ((self.error/other.value)**2 + (self.value*other.error/other.value**2)**2)**.5)
        else: raise TypeError(f'Type {type(other)} is not supported!')
    def __rtruediv__(self, other):
        if type(other) is ErrValue:
            return ErrValue(other.value/self.value, np.sqrt((other.value/self.value**2 * self.error)**2 + (other.error/self.value)**2))
        else:
            return ErrValue(other/self.value, np.abs(other/self.value**2 * self.error))
    def __pow__(self, other):
        if type(other) is ErrValue:
            return ErrValue(self.value**other.value, np.sqrt((other.value*self.error*self.value**(other.value-1))**2 + (self.value**other.value*np.log(self.value)*other.error)**2))
        else:
            return ErrValue(self.value**other, np.abs(self.error*other*self.value**(other-1)))
    def __rpow__(self, other):
        if type(other) is ErrValue:
            return ErrValue(other.value**self.value, np.sqrt((self.value*other.error*other.value**(self.value-1))**2 + (other.value**self.value*np.log(other.value)*self.error)**2))
        else:
            return ErrValue(other**self.value, np.abs(other**self.value*np.log(self.value)*self.error))
    def __neg__(self):
        return -1*self
    def sqrt(self):
        return self**.5
    def exp(self):
        return ErrValue(np.exp(self.value), np.exp(self.value)*self.error)
    def log(self):
        return ErrValue(np.log(self.value), self.error / np.abs(self.value))
    def __repr__(self):
        if type(self.value) in (int, float, np.float64, np.float32, np.int64, np.int32):
            if type(self.error) in (int, float, np.float64, np.float32, np.int64, np.int32): return f'{self.value} ± {self.error}'
            else:
                return '\n'.join([f'{self.value} ± {Error}' for Error in self.error])
        else:
            if type(self.error) in (int, float, np.float64, np.float32, np.int64, np.int32):
                return '\n'.join([f'{Value} ± {self.error}' for Value in self.value])
            else:
                return '\n'.join([f'{Value} ± {Error}' for Value, Error in zip(self.value, self.error)])
    def __getitem__(self, key):
        if type(self.value) is np.ndarray:
            if type(self.error) is np.ndarray:
                return ErrValue(self.value[key], self.error[key])
            else:
                return ErrValue(self.value[key], self.error)
        else:
            raise Exception('Single Error Value not subscriptable!')
    def __len__(self):
        if type(self.value) is np.ndarray: return self.value.size
        else: return 1
    def __round__(self, n = None):
        return ErrValue(round(self.value, n), round(self.error, n))
    def __gt__(self, other):
        if type(other) is ErrValue:
            return self.value > other.value
        else: return self.value > other
    def __lt__(self, other):
        if type(other) is ErrValue:
            return self.value < other.value
        else: return self.value < other
    def __ge__(self, other):
        if type(other) is ErrValue:
            return self.value >= other.value
        else: return self.value >= other
    def __le__(self, other):
        if type(other) is ErrValue:
            return self.value <= other.value
        else: return self.value <= other
    def __abs__(self):
        return ErrValue(np.abs(self.value), self.error)
    def mean(self):
        if type(self.value) in (int, float, np.float64, np.float32, np.int64, np.int32):
            return self
        elif type(self.value) is np.ndarray:
            if type(self.error) is np.ndarray:
                err = 1/np.sum(1/self.error)
                return ErrValue(np.sum(self.value/self.error) * err, err)
            else:
                return ErrValue(self.value.mean(), self.error/np.sqrt(self.value.size))
        else: raise Exception('Huups!!!')
    def sin(self):
        return ErrValue(np.sin(self.value), np.abs(np.cos(self.value)*self.error))
    def cos(self):
        return ErrValue(np.cos(self.value), np.abs(np.sin(self.value)*self.error))
    def tan(self):
        return ErrValue(np.tan(self.value), np.abs(self.error/np.cos(self.value)**2))
    def from_list(l, error = None):
        if error is None:
            if type(l) is ErrValue or type(l[0]) is ErrValue:
                return ErrValue(np.array([L.value for L in l]), np.array([L.error for L in l]))
            else: raise Exception("Need an error value!")
        else:
            if type(l) is ErrValue: raise Exception("Two error value supplied!")
            else: return ErrValue(l, error)
#    def __round__(self, ndigits = None):
#        if ndigits == None:
#            return self.roundsmart()
#        return ErrValue(round(self.value, ndigits), round(self.error, ndigits))
    def roundsmart(self, nsigdigits = 2):
        ndigits = (nsigdigits - np.ceil(np.log10(self.error))).astype(np.int64)
        return round(self, ndigits)

class Style:
    style_list = ['seaborn-v0_8', 'ggplot', 'default']
    for Style in style_list:
        if Style in plt.style.available:
            style = Style
            break
    else:
        raise Exception("None of the choosen styles are available")
    dpi = 300

plt.style.use(Style.style)

def plot_spectra(x, y, yerr = None, title = None, ylabel = None, xlabel = None, path = None, show = False, **kwargs) -> None:
    if type(y) is ErrValue:
        if yerr: yerr = y.error
        y = y.value
    if type(x) is ErrValue:
        x = x.value
    if yerr is None: plt.plot(x, y, **kwargs)
    else: plt.errorbar(x, y, yerr = yerr, **kwargs)
    if not title is None: plt.title(title)
    if not ylabel is None: plt.ylabel(ylabel)
    if not xlabel is None: plt.xlabel(xlabel)
    if not path is None: plt.savefig(path, dpi = Style.dpi)
    if show: plt.show()
    plt.close('all')

class Plot:
    def __init__(self, title = None, xlabel = None, ylabel = None, legend = False, log = None, scale_x = 1, scale_y = 1, legend_color = None):
        """
        Parameters
        ----------
        title : str, optional
            The title to be displayed at the top of the plot.
        xlabel, ylabel : str, optional
            A label for the corresponding axis.
        legend : bool, optional
            Whether or not a legend should be drawn.
        log : bool, str, optional
            Select whether the axis should be in logarithmic scale.
            If True, both axes are drawn in log.
            If string, the x axis is drawn in log if 'x' is in the string,
            same with 'y'.
        scale_x, scale_y : float, int, optional
            Factor to scale the corresponding axis.
            This is useful for converting with SI units. If the x unit is in e.g m
            but you want to plot in nm you can use scale_x = 1000.
        legend_color : str, list[str], optional
            Color to use for legend text.
        """
        self.legend = legend
        if not title is None: plt.title(title)
        if not xlabel is None: plt.xlabel(xlabel)
        if not ylabel is None: plt.ylabel(ylabel)
        if log == True:
            plt.xscale('log')
            plt.yscale('log')
        elif isinstance(log, str):
            if 'x' in log:
                plt.xscale('log')
            if 'y' in log:
                plt.yscale('log')
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.legend_color = legend_color
    def add(self, x, y = None, show_error = True, autoscale = True, **kwargs):
        if y is None:
            y, yerr = unpack_error(x)
            x, xerr = unpack_error(np.arange(len(y)))
        else:
            y, yerr = unpack_error(y)
            x, xerr = unpack_error(x)

        with self.autoscale(autoscale):
            if show_error:
               plt.errorbar(x*self.scale_x, y*self.scale_y, xerr=xerr*self.scale_x, yerr=yerr*self.scale_y, fmt='.', **kwargs)
            else:
                plt.plot(x*self.scale_x, y*self.scale_y, '.', linestyle = '', **kwargs)
        return self
    def line(self, x, y = None, label = None, show_error = True, autoscale = True, **kwargs):
        if y is None:
            y, yerr = unpack_error(x)
            x, _ = unpack_error(np.arange(len(y)))
        else:
            y, yerr = unpack_error(y)
            x, _ = unpack_error(x)

        with self.autoscale(autoscale):
            if label is None:
                plt.plot(x*self.scale_x, y*self.scale_y, **kwargs)
            else:
                plt.plot(x*self.scale_x, y*self.scale_y, label = label, **kwargs)
            if show_error:
                error_area = np.array([y - yerr, y + yerr])*self.scale_y
                plt.fill_between(x*self.scale_x, np.min(error_area, axis = 0), np.max(error_area, axis = 0), alpha = .15)
        return self
    def linear(self, m, n, x = None, label = None, show_error = True, autoscale = False, **kwargs):
        m, merr = unpack_error(m)
        n, nerr = unpack_error(n)
        if x is None:
            x = np.asarray(plt.gca().get_xlim())
        x, _ = unpack_error(x)
        padding = 0 if autoscale else (x.max() - x.min()) / 2
        rang = np.linspace(x.min() - padding, x.max() + padding, 1000)
        with self.autoscale(autoscale):
            if label is None:
                plt.plot(rang*self.scale_x, (m*rang + n)*self.scale_y, **kwargs)
            else:
                plt.plot(rang*self.scale_x, (m*rang + n)*self.scale_y, label = label, **kwargs)
            if show_error:
                plt.fill_between(rang*self.scale_x, ((m+merr)*rang + (n+nerr))*self.scale_y, ((m-merr)*rang + (n-nerr))*self.scale_y, alpha = 0.15)
        return self
    def func(self, f, param = (), x = None, overdraw = True, **kwargs):
        if x is None:
            x = plt.gca().get_xlim()
        x = smooth_range(x, overdraw = overdraw)
        self.line(x, f(x, *param), autoscale = not overdraw, **kwargs)
        return self
    def gauss(self, param, x = None, overdraw = True, draw_individual = False, **kwargs):
        if x is None:
            x = plt.gca().get_xlim()
        x = smooth_range(x, overdraw = overdraw)
        if draw_individual:
            for A, Sigma, Mu in zip(param['A'], param['sigma'], param['mu']):
                self.func(gauss, [{'A': [A], 'sigma': [Sigma], 'mu': [Mu], 'background': []}], x = x, overdraw = overdraw, **kwargs)
        else:
            self.func(gauss, [param], x = x, overdraw = overdraw, **kwargs)
    def area(self, value, stop = None, color = 'green', alpha = 0.5, autoscale = True, **kwargs):
        with self.autoscale(autoscale):
            if stop is None:
                plt.axvspan((value.value-value.error)*self.scale_x, (value.value+value.error)*self.scale_x, alpha = alpha, color = color, **kwargs)
            else:
                plt.axvspan(value.value*self.scale_x, stop.value*self.scale_x, alpha = alpha, color = color, **kwargs)
            return self
    def hist(self, x, **kwargs):
        x, _ = unpack_error(x)
        plt.hist(x, **kwargs)
    def heatmap(self, z, x = None, y = None, zlabel = None, contour = False, **kwargs):
        z, _ = unpack_error(z)
        x, _ = unpack_error(x)
        y, _ = unpack_error(y)

        if x is None or y is None:
            if type(z) is np.ndarray:
                x = np.linspace(0, 1, z.shape[1])
                y = np.linspace(0, 1, z.shape[0])
            else:
                raise NotImplemented()
        if type(x) is np.ndarray and type(y) is np.ndarray:
            if x.ndim == 1 and y.ndim == 1:
                x, y = np.meshgrid(x, y)
        plt.pcolormesh(x, y, z, cmap = 'viridis', **kwargs)
        if zlabel is None:
            plt.colorbar()
        else:
            plt.colorbar(label = zlabel)
        if contour: plt.contour(x, y, z)
        return self
    def hline(self, y, *args, **kwargs):
        plt.axhline(y*self.scale_y, *args, **kwargs)
        return self
    def vline(self, x, *args, **kwargs):
        plt.axvline(x*self.scale_x, *args, **kwargs)
        return self
    def make_legend(self):
        if self.legend:
            if self.legend_color:
                plt.legend(facecolor = 'k', labelcolor = self.legend_color)
            else:
                plt.legend()
    @contextmanager
    def autoscale(self, do_autoscale):
        if not do_autoscale:
            ax = plt.gca()
            lims = [ax.get_xlim(), ax.get_ylim()]
            yield
            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
        else:
            yield
    def show(self):
        self.make_legend()
        plt.show()
        return self
    def save(self, path, **kwargs):
        self.make_legend()
        plt.savefig(path, dpi = Style.dpi, **kwargs)
        return self
    def delete(self): plt.close('all')

def sqrt(x):
    """Return the square root of x."""
    if type(x) is ErrValue: return x.sqrt()
    else: return np.sqrt(x)

def exp(x):
    """Return the natural exponential of x."""
    if type(x) is ErrValue: return x.exp()
    else: return np.exp(x)

def log(x):
    """Return the natural log of x."""
    if type(x) is ErrValue: return x.log()
    else: return np.log(x)

def sin(x):
    """Return the sine of x."""
    if type(x) is ErrValue: return x.sin()
    else: return np.sin(x)

def cos(x):
    """Return the cosine of x."""
    if type(x) is ErrValue: return x.cos()
    else: return np.cos(x)

def tan(x):
    """Return the tangent of x."""
    if type(x) is ErrValue: return x.tan()
    else: return np.tan(x)

def load_from_csv(path: str, delimiter = ','):
    """Return values from a comma-seperated file as a list of numpy arrays."""
    with open(path, 'r') as f:
        values = [[]]
        for Line_num, Line in enumerate(f.readlines()):
            Line = Line.split('#')[0]
            if not Line: continue
            for i, Element in enumerate(Line.strip().split(delimiter)):
                if i >= len(values):
                    values.append([None for _ in range(len(values[0])-1)])
                try:
                    values[i].append(float(Element))
                except ValueError:
                    raise ValueError(f"Couldn't interpret value '{Element}' in line {Line_num} as float!")
    if len(values) == 1:
        return np.array(values[0])
    else:
        values = [np.array(Element) for Element in values]
        return values

def save_as_csv(path: str, *data, labels = None, round_to = None):
    with open(path, "w") as f:
        if labels:
            labels_to_write = []
            for Data, Label in zip(data, labels):
                labels_to_write.append(Label)
                if type(Data) is ErrValue:
                    labels_to_write.append(Label + "-Error")
            f.write("# " + ", ".join(labels_to_write) + "\n")
        for i in range(len(data[0])):
            Line = []
            if round_to:
                for Data in data:
                    if type(Data) is ErrValue:
                        Line.append(str(round(Data[i].value, round_to)))
                        Line.append(str(round(Data[i].error, round_to)))
                    else:
                        Line.append(str(round(Data[i], round_to)))
            else:
                for Data in data:
                    if type(Data) is ErrValue:
                        Line.append(str(Data[i].value))
                        Line.append(str(Data[i].error))
                    else:
                        Line.append(str(Data[i]))
            f.write(", ".join(Line) + "\n")

def linear_fit(x, y):
    """Perform a linear fit on x and y and return the slope, y-intercept and chi-squared."""
    if isinstance(x, ErrValue):
        return linear_fit_both_error(x, y)
    x, _ = unpack_error(x)
    y, yerr = unpack_error(y)


    xy = x*y
    x2 = x*x


    if hasattr(yerr, 'size'):
        mitxy = 0
        mitx = 0
        mity = 0
        mitx2 = 0
        var = 0
        for X,Y,XY,X2,YERR in zip(x,y,xy,x2,yerr):
            var +=  1/(YERR*YERR)
            mitxy += XY/(YERR*YERR)
            mitx += X/(YERR*YERR)
            mity += Y/(YERR*YERR)
            mitx2 += X2/(YERR*YERR)
        mitxy = mitxy/var
        mitx = mitx/var
        mity = mity/var
        mitx2 = mitx2/var
        mitvar = x.size/var

    else:
        mitxy = np.mean(xy)
        mitx = np.mean(x)
        mity = np.mean(y)
        mitx2 = np.mean(x2)
        mitvar = yerr*yerr


    m = (mitxy-mitx*mity)/(mitx2-mitx*mitx)
    n = (mitx2*mity-mitx*mitxy)/(mitx2-mitx*mitx)
    m_err = np.sqrt(mitvar/(x.size*(mitx2-mitx*mitx)))
    n_err = np.sqrt(mitvar*mitx2/(x.size*(mitx2-mitx*mitx)))

    chi_sq = np.sum(((y - (m*x + n)) / yerr)**2)


    return ErrValue(m, m_err), ErrValue(n, n_err), chi_sq

def linear_fit_both_error(x, y):
    x, xerr = unpack_error(x)
    m, n, _ = linear_fit(x, y)
    y, yerr = unpack_error(y)
    threshold = 1e-10

    m, _ = unpack_error(m)
    n, nerr = unpack_error(n)

    prev_m = m

    for _ in range(1000):
        tot_err_sq = (m*xerr)**2 + yerr**2
        D = np.sum(1/tot_err_sq) * np.sum(x**2 / tot_err_sq) - np.sum(x / tot_err_sq)**2
        m = 1/D * (np.sum(1/tot_err_sq) * np.sum(x*y / tot_err_sq) - np.sum(x / tot_err_sq) * np.sum(y / tot_err_sq))

        if abs((prev_m - m) / prev_m) < threshold: break
        prev_m = m

    merr = (1/D * np.sum(1/tot_err_sq))**.5

    chi_sq = np.sum(((y - (m*x + n)) / yerr)**2)


    return ErrValue(m, merr), ErrValue(n, nerr), chi_sq

def fit_func(f, x, y, *args, **kwargs):
    """Fit an arbitrary function and return the fitted function parameters."""
    from scipy.optimize import curve_fit

    x, _ = unpack_error(x)
    y, yerr = unpack_error(y)

    if np.any(yerr > 0):
        popt, pcov = curve_fit(f, x, y, *args, sigma = yerr, absolute_sigma = True, **kwargs)
    else:
        popt, pcov = curve_fit(f, x, y, *args, **kwargs)
    values = ErrValue(np.asarray(popt), np.sqrt(np.diag(pcov)))

    return values

def unpack_error(x):
    if isinstance(x, ErrValue):
        return x.value, x.error
    elif isinstance(x, np.ndarray):
        return x, np.zeros(x.shape)
    elif isinstance(x, list):
        return x, [0 for _ in x]
    else: return x, 0


def gauss(x, paras: dict):
    from praktikum_tools.functions.gauss import getter
    background = [None, 'const', 'linear', 'quadratic'][len(paras['background']) if hasattr(paras['background'], '__len__') else 0]
    f = getter(len(paras['A']), background = background)
    return f(x, *paras['background'], *paras['A'], *paras['sigma'], *paras['mu'])



def smooth_range(x, n = 1000, overdraw = False):
    if type(x) is tuple:
        xmin, _ = unpack_error(x[0])
        xmax, _ = unpack_error(x[1])
    else:
        x, _ = unpack_error(x)
        xmin = x.min()
        xmax = x.max()

    if overdraw:
        padding = (xmax - xmin) / 4
        xmin -= padding
        xmax += padding
    return np.linspace(xmin, xmax, n)


def fit_gauss(x, y, n, background = None, A0 = None, sigma0 = None, mu0 = None, background0 = None, A_range = None, sigma_range = None, mu_range = None, background_range = None, *args, **kwargs):
    """Fit a gauss curve with optional background and return the fitting parameters."""
    from praktikum_tools.functions.gauss import getter
    f = getter(n, background)

    if background0: background_size = len(background0)
    else: background_size = (background == 'const') + 2 * (background == 'linear') + 3 * (background == 'quadratic')

    p0 = [1 for _ in range(3*n + background_size)]

    if background0:
        p0[:background_size] = background0
    if A0:
        p0[background_size:n+background_size] = A0
    if sigma0:
        p0[background_size+n:background_size+2*n] = sigma0
    if mu0:
        p0[background_size+2*n:] = mu0


    bounds_min = [-np.inf]*background_size + [0]*2*n + [-np.inf]*n
    bounds_max = [np.inf for _ in range(3*n + background_size)]

    if background_range:
        if type(background_range) is list:
            bounds_min[:background_size] = [B0 - B_range for B0, B_range in zip(background0, background_range)]
            bounds_max[:background_size] = [B0 + B_range for B0, B_range in zip(background0, background_range)]
        else:
            bounds_min[:background_size] = [B0 - background_range for B0 in background0]
            bounds_max[:background_size] = [B0 + background_range for B0 in background0]
    if A_range:
        if type(background_range) is list:
            bounds_min[background_size:background_size + n] = [Value - Range for Value, Range in zip(A0, A_range)]
            bounds_max[background_size:background_size + n] = [Value + Range for Value, Range in zip(A0, A_range)]
        else:
            bounds_min[background_size:background_size + n] = [Value - A_range for Value in A0]
            bounds_max[background_size:background_size + n] = [Value + A_range for Value in A0]
    if sigma_range:
        if type(background_range) is list:
            bounds_min[background_size + n:background_size + 2*n] = [max(0, Value - Range) for Value, Range in zip(sigma0, sigma_range)]
            bounds_max[background_size + n:background_size + 2*n] = [Value + Range for Value, Range in zip(sigma0, sigma_range)]
        else:
            bounds_min[background_size + n:background_size + 2*n] = [max(0, Value - sigma_range) for Value in sigma0]
            bounds_max[background_size + n:background_size + 2*n] = [Value + sigma_range for Value in sigma0]
    if mu_range:
        if type(background_range) is list:
            bounds_min[background_size + 2*n:] = [Value - Range for Value, Range in zip(mu0, mu_range)]
            bounds_max[background_size + 2*n:] = [Value + Range for Value, Range in zip(mu0, mu_range)]
        else:
            bounds_min[background_size + 2*n:] = [Value - mu_range for Value in mu0]
            bounds_max[background_size + 2*n:] = [Value + mu_range for Value in mu0]


    paras = fit_func(f, x, y, *args, p0 = p0, bounds = (bounds_min, bounds_max), **kwargs)

    return {'background': paras[:background_size] if background_size else [], 'A': paras[background_size:background_size + n], 'sigma': paras[background_size + n:background_size + 2*n], 'mu': paras[background_size + 2*n:]}


def round_pretty(x, n):
    return f"%.{n}f" % round(x, n)

def create_latex_table(labels, *values, rounding_factor = None, caption = '', label = ''):
    res = r"\begin{table}[!h]" + "\n\t" + r"\centering" + "\n\t" + r"\footnotesize" + "\n\t" + r"\begin{tabular}{" + '|'.join(['c' for _ in labels]) + r"}" + "\n\t\t" + r"\toprule"

    res += '\n\t\t' + ' & '.join(labels) + r' \\ \midrule'

    for i in range(len(values[0])):
        res += '\n\t\t'# + ' & '.join([Value[i] for Value in values]) + r' \\'
        for j, Value in enumerate(values):
            x = Value[i]
            if type(x) is ErrValue:
                if rounding_factor:
                    x = round_pretty(x.value, rounding_factor[j]) + r' $\pm$ ' + round_pretty(x.error, rounding_factor[j])
                else:
                    x = str(x.value) + r' $\pm$ ' + str(x.error)
            elif type(x) is not str:
                if rounding_factor:
                    x = round_pretty(x, rounding_factor[j])
                else:
                    x = str(x)
            res += x + ' & '
        res = res[:-3] + r' \\'

    res += '\n\t' + r'\end{tabular}' + '\n\t' + r'\caption{' + caption + r'}' + '\n\t' + r'\label{' + label + r'}' + '\n' + r'\end{table}'
    return res
