""" environment

The `Environment` class contains all the necessary parameters to initialize a
network for a simulation.

"""

#############
## Imports ##
#############

# Standard Library
import re
import sys
import inspect
from collections import deque

# Torch
import torch

# Other
import numpy as np

py2 = sys.version_info.major == 2


#############
## Globals ##
#############

_current_environments = deque()


#############
## Helpers ##
#############


class _DefaultArgument:
    """ wrap default function arguments in this class to figure out if argument
    was supplied manually or due to being there by defualt."""

    pass


class _float(float, _DefaultArgument):
    """ wrap default float function arguments in this class to figure out if
    argument was supplied manually or due to being there by defualt."""

    def __repr__(self):
        return "%.3e" % self


class _int(int, _DefaultArgument):
    """ wrap default int function arguments in this class to figure out if
    argument was supplied manually or due to being there by defualt."""

    pass


class _bool(int, _DefaultArgument):
    """ wrap default bool function arguments in this class to figure out if
    argument was supplied manually or due to being there by defualt."""

    pass


class _str(str, _DefaultArgument):
    """ wrap default string function arguments in this class to figure out if
    argument was supplied manually or due to being there by defualt."""

    pass


class _Array(np.ndarray):
    """ numpy array with more concise repr """

    def __repr__(self):
        if self.ndim != 1:
            return (
                super(_Array, self).__repr__().replace(self.__class__.__name__, "array")
            )
        if self.shape[0] > 4:
            return "array([%.3e, %.3e, ..., %.3e])" % (self[0], self[1], self[-1])
        else:
            return "array(%s)" % (str(["%.3e" % v for v in self]).replace("'", ""))

    def __str__(self):
        if self.ndim != 1:
            return super(_Array, self).__str__()
        if self.ndim == 1 and self.shape[0] > 4:
            return "[%.3e, %.3e, ..., %.3e]" % (self[0], self[1], self[-1])
        else:
            s = "[" + "%.3e " * self.shape[0] + "]"
            return s % self


class _DefaultArray(_Array, _DefaultArgument):
    """ wrap default numpy array function arguments in this class to figure out
    if argument was supplied manually or due to being there by defualt."""

    pass


def _array(arr, default_array=False):
    """ create either an _Array or a _DefaultArray. """
    arr = arr.view(_DefaultArray) if default_array else arr.view(_Array)
    return arr


def _is_default(arg):
    """ check if a certain value is a default argument """
    return isinstance(arg, _DefaultArgument) or arg is None


def _convert_default(value):
    """ convert any value to a default argument """
    if isinstance(value, bool):
        return _bool(value)
    elif isinstance(value, int):
        return _int(value)
    elif isinstance(value, float):
        return _float(value)
    elif isinstance(value, str):
        return _str(value)
    elif isinstance(value, np.ndarray):
        return _array(value, default_array=True)
    else:
        return value


#################
## Environment ##
#################


class Environment(object):
    """ Simulation Environment

    The simulation environment is a smart data class that contains all
    the necessary parameters to initialize a network for a simulation.

    It is able to initialize parameters that depend on each other correctly.

    Note:
        After creating an Environment, the parameters of the environment
        get frozen and cannot be changed. However, a new environment can be
        created from the old one with the ``.copy`` method, which accepts the
        same arguments as Environment, but uses the current values as defaults.

    """

    def __init__(
        self,
        dt=_float(1e-14),
        samplerate=_float(1e14),
        num_t=_int(100),
        t0=_float(0),
        t1=_float(1e-12),
        t=_array(np.arange(0, 1e-12, 1e-14), default_array=True),
        bitrate=None,
        bitlength=None,
        wl=_float(1.55e-6),
        f=_float(299792458.0 / 1.55e-6),
        dwl=_float(1e-9),
        df=_float(299792458.0 / 1.550e-6 - 99792458.0 / 1.551e-6),
        num_wl=_int(1),
        num_f=_int(1),
        wl0=_float(1.5e-6),
        f0=_float(299792458.0 / 1.6e-6),
        wl1=_float(1.6e-6),
        f1=_float(299792458.0 / 1.5e-6),
        c=_float(299792458.0),
        freqdomain=_bool(False),
        grad=_bool(False),
        name=_str("env"),
        **kwargs
    ):
        """
        Args:
            t (np.ndarray): [s] full 1D time array (mutually exclusive with dt, t0, t1, num_t and samplerate).
            t0 (float): [s] starting time of the simulation (mutually exclusive with t).
            t1 (float): [s] ending time of the simulation (mutually exclusive with t and num_t).
            num_t (int): number of timesteps in the simulation (mutually exclusive with t, dt and samplerate).
            dt (float): [s] timestep of the simulation (mutually exclusive with t and samplerate)
            samplerate (float): [1/s] samplerate of the simulation (mutually exclusive with t and t1).
            bitrate (optional, float): [1/s] bitrate of the signal (mutually exclusive with bitlength).
            bitlength (optional, float): [s] bitlength of the signal (mutually exclusive with bitrate).
            wl (float): [m] full 1D wavelength array (mutually exclusive with f, dwl, df, num_wl, num_f, wl0, f0, wl1 and f1).
            wl0 (float): [m] start of wavelength range (mutually exclusive with wl, f and f0).
            wl1 (float): [m] end of wavelength range (mutually exclusive with wl, f and f1).
            num_wl (int): number of independent wavelengths in the simulation (mutually exclusive with wl, f, num_f, dwl and df)
            dwl (float): [m] wavelength step sizebetween wl0 and wl1 (mutually exclusive with wl, f, df, num_wl and num_f).
            f (float): [1/s] full 1D frequency array (mutually exclusive with wl, dwl, df, num_wl, num_f, wl0, f0, wl1 and f1).
            f0 (float): [1/s] start of frequency range (mutually exclusive with wl, f and wl0).
            f1 (float): [1/s] end of frequency range (mutually exclusive with wl, f and wl1).
            num_f (int): number of independent frequencies in the simulation (mutually exclusive with wl, f, num_wl, dwl and df)
            df (float): [1/s] frequency step between f0 and f1 (mutually exclusive with wl, f, dwl, num_wl and num_f).
            c (float): [m/s] speed of light used during simulations.
            freqdomain (bool): only do frequency domain calculations.
            grad (bool): track gradients during the simulation (set this to True during training.)
            name (str): name of the environment
            **kwargs (optional): any number of extra keyword arguments will be stored as attributes to the environment.
        """
        c = float(c)

        # parse kwargs for backward compatibility:
        wl = kwargs.pop("wavelength", wl)
        t0 = kwargs.pop("t_start", t0)
        t1 = kwargs.pop("t_end", t1)
        num_t = kwargs.pop("num_timesteps", num_t)
        t = kwargs.pop("time", t)
        dt = kwargs.pop("timestep", dt)
        wavelength = kwargs.pop("wavelength", wl)
        num_wl = kwargs.pop("num_wavelengths", num_wl)
        dwl = kwargs.pop("wavelength_step", dwl)
        wl0 = kwargs.pop("wavelength_start", wl0)
        wl1 = kwargs.pop("wavelength_end", wl1)
        freqdomain = kwargs.pop("frequency_domain", freqdomain)
        grad = kwargs.pop("enable_grad", grad)

        if sum([not _is_default(v) for v in (dt, samplerate, t)]) > 1:
            raise ValueError(
                "Environment: too many arguments given to determine t array: arguments 'dt', 'samplerate' and 't' are mutually exclusive"
            )
        if not _is_default(t0) and not _is_default(t):
            raise ValueError(
                "Environment: too many arguments given to determine t array: arguments 't0' and 't' are mutually exclusive"
            )
        if sum([not _is_default(v) for v in (t1, num_t, t)]) > 1:
            raise ValueError(
                "Environment: too many arguments given to determine t array: arguments 't1', 'num_t' and 't' are mutually exclusive"
            )
        if not _is_default(bitrate) and not _is_default(bitlength):
            raise ValueError(
                "Environment: too many arguments given to determine bitrate: arguments 'bitrate' and 'bitlength' are mutually exclusive"
            )
        if not _is_default(wl) and not _is_default(f):
            raise ValueError(
                "Environment: too many arguments given to determine wavelength array: arguments 'wl' and 'f' are mutually exclusive"
            )
        if sum([not _is_default(v) for v in (dwl, df, num_wl, num_f, wl, f)]) > 1:
            raise ValueError(
                "Environment: too many arguments given to determine wavelength array: arguments 'dwl', 'df', 'num_wl', 'num_f', 'wl' and 'f' are mutually exclusive."
            )
        if sum([not _is_default(v) for v in (wl0, f0, wl, f)]) > 1:
            raise ValueError(
                "Environment: too many arguments given to determine wavelength array: arguments 'wl0', 'f0', 'wl' and 'f' are mutually exclusive."
            )
        if sum([not _is_default(v) for v in (wl1, f1, wl, f)]) > 1:
            raise ValueError(
                "Environment: too many arguments given to determine wavelength array: arguments 'wl1', 'f1', 'wl' and 'f' are mutually exclusive."
            )
        if dt < 0:
            raise ValueError("'dt >= 0' required.")
        if t0 > t1:
            raise ValueError("'t0 < t1' required.")

        if not _is_default(bitlength):
            bitrate = float(1 / bitlength)

        if (
            not _is_default(t)
            or sum([not _is_default(v) for v in (dt, samplerate, num_t, t0, t1)]) == 0
        ):
            if not isinstance(t, np.ndarray):
                t = np.array(t)
            if t.ndim == 0:
                t = t[None]
            elif t.ndim > 1:
                raise ValueError(
                    "Dimensionality of 't' array too high: expected 1D array, got %iD array."
                    % t.ndim
                )
            t = _array(t)
        else:
            if not _is_default(samplerate):
                dt = float(1 / samplerate)
            if not _is_default(num_t):
                t = _array(np.linspace(t0, t0 + num_t * dt, num_t, endpoint=False))
            else:
                try:
                    t = _array(np.arange(t0, t1, dt))
                except ValueError:
                    raise ValueError(
                        "Cannot create time range. Are dt or num_t, t0 and t1 all specified?"
                    )
        if freqdomain:
            t = t[:1]
            bitrate = bitlength = None

        if (not _is_default(wl) or not _is_default(f)) or sum(
            [not _is_default(v) for v in (num_wl, num_f, dwl, df, wl0, f0, wl1, f1)]
        ) == 0:
            if not _is_default(f):
                wl = c / f
            if torch.is_tensor(wl):
                wl = wl.to(torch.float64).detach().cpu().numpy()
            if not isinstance(wl, np.ndarray):
                wl = np.array(wl)
            if wl.ndim == 0:
                wl = wl[None]
            elif wl.ndim > 1:
                raise ValueError(
                    "Dimensionality of 'wl' or 'f' array too high: expected 1D array, got %iD array."
                    % wl.ndim
                )
        else:
            if not _is_default(num_f):
                if not _is_default(wl0):
                    f1 = c / wl0
                f1 = float(f1)
                if not _is_default(wl1):
                    f0 = c / wl0
                f0 = float(f0)
                try:
                    df = float((f1 - f0) / num_f)
                    wl = c / np.arange(f0, f1, df)
                except ValueError:
                    raise ValueError(
                        "Cannot create frequency range. Are df or num_f, f0 and f1 all specified?"
                    )
            elif not _is_default(df):
                if not _is_default(wl0):
                    f1 = c / wl0
                f1 = float(f1)
                if not _is_default(wl1):
                    f0 = c / wl0
                f0 = float(f0)
                try:
                    df = float(abs(df) if f1 >= f0 else -abs(df))
                    wl = c / np.arange(f0, f1, df)
                except ValueError:
                    raise ValueError(
                        "Cannot create frequency range. Are df or num_f, f0 and f1 all specified?"
                    )
            else:
                if not _is_default(f0):
                    wl0 = c / f0
                wl0 = float(wl0)
                if not _is_default(f1):
                    wl1 = c / f1
                wl1 = float(wl1)
                if not _is_default(num_wl):
                    dwl = (wl1 - wl0) / num_wl
                dwl = float(dwl)
                try:
                    wl = np.arange(wl0, wl1, dwl)
                except ValueError:
                    raise ValueError(
                        "Cannot create frequency range. Are dwl or num_wl, wl0 and wl1 all specified?"
                    )

        self.name = str(name)
        self.t = self.time = _array(t)
        self.t0 = self.t_start = float(t[0])
        self.t1 = self.t_end = (
            None if t.shape[0] < 2 else float(t[-1]) + float(t[1] - t[0])
        )
        self.num_t = self.num_timesteps = int(t.shape[0])
        self.dt = self.timestep = None if t.shape[0] < 2 else float(t[1] - t[0])
        self.samplerate = None if t.shape[0] < 2 else float(1 / self.dt)
        self.bitrate = None if bitrate is None else float(bitrate)
        self.bitlength = None if bitrate is None else float(1 / bitrate)
        self.wl = self.wavelength = _array(wl)
        self.wl0 = self.wavelength_start = float(self.wl[0])
        self.wl1 = self.wavelength_end = (
            None
            if self.wl.shape[0] < 2
            else float(self.wl[-1] + (self.wl[-1] - self.wl[-2]))
        )
        self.num_wl = self.num_wavelengths = int(self.wl.shape[0])
        self.dwl = self.wavelength_step = (
            None if wl.shape[0] < 2 else float(self.wl[1] - self.wl[0])
        )
        self.f = _array(c / self.wl)
        self.f0 = float(self.f[0])
        self.f1 = (
            None
            if self.f.shape[0] < 2
            else float(self.f[-1] + (self.f[-1] - self.f[-2]))
        )
        self.num_f = int(self.f.shape[0])
        self.df = None if self.f.shape[0] < 2 else float(self.f[1] - self.f[0])
        self.c = float(c)
        self.freqdomain = self.frequency_domain = bool(freqdomain)
        self.grad = self.enable_grad = bool(grad)
        self.__dict__.update(kwargs)
        self._grad_manager = torch.enable_grad() if self.grad else torch.no_grad()
        # synonyms for backward compatibility:
        self._synonyms = (
            "wavelength",
            "t_start",
            "t_end",
            "num_timesteps",
            "time",
            "timestep",
            "wavelength",
            "num_wavelengths",
            "wavelength_step",
            "wavelength_start",
            "wavelength_end",
            "frequency_domain",
            "enable_grad",
        )
        self._initialized = True

    def copy(self, **kwargs):
        """ Create a copy of the environment

        Note:
            This copy method accepts the same keyword arguments as the
            Environment initialization.  Supplying arguments overrides the
            values in the copied environment. The original environment remains
            unchanged.

        Args:
            t (np.ndarray): [s] full 1D time array (mutually exclusive with dt, t0, t1, num_t and samplerate).
            t0 (float): [s] starting time of the simulation (mutually exclusive with t).
            t1 (float): [s] ending time of the simulation (mutually exclusive with t and num_t).
            num_t (int): number of timesteps in the simulation (mutually exclusive with t, dt and samplerate).
            dt (float): [s] timestep of the simulation (mutually exclusive with t and samplerate)
            samplerate (float): [1/s] samplerate of the simulation (mutually exclusive with t and t1).
            bitrate (optional, float): [1/s] bitrate of the signal (mutually exclusive with bitlength).
            bitlength (optional, float): [s] bitlength of the signal (mutually exclusive with bitrate).
            wl (float): [m] full 1D wavelength array (mutually exclusive with f, dwl, df, num_wl, num_f, wl0, f0, wl1 and f1).
            wl0 (float): [m] start of wavelength range (mutually exclusive with wl, f and f0).
            wl1 (float): [m] end of wavelength range (mutually exclusive with wl, f and f1).
            num_wl (int): number of independent wavelengths in the simulation (mutually exclusive with wl, f, num_f, dwl and df)
            dwl (float): [m] wavelength step sizebetween wl0 and wl1 (mutually exclusive with wl, f, df, num_wl and num_f).
            f (float): [1/s] full 1D frequency array (mutually exclusive with wl, dwl, df, num_wl, num_f, wl0, f0, wl1 and f1).
            f0 (float): [1/s] start of frequency range (mutually exclusive with wl, f and wl0).
            f1 (float): [1/s] end of frequency range (mutually exclusive with wl, f and wl1).
            num_f (int): number of independent frequencies in the simulation (mutually exclusive with wl, f, num_wl, dwl and df)
            df (float): [1/s] frequency step between f0 and f1 (mutually exclusive with wl, f, dwl, num_wl and num_f).
            c (float): [m/s] speed of light used during simulations.
            freqdomain (bool): only do frequency domain calculations.
            grad (bool): track gradients during the simulation (set this to True during training.)
            name (str): name of the environment
            **kwargs (optional): any number of extra keyword arguments will be stored as attributes to the environment.
        """
        if py2:
            init_argspec = inspect.getargspec(self.__init__)
        else:
            init_argspec = inspect.getfullargspec(self.__class__)
        default_kwargs = dict(zip(init_argspec.args[1:], init_argspec.defaults))
        new = {
            k: (_convert_default(v) if v is not None else default_kwargs.get(k))
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not k in self._synonyms
        }
        new.update(kwargs)
        return self.__class__(**new)

    def __enter__(self):
        _current_environments.appendleft(self)
        self._grad_manager.__enter__()
        return self

    def __exit__(self, error, value, traceback):
        """ exit the with block (close the current environment) """
        if _current_environments[0] is self:
            del _current_environments[0]
        self._grad_manager.__exit__()
        if error is not None:
            raise  # raise the last error thrown
        return self

    def __eq__(self, other):
        if not isinstance(other, Environment):
            return False
        self = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and k not in ("f", "t", "wl") + self._synonyms
        }
        other = {
            k: v
            for k, v in other.__dict__.items()
            if not k.startswith("_") and k not in ("f", "t", "wl") + other._synonyms
        }
        return self == other

    def __repr__(self):
        s = "Environment("
        for k, v in self.__dict__.items():
            if k.startswith("_") or k in self._synonyms:
                continue
            if isinstance(v, float):
                s = s + "%s=%s, " % (str(k), "%.3e" % v)
                continue
            s = s + "%s=%s, " % (str(k), repr(v))
        s = s[:-2] + ")"
        return s

    def __str__(self):
        s = "Simulation Environment:\n" "-----------------------\n"
        colwidth = (
            max(
                len(k)
                for k in self.__dict__.keys()
                if not k.startswith("_") and not k in self._synonyms
            )
            + 1
        )
        for k, v in self.__dict__.items():
            if k.startswith("_") or k in self._synonyms:
                continue
            if isinstance(v, float):
                s = s + "%s%s: %s\n" % (str(k), " " * (colwidth - len(k)), "%.3e" % v)
                continue
            s = s + "%s%s: %s\n" % (str(k), " " * (colwidth - len(k)), repr(v))
        return s

    def _repr_html_(self):
        descriptions = [
            row.strip().split(":")
            for row in self.__init__.__doc__.split("Args:\n")[1]
            .split("\n\n")[0]
            .split("\n")
            if (row.strip() and ":" in row)
        ]
        descriptions = {
            re.sub(r" \(.*\)", "", k).strip(): re.sub(r" \(.*\)", "", v).strip()
            for k, v in descriptions
        }
        row = "<tr><th>%s</th><td>%s</td><td>%s</td></tr>\n"
        html = "<div>\n<table>\n<thead>\n<tr>\n<th>key</th>\n<th>value</th>\n<th>description</th>\n</tr>\n</thead>\n<tbody>\n"
        for k, v in self.__dict__.items():
            if k.startswith("_") or k in self._synonyms:
                continue
            if isinstance(v, float):
                html = html + row % (k, "%.3e" % v, descriptions.get(k, ""))
                continue
            html = html + row % (k, str(v), descriptions.get(k, ""))
        html = html + "</tbody>\n</table>\n</div>"
        return html

    def __setattr__(self, name, value):
        """ this locks the attributes """
        if hasattr(self, "_initialized") and self._initialized:
            raise AttributeError(
                "Changing the attributes of an environment is not allowed. "
                "Consider creating a new environment or use the copy method."
            )
        else:
            super(Environment, self).__setattr__(name, value)

    def __setitem__(self, name, value):
        """ this locks the attributes """
        if hasattr(self, "_initialized") and self._initialized:
            raise AttributeError(
                "Changing the attributes of an environment is not allowed. "
                "Consider creating a new environment or use the copy method."
            )
        else:
            super(Environment, self).__setitem__(name, value)


#########################
## Current Environment ##
#########################


def current_environment():
    """ get the current environment """
    if _current_environments:
        return _current_environments[0]
    else:
        raise RuntimeError(
            "Environment not found. Execute your code inside an "
            "environment-defining with-block or use the 'set_environment()' "
            "function to set the environment globally."
        )


def set_environment(*args, **kwargs):
    """ set the environment globally

    Args:
        env (Environment): The environment to set globally.
        **kwargs: keyword arguments to define a new environment.

    Note:
        It is recommended to set the Environment using a with-block. However,
        if you would like to set the environment globally, this can be done
        with this function.

    """
    if len(args) > 2:
        raise ValueError("Only one positional argument allowed")
    elif len(args) == 1:
        env = args[0].copy(**kwargs)
    elif "env" in kwargs:
        env = kwargs.pop("env")
        env = env.copy(**kwargs)
    else:
        env = Environment(**kwargs)

    env.__enter__()
