""" Photontorch Simulation Environment

The simulation environment module contains a single class: ``Environment``

This class contains all the necessary parameters to initialize a network for a
simulation.

"""

#############
## Imports ##
#############

# Standard Library
import re
from copy import deepcopy
from collections import OrderedDict
from collections import deque

# Torch
import torch

# Other
import numpy as np


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
    arr.setflags(write=False)
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
            dt (float): [s] timestep of the simulation (mutually exclusive with t, samplerate and num_t)
            samplerate (float): [1/s] samplerate of the simulation (mutually exclusive with t, dt and num_t).
            num_t (int): number of timesteps in the simulation (mutually exclusive with t, dt and samplerate).
            t0 (float): [s] starting time of the simulation (mutually exclusive with t).
            t1 (float): [s] ending time of the simulation (mutually exclusive with t).
            t (np.ndarray): [s] full 1D time array (mutually exclusive with dt, t0, t1, num_t and samplerate).
            bitrate (optional, float): [1/s] bitrate of the signal (mutually exclusive with bitlength).
            bitlength (optional, float): [s] bitlength of the signal (mutually exclusive with bitrate).
            wl (float): [m] full 1D wavelength array (mutually exclusive with f, dwl, df, num_wl, num_f, wl0, f0, wl1 and f1).
            f (float): [1/s] full 1D frequency array (mutually exclusive with wl, dwl, df, num_wl, num_f, wl0, f0, wl1 and f1).
            dwl (float): [m] wavelength step sizebetween wl0 and wl1 (mutually exclusive with wl, f, df, num_wl and num_f).
            df (float): [1/s] frequency step between f0 and f1 (mutually exclusive with wl, f, dwl, num_wl and num_f).
            num_wl (int): number of independent wavelengths in the simulation (mutually exclusive with wl, f, num_f, dwl and df)
            num_f (int): number of independent frequencies in the simulation (mutually exclusive with wl, f, num_wl, dwl and df)
            wl0 (float): [m] start of wavelength range (mutually exclusive with wl, f and f0).
            f0 (float): [1/s] start of frequency range (mutually exclusive with wl, f and wl0).
            wl1 (float): [m] end of wavelength range (mutually exclusive with wl, f and f1).
            f1 (float): [1/s] end of frequency range (mutually exclusive with wl, f and wl1).
            c (float): [m/s] speed of light used during simulations.
            freqdomain (bool): only do frequency domain calculations.
            grad (bool): track gradients during the simulation (set this to True during training.)
            name (optional, str): name of the environment
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

        if sum([not _is_default(v) for v in (dt, num_t, samplerate, t)]) > 1:
            raise ValueError(
                "Environment: too many arguments given to determine t array: arguments 'dt', 'num_t', 'samplerate' and 't' are mutually exclusive"
            )
        if not _is_default(t0) and not _is_default(t):
            raise ValueError(
                "Environment: too many arguments given to determine t array: arguments 't0' and 't' are mutually exclusive"
            )
        if not _is_default(t1) and not _is_default(t):
            raise ValueError(
                "Environment: too many arguments given to determine t array: arguments 't1' and 't' are mutually exclusive"
            )
        if bitrate is not None and bitlength is not None:
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
            elif not _is_default(num_t):
                dt = float((t1 - t0) / num_t)
            try:
                t = _array(np.arange(min(t0, t1), max(t0, t1), abs(dt)))
            except ValueError:
                raise ValueError(
                    "Cannot create time range. Are dt or num_t, t0 and t1 all specified?"
                )
        if freqdomain:
            t = t[:1]

        if (not _is_default(wl) or not _is_default(f)) or sum(
            [not _is_default(v) for v in (num_wl, num_f, dwl, df, wl0, f0, wl1, f1)]
        ) == 0:
            if not _is_default(f):
                wl = c / f
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
            if not _is_default(df):
                if not _is_default(wl0):
                    f1 = c / wl0
                f1 = float(f1)
                if not _is_default(wl1):
                    f0 = c / wl0
                f0 = float(f0)
                try:
                    wl = c / np.arange(max(f0, f1), min(f0, f1), -abs(df))
                except ValueError:
                    raise ValueError(
                        "Cannot create frequency range. Are df or num_f, f0 and f1 all specified?"
                    )
            else:
                if not _is_default(f0):
                    wl1 = c / f0
                wl1 = float(wl1)
                if not _is_default(f0):
                    wl0 = c / f1
                wl0 = float(wl0)
                if not _is_default(num_wl):
                    dwl = float((wl1 - wl0) / num_wl)
                elif not _is_default(num_f):
                    dwl = float((wl1 - wl0) / num_f)
                dwl = float(dwl)
                try:
                    wl = np.arange(min(wl0, wl1), max(wl0, wl1), abs(dwl))
                except ValueError:
                    raise ValueError(
                        "Cannot create frequency range. Are dwl or num_wl, wl0 and wl1 all specified?"
                    )
        if wl[0] > wl[-1]:
            wl = wl[::-1].copy()

        self.dt = self.timestep = np.inf if t.shape[0] < 2 else float(t[1] - t[0])
        self.samplerate = 0 if t.shape[0] < 2 else float(1 / self.dt)
        self.num_t = self.num_timesteps = int(t.shape[0])
        self.t0 = self.t_start = float(t[0])
        self.t1 = self.t_end = np.inf if t.shape[0] < 2 else float(t[-1]) + self.dt
        self.t = self.time = _array(t)
        self.bitrate = None if bitrate is None else float(bitrate)
        self.bitlength = None if bitrate is None else float(1 / bitrate)
        self.wl = self.wavelength = _array(wl)
        self.f = _array(c / wl)
        self.dwl = self.wavelength_step = (
            np.inf if wl.shape[0] < 2 else float(wl[1] - wl[0])
        )
        self.df = np.inf if wl.shape[0] < 2 else float(c / wl[1] - c / wl[0])
        self.num_wl = self.num_wavelengths = int(wl.shape[0])
        self.num_f = int(wl.shape[0])
        self.wl0 = self.wavelength_start = float(wl[0])
        self.f0 = float(c / wl[0])
        self.wl1 = self.wavelength_end = (
            np.inf if wl.shape[0] < 2 else float(wl[-1]) + self.dwl
        )
        self.f1 = np.inf if wl.shape[0] < 2 else float(c / (wl[-1] + self.dwl))
        self.c = float(c)
        self.freqdomain = self.frequency_domain = bool(freqdomain)
        self.grad = self.enable_grad = bool(grad)
        self.name = str(name)
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
            dt (float): tstep of the simulation (mutually exclusive with t, samplerate and num_t)
            samplerate (float): samplerate of the simulation (mutually exclusive with t, dt and num_t).
            num_t (int): number of tsteps in the simulation (mutually exclusive with t, dt and samplerate).
            t0 (float): starting time of the simulation (mutually exclusive with t).
            t1 (float): ending time of the simulation (mutually exclusive with t).
            t (np.ndarray): full 1D t array (mutually exclusive with dt, t0, t1, num_t and samplerate).
            bitrate (optional, float): bitrate of the signal (mutually exclusive with bitlength).
            bitlength (optional, float): bitlength of the signal (mutually exclusive with bitrate).
            wl (float): wavelength(s) to simulate for (mutually exclusive with f, dwl, df, num_wl, num_f, wl0, f0, wl1 and f1).
            f (float): frequencie(s) to simulate for (mutually exclusive with wl, dwl, df, num_wl, num_f, wl0, f0, wl1 and f1).
            dwl (float): wavelength step between wl0 and wl1 (mutually exclusive with wl, f, df, num_wl and num_f).
            df (float): frequency step between f0 and f1 (mutually exclusive with wl, f, dwl, num_wl and num_f).
            num_wl (int): number of independent wavelengths in the simulation (mutually exclusive with wl, f, num_f, dwl and df)
            num_f (int): number of independent frequencies in the simulation (mutually exclusive with wl, f, num_wl, dwl and df)
            wl0 (float): start of wavelength range (mutually exclusive with wl, f and f0).
            f0 (float): start of frequency range (mutually exclusive with wl, f and wl0).
            wl1 (float): end of wavelength range (mutually exclusive with wl, f and f1).
            f1 (float): end of frequency range (mutually exclusive with wl, f and wl1).
            c (float): speed of light used during simulations.
            freqdomain (bool): only do frequency domain calculations.
            grad (bool): track gradients during the simulation (set this to True during training.)
            name (optional, str): name of the environment
            **kwargs (optional): any number of extra keyword arguments will be stored as attributes to the environment.
        """
        new = {
            k: _convert_default(v)
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
        env = env.copy(**kargs)
    else:
        env = Environment(**kwargs)

    env.__enter__()
