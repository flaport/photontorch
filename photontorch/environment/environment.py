"""
# PhotonTorch Simulation Environment

The simulation environment module contains a single class: Environment

This class contains all the necessary parameters to initialize a network for a simulation.

"""

#############
## Imports ##
#############

# Standard Library
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


#################
## Environment ##
#################


class Environment(dict):
    """ Simulation Environment

    The simulation environment is a smart data class that contains all
    the necessary parameters to initialize a network for a simulation.

    It is able to initialize parameters that depend on each other correctly.
    Changing a parameter that depends on another, changes both.


    Attributes:
        c: float: speed of light
        t: np.ndarray: array with the times of the simulation
        wls: np.ndarray: array with all the wavelengths of the simulation
        use_delays: bool: boolean signifying to use the component delays or not
        cuda: bool: boolean signifying to use cuda for the simulations or not
    """

    _initialized = False

    def __init__(self, **kwargs):

        """
        Args:
            dt: float = 1e-14: timestep of the simulation
            t_start: float = 0: start time of the simulation in seconds
            t_end: float = 1000*dt: end time of the simulation in seconds
            num_timesteps: int = 1000: number of timesteps in the simulation (use this
                in stead of t_end.)
            t: np.ndarray = np.arange(0, 1000*dt, dt): array with the times of the
                simulation (use this in stead of dt, t_start, t_end and num_timesteps)

            wavelength: float or np.ndarray = np.array([1.55e-6]): wavelength(s) of the
                simulation
            num_wavelengths: int = 1: number of wavelengths in the simulation
            wavelength_start: float = 1.5e-6: starting wavelength of the simulation
            wavelength_end: float = 1.6e-6: ending wavelength of the simulation

            frequency_domain: bool = True: ignore the delays in the network to
                perform a frequency domain simulation.

            device: str = None: which device to do the simulation on. If `None`, the
                default of the network is used.

            c: float = 299792458.0: speed of light used for the simulations

            name: str = "env": the name of the environment

        Note:
            The environment can take unlimited extra keyword arguments, which will be
            stored as attributes.
        """

        ## Handle constants and booleans
        self["c"] = kwargs.pop("c", 299792458.0)  # speed of light
        use_delays = kwargs.pop("use_delays", True)  # for backward compatibility
        self["frequency_domain"] = kwargs.pop("frequency_domain", not use_delays)
        self["name"] = kwargs.pop("name", "env")

        ## Simulation device
        # for backward compatibility:
        device = {None: None, True: "cuda", False: "cpu"}[kwargs.pop("cuda", None)]
        # select device
        self["device"] = kwargs.pop("device", device)

        ## Handle wavelength
        # create default wavelength:
        wl_start = float(
            kwargs.pop("wavelength_start", kwargs.pop("wl_start", 1.55e-6))
        )
        wl_end = float(kwargs.pop("wavelength_end", kwargs.pop("wl_end", 1.55e-6)))
        num_wavelengths = int(kwargs.pop("num_wavelengths", kwargs.pop("num_wl", 1)))
        default_wavelength = np.linspace(wl_start, wl_end, num_wavelengths)
        # get wavelength (we leave the other options for backward compatibility)
        wavelength = np.array(
            kwargs.pop(
                "wavelength", kwargs.pop("wl", kwargs.pop("wls", default_wavelength))
            )
        )
        if wavelength.ndim == 0:
            wavelength = wavelength[None]  # make wavelength a 1D array
        elif wavelength.ndim > 1:
            raise ValueError("wavelength should be a float or a  1D array")
        # set environment attributes
        self["wavelength"] = wavelength
        self["wavelength_start"] = wavelength[0]
        self["wavelength_end"] = wavelength[-1]
        self["num_wavelengths"] = wavelength.shape[0]
        if len(wavelength) > 1:
            self["wavelength_step"] = wavelength[1] - wavelength[0]

        ## Handle time
        # create default time:
        dt = kwargs.pop("dt", None)
        t_start = float(kwargs.pop("t_start", 0))
        if self["frequency_domain"]:
            num_timesteps = 1
        else:
            num_timesteps = kwargs.pop("num_timesteps", 1000)
        if dt is None:
            t_end = float(kwargs.pop("t_end", 1e-12))
            dt = t_end / num_timesteps
        else:
            t_end = float(kwargs.pop("t_end", num_timesteps * dt))

        if dt > t_end:
            t_end = 1.6 * dt

        default_time = np.arange(t_start, t_end - 0.5 * dt, dt)
        # get time (leave the other options for backward compatibility)
        time = np.array(kwargs.pop("time", kwargs.pop("t", default_time)))
        if time.ndim == 0:
            time = time[None]
        elif time.ndim > 1:
            raise ValueError("time should be a float or a 1D array")
        self["time"] = time
        self["t_start"] = time[0]
        self["t_end"] = time[-1] + dt
        self["dt"] = dt if time.shape[0] == 1 else time[1] - time[0]
        self["num_timesteps"] = time.shape[0]

        # we need this check as long as there are no complex tensors in torch, as
        # having only two timesteps might yield problems. (the two time steps might
        # be misinterpreted as the real and imaginary part of a single timestep)
        if self["num_timesteps"] == 2:
            raise ValueError(
                "A simulation with two timesteps is for the moment not allowed."
                "Take a single timestep OR at least 3 timesteps."
            )

        self["enable_grad"] = kwargs.pop("enable_grad", not kwargs.pop("no_grad", True))
        if self["enable_grad"]:
            self["grad_manager"] = torch.enable_grad()
        else:
            self["grad_manager"] = torch.no_grad()

        # other keyword arguments are added to the attributes:
        super(Environment, self).__init__(**kwargs)

        # add keywords as attributes
        self.__dict__ = self

        # finish initialization
        self._initialized = True

    def copy(self, **kwargs):
        """ Create a (deep) copy of the environment

        Note:
            You can optionally change the copied environment attributes by specifying
            keyword arguments.
        """
        new = deepcopy(dict(self))
        del new["_initialized"]
        del new["grad_manager"]
        new.update(kwargs)

        if "dt" in kwargs or "num_timesteps" in kwargs:
            if "dt" in kwargs:
                if new["dt"] > new["t_end"]:
                    new["t_end"] = new["dt"]
            if "num_timesteps" not in kwargs:
                new["time"] = np.arange(
                    new["t_start"], new["t_end"] - 0.5 * new["dt"], new["dt"]
                )
            else:
                if "dt" not in kwargs and new["num_timesteps"] > 1:
                    new["dt"] = new["time"][1] - new["time"][0]
                new["t_end"] = new["t_start"] + new["num_timesteps"] * new["dt"]
                new["time"] = np.arange(
                    new["t_start"], new["t_end"] - 0.5 * new["dt"], new["dt"]
                )

        return self.__class__(**new)

    def close(self):
        if _current_environments[0] is self:
            del _current_environments[0]
        self.grad_manager.__exit__()

    def _set(self):
        _current_environments.appendleft(self)
        self.grad_manager.__enter__()
        return self

    def __enter__(self):
        """ enter the with block """
        return self._set()

    def __exit__(self, error, value, traceback):
        """ exit the with block """
        self.close()
        if error is not None:
            raise  # raise the last error thrown
        return self

    def __setattr__(self, name, value):
        """ this locks the attributes """
        if self._initialized:
            raise AttributeError(
                "Changing the attributes of an environment is not allowed. "
                "Consider creating a new environment or use the copy method."
            )
        else:
            super(Environment, self).__setattr__(name, value)

    def __setitem__(self, name, value):
        """ this locks the attributes """
        if self._initialized:
            raise AttributeError(
                "Changing the attributes of an environment is not allowed. "
                "Consider creating a new environment or use the copy method."
            )
        else:
            super(Environment, self).__setitem__(name, value)

    def __eq__(self, other):
        """ check if two environments are equal to each other """
        if not isinstance(other, Environment):
            return False
        d1 = dict(self)
        del d1["time"]
        del d1["wavelength"]
        d2 = dict(other)
        del d2["time"]
        del d2["wavelength"]
        return d1 == d2

    def __to_str_dict(self):
        """ Dictionary representation of the environment """
        env = OrderedDict()
        env["name"] = repr(self.name)
        env["frequency_domain"] = self.frequency_domain
        if not self.frequency_domain:
            env["t_start"] = "%.2e" % self.t_start
            env["t_end"] = "%.2e" % self.t_end
            env["dt"] = "%.2e" % self.dt
            env["num_timesteps"] = self.num_timesteps
        if self.num_wavelengths > 1:
            env["wavelength_start"] = "%.3e" % self.wavelength_start
            env["wavelength_end"] = "%.3e" % self.wavelength_end
            env["num_wavelengths"] = repr(self.num_wavelengths)
        else:
            env["wavelength"] = "%.3e" % self.wavelength[0]
        env.update({k: repr(v) for k, v in self.__dict__.items() if k[0] != "_"})
        del env["wavelength"]
        del env["time"]
        return env

    def __repr__(self):
        """ Get the string representation of the environment """
        dic = self.__to_str_dict()
        return (
            "Environment(" + ", ".join("=".join([k, v]) for k, v in dic.items()) + ")"
        )

    def __str__(self):
        """ Set the string representation of the environment """
        dic = self.__to_str_dict()
        s = "Simulation Environment:\n"
        s = s + "-----------------------\n"
        for k, v in dic.items():
            s = s + "%s = %s\n" % (str(k), str(v))
        return s


#########################
## Current Environment ##
#########################


def current_environment():
    """ get the current environment """
    if _current_environments:
        return _current_environments[0]
    else:
        raise RuntimeError(
            "Environment is not set. Execute your code inside an "
            "environment-defining with-block or use the 'set_environment()' "
            "function to set your environment globally."
        )


def set_environment(*args, **kwargs):
    """ set the current environment

    Args:
        env: Environment: The environment to set globally.

    Kwargs:
        **kwargs: The keyword arguments to create a new environment to set globally

    Note:
        It is recommended to set the Environment using a with-block. However, if you
        would like to set the environment globally, this can be done with this function.
    """
    if len(args) > 2:
        raise ValueError("Only one positional argument allowed")
    elif len(args) == 1:
        env = args[0]
    elif "env" in kwargs:
        env = kwargs["env"]
    else:
        env = Environment(**kwargs)

    env._set()
