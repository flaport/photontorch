components
==========

Each Component is generally defined by several key attributes defining the
behavior of the component in a network.

* `num_ports`: The number of ports of the components.

* `S`: The scattering matrix of the component.

* `C`: The connection matrix for the component (usually all zero for base
  components)

* `sources_at`: The location of the sources in the component (usually all zero
  for base components)

* `detectors_at`: The location of the detectors in the component (usually all
  zero for base components)

* `actions_at`: The location of the active nodes in the component (usually all
  zero for passive components)

* `delays`: delays introduced by the nodes of the component.

Defining your own Component comes down to subclassing `Component` and
redefining the relevant setters `set_*` for these attributes. For example::

    class Waveguide(pt.Component):
        num_ports = 2
        def __init__(self, length=1e-5, neff=2.34, ng=3.40, name=None):
            super(Waveguide, self).__init__(self, name=name)
            self.neff = float(neff)
            self.wl0 = float(wl0)
            self.ng = float(ng)
            self.length = float(length)
        def set_delays(self, delays):
            delays[:] = self.ng * self.length / self.env.c
        def set_S(self, S):
            wls = torch.tensor(self.env.wl, dtype=torch.float64, device=self.device)
            phase = (2 * np.pi * neff * self.length / wls) % (2 * np.pi)
            S[0, :, 0, 1] = S[0, :, 1, 0] = torch.cos(phase).to(torch.float32) # real part
            S[1, :, 0, 1] = S[1, :, 1, 0] = torch.sin(phase).to(torch.float32) # imag part

