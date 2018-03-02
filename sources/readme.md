# Sources

A collection of sources that can be created by a Network.

## Implemented so far:

  * ConstantSource
  * TimeSource
  * BitSource


## Note about the way to create sources:

### Recommended:

Sources should be created from an initialized network. The recommended
way to create a source is as follows:

```
    env = Environment(...)
    nw = Network(...)
    nw.initialize(env)
    source = nw.ConstantSource(...)
```

### Not Recommended:
Sources that are imported and created straight from this submodule, such as
```
    from photontorch.sources.sources import ConstantSource
    source = ConstantSource(...)
```
won't contain a reference to the network that was supposed to create it, and will not
be functional.