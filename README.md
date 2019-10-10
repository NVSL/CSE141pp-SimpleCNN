# Canela

Canela is a simple convolutional neural network library.  It's main goal is to be easy to understand and optimize.

It was developed for teaching purposes.

## Setup and Installation

To setup Canela, first setup your environment

```
make
```

Then, to start hacking:

```
source env.sh
```

To run the tests and build the examples:

```
make all
```

## Dependencies

Canela relies on `googletest` and uses the build system from `libarchlab`.

The utilities depend on and  `libpng`, `libjpeg` which should be installed by default on most systems.

## Finding Your Way Around

Here's where you'll find the parts of Cenala:

1.  `tests` -- the Canela test suite
2.  `examples` -- example code.
3.  `CNN` -- The core Canela source code.
4.  `util` -- Utility and helper functions (e.g., image loaders)
5.  `datasets` -- sample data sets.

## Credits

Canela is based on https://github.com/can1357/simple_cnn.

