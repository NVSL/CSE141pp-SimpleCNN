# Canela

![](https://travis-ci.com/NVSL/CSE141pp-SimpleCNN.svg?branch=master)

Canela is a simple convolutional neural network library.  It is, by design, completely unoptimized: The code is
easy to understand but slow.

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

You will need a C++ 11 compiler.

Canela relies on `googletest` and uses the build system from `libarchlab`.  The `Makefile` will install both of these locally for you.

The utilities depend on and  `libpng` and `libjpeg` which should be installed by default on most systems.

## Finding Your Way Around

Here's where you'll find the parts of Cenala:

1.  `CNN` -- The core Canela source code.  Checkout `CNN/README.md` for details.
2.  `tests` -- the Canela test suite
3.  `examples` -- example code.
4.  `util` -- Utility and helper functions (e.g., image loaders)
5.  `datasets` -- sample data sets.


## Credits

Canela is based on https://github.com/can1357/simple_cnn.

