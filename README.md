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

You will need a C++11 compiler.

Canela relies on `googletest` and uses the build system from `libarchlab`.  The `Makefile` will install both of these locally for you.

The utilities depend on and  `libpng` and `libjpeg` which should be installed by default on most systems.

## Finding Your Way Around

Here's where you'll find the parts of Cenala:

1.  `CNN` -- The core Canela source code.  Checkout `CNN/README.md` for details.
2.  `tests` -- the Canela test suite
3.  `examples` -- example code.
4.  `util` -- Utility and helper functions (e.g., image loaders)
5.  `datasets` -- sample data sets.

### Basic Data Types

Canela relies heavily on several basic data types:

* `tensor_t` : 3D array for storing inputs and outputs (defined in
  `CNN/tensor_t.hpp`).

* `model_t` : A container for `layer_t` objects in a CNN model and
  high-level algorithms for training and classification
  (`CNN/model_t.hpp`)

* `layer_t` : Base class for CNN layers.  It defines a consistent
  interface for layers that `model_t` uses to to training and
  classification. (`CNN/layer_t.hpp`)

* `tdsize` : The size of a tensor (x,y,z).  It is a synonym for
  `point_t`. (defined in `CNN/types_t.hpp')

* `range_t` : Represents a rectangular range of a tensor (defined in
  `CNN/range_t.hpp`).

### Layer Types

Canela defines three main types of CNN layers: Fully-connected neural
networks (`CNN/fc_layer_t.hpp`), convolutional layers
(`CNN/conv_layer_t.hpp`), and pooling layers (`CNN/pool_layer_t.hpp`).

In addition it also has several types of "auxillary layers" that
implement common features of CNNs: the relu layer
(`CNN/relu_layer_t.hpp`) implement relu and neural net drop out is
implemented as `dropout_layer_t` (`CNN/drop_layer_t.hpp`).

## Credits

Canela is based on https://github.com/can1357/simple_cnn.

