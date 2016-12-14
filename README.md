# mininet

[![Build Status](https://travis-ci.org/dfridovi/mininet.svg?branch=master)](https://travis-ci.org/dfridovi/mininet)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](https://github.com/dfridovi/mininet/LICENSE)

A lightweight C++ framework for deep learning. **mininet** is written by [David Fridovich-Keil](http://people.eecs.berkeley.edu/~dfk/), a second-year PhD student in the [Berkeley Artificial Intelligence Research (BAIR) Lab](http://bair.berkeley.edu), and Sara Fridovich-Keil, a junior electrical engineering student student at Princeton.

## Status
**mininet** is still under active development. We hope to have a first release soon though, so stay tuned!

## Structure
All source code is located in `src/`; headers are in `include/`; unit tests are in `/test/`; and executables are in `exec/`. Compiled binaries will be placed in `bin/`.

## Dependencies
I may miss a few here, but here is a list of dependencies:

* [Eigen](http://eigen.tuxfamily.org/dox/) (header-only linear algebra library)
* Gflags (Google's command-line flag manager)
* Glog (Google's logging tool)

All of these may be installed very easily. If you run into any trouble, though, we are more than happy to help you figure out what's going on. Just post an [issue](https://github.com/dfridovi/mininet/issues) on this repository and we will reply as soon as possible.

## Usage
You'll need to begin by building the repository. From the top directory, type the following sequence of commands:

```
mkdir bin
mkdir build
cd build
cmake ..
make -j4
```

This should build all tests and executables. In order to run tests, you can run the following command:

```
./run_tests
```

from within the `build/` directory you just made. All the tests should pass, and none should take more than a second or so to run.

Executables are automatically placed within the `bin/` directory that you created. To run them, just type `./(name-of-executable)`.

To the extent that it makes sense, all parameters are accessible from the command line via Gflags. For help with command line options, simply run the following command:

```
./(name-of-executable) --help
```

## API documentation
We use Doxygen to auto-generate web-based [documentation](https://dfridovi.github.io/mininet/documentation/html/). Although we do not follow the Doxygen guidelines for writing comments, auto-generation still seems to do a fairly reasonable job.
