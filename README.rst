##############################################################################
Gaussian Blur
##############################################################################

==============================================================================
Getting Started
==============================================================================

::

    $ cd blur
    $ mkdir _build
    $ make
    $ _build/test_gaussian > log
    $ vim log

==============================================================================
Overview
==============================================================================

Features:
- CPU and GPU (CUDA) support

==============================================================================
Algorithms
==============================================================================


Gaussian Kernel Generation

1. kernel width

kernel_width = ceil(0.3 * (sigma / 2 - 1) + 0.8) * gauss_window_factor

kernel_width should be odd: (kw % 2 == 0) -> kw++


2. kernel center = kernel_width / 2

3. exp_coeff = - 1.0 / (sigma * sigma * 2)

4. kernel[i]

kernel[center] = 1

kernel[-i] = kernel[i] = i * i * exp_coeff

5. sum[kernel]: sum each kernel

6. factor = 1 / sum

7. final kernel[i] = kernel[i] * fac


Convolution


- by columns



- by rows
