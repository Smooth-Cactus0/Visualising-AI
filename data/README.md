# data/

`mnist16.bin`: 3,000 training and 1,000 test MNIST digits (300 and 100 per class), area-resampled from 28×28 to 16×16, the input size of the toy network on every page. Made by `tools/make_mnist16.py` (standard-library Python, seed 1): `python tools/make_mnist16.py` regenerates it byte for byte.

Format, little-endian: bytes 0–3 `MN16`; 4–5 number of training digits (uint16); 6–7 number of test digits (uint16); 8 side length (16); 9–15 zero. Then the training labels (one byte each), the training pixels (256 bytes per digit, row-major, 0 = background, 255 = full ink), the test labels, the test pixels. 1,028,016 bytes. Read it with `core/data.js`.

Source: the MNIST database of handwritten digits by Yann LeCun, Corinna Cortes and Christopher J.C. Burges (http://yann.lecun.com/exdb/mnist/), downloaded from the mirror `https://ossci-datasets.s3.amazonaws.com/mnist/`. MNIST is made available under the Creative Commons Attribution-Share Alike 3.0 licence; this subset is a derived work under the same licence, with the same attribution.
