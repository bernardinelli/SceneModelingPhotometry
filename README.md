# SceneModelingPhotometry
 Scene modeling photometry software for flux measurements of trans-Neptunian objects from the Dark Energy Survey

### Information
 This repository provides the implementation of the scene modeling photometry procedure described in [Bernardinelli et al (2023)](https://arxiv.org/abs/2304.03017), as well as auxiliary tools for saving the data and visualizing the results.

 The implementation in `smp.py` consists of two primary classes, `Detection` and `BinaryDetection`, that fit the background (scene) and the target single (`Detection`) or binary sources (`BinaryDetection`) for a given set of positions. A Jupyter Notebook that provides a few usage examples are also provided.  

### Dependencies
This implementation depends on a few standard Python packages:
- [`numpy`](https://pypi.org/project/numpy/)
- [`astropy`](https://pypi.org/project/astropy/)
- [`scipy`](https://pypi.org/project/scipy/)
- [`compress-pickle`](https://pypi.org/project/compress-pickle/)
- [`matplotlib`](https://pypi.org/project/matplotlib/) for visualization

As well as a few DES-specific packages:
- [`pixmappy`](https://github.com/gbernstein/pixmappy): DES astrometric solutions
- [`Piff`](https://github.com/rmjarvis/Piff): DES point spread functions
- [`destnosim`](https://github.com/bernardinelli/DESTNOSIM): compilation of exposures used in the DES search for TNOs

### Generalization and usage for surveys other than DES
The `Detection` class implements both the scene modeling solutions and the DES-specific data handling. For surveys other than DES, the astrometric solutions, PSF drawing and exposure information might be handled differently. A simple and elegant solution to this problem is to implement a class that inherits from `Detection` with their own `self.findAllExposures`, `self.findPixelCoords` and `self.constructPSFs` methods.

The scene modeling procedure is also quite flexible, and may be generalized to observing strategies and conditions, for example, that of a series of repeated exposures of a given object in a short time span (as in a shift and stack survey). Implementation of these cases can also be handled by inheriting from `Detection` and `BinaryDetection`.