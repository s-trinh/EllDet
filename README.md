# EllDet
An ellipse detector for RGB images implemented in C++. 

This is the code for paper ["Combining Convex Hull and Directed Graph for Fast and Accurate Ellipse Detection"](https://doi.org/10.1016/j.gmod.2021.101110).

(The source code is currently being cleaned and checked, and will be released as soon as possible.)

![Workflow of the ellipse detector.](img/workflowimage.png)

## Requirements
* CMake 3.18
* C++ compiler with C++17 support.
* OpenCV 4.3.0

## Usage
### Setup
* Clone this repo:

```cmd
git clone https://github.com/meiyy/EllDet.git
cd EllDet
```
* Usage CMake to generate the project:

```cmd
mkdir build
cd build
cmake ..
```

* Build the project by `make` (Linux) or `Visual Studio` (Windows).

### Test

Run the following command to detect the ellipses in an given image file:

```cmd
elldet demo.bmp
```
## Dataset

Dataset Tableware will be released here as a zip file.

## Contact

If you have any question about the code, please feel free to contact me.
Email: shenzeyu2018@ia.ac.cn