CUDA Accelerated Visual Demo
=============================

![lens](https://github.com/dudeofea/Engg-Carnival/blob/master/lens.png?raw=true)

Just a neat thing that uses potentiometers from an old arduino pedal
connected by usb to modify the visuals shown on screen. The visual are
generated by python code that has been accelerated by CUDA using the
`numba` library.

Installation
------------

Install python 2.7, `numba`, `numpy`, and OpenCV 2. OpenCV doesn't do any
heavy lifting, it's just for displaying multiple image frames. You'll also
need the latest version of CUDA installed, as well as nvidia drivers (I was
using `nvidia-396` and `cuda-9.2`)
