import math
import numpy as np
import cv2
import os
import serial
import signal
from threading import Thread
from timeit import default_timer as timer
from numba import cuda
from numba import autojit

def cos_fun(d1, d2):
	return (math.cos(float(d1))+math.cos(float(d2)))/2

#the meat and potatoes, the actual function
def center_func_1(x, y, width, height, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, time):
	x_rel = float(x - width/2)								#horizontal distance from center
	y_rel = float(y - height/2)							#vertical distance from center
	dist = math.sqrt(x_rel*x_rel+y_rel*y_rel)		#distance from center
	dist2 = p9*(x_rel + y_rel)*(10.0/(50.0+p8))/255.0
	#dist = math.floor(dist/10)*10
	dist = dist/(p4/5.0+10)
	env = math.sin(float(time)*p10/100.0)
	r = (math.cos(env+dist*(1.0-(p5/500.0)))+math.cos(dist2*(1.0-(p1/500.0))))/2
	g = (math.cos(env+dist*(1.0-(p6/500.0)))+math.cos(dist2*(1.0-(p2/500.0))))/2
	b = (math.cos(env+dist*(1.0-(p7/500.0)))+math.cos(dist2*(1.0-(p3/500.0))))/2
	#return the pixel
	return (r,g,b)

#the meat and potatoes, the actual function
def center_func_2(x, y, width, height, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, time):
	x_rel_o = float(x - width/2)								#horizontal distance from center
	y_rel_o = float(y - height/2)							#vertical distance from center
	#rotate
	a = float(time)*p9/10000.0
	x_rel = x_rel_o*math.cos(a) - y_rel_o*math.sin(a)
	y_rel = y_rel_o*math.cos(a) + x_rel_o*math.sin(a)

	dist = math.sqrt(x_rel*x_rel+y_rel*y_rel)		#distance from center
	dist2 = p9*(x_rel + y_rel)*(10.0/(50.0+p8))/255.0
	#dist = math.floor(dist/10)*10
	dist = dist/(p4/5.0+10)
	env = math.sin(float(time)*p10/100.0)
	r = (math.cos(env+dist*(1.0-(p5/500.0)))+math.cos(dist2*(1.0-(p1/500.0))))/2
	g = (math.cos(env+dist*(1.0-(p6/500.0)))+math.cos(dist2*(1.0-(p2/500.0))))/2
	b = (math.cos(env+dist*(1.0-(p7/500.0)))+math.cos(dist2*(1.0-(p3/500.0))))/2
	#return the pixel
	return (r,g,b)

def center_func_3(x, y, width, height, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, time):
	x_rel = float(x - width/2)								#horizontal distance from center
	y_rel = float(y - height/2)							#vertical distance from center
	dist = math.sqrt(x_rel*x_rel+y_rel*y_rel)		#distance from center
	dist = p1*dist / 100000000

	r = (math.floor(1/dist)%1000)/1000.0
	g = (math.floor(1/dist-p2/10.0)%1000)/1000.0
	b = (math.floor(1/dist-p3/10.0)%1000)/1000.0
	#return the pixel
	return (r,g,b)

center_gpu = cuda.jit(device=True)(center_func_2)

def sigint_handler(signal, frame):
	global running
	print('Quitting')
	running = False
	serial_thread.join()
	exit(0)
signal.signal(signal.SIGINT, sigint_handler)

(i_width, i_height) = (1080/2,1920/2)

# def create_image(image):
# 	width = image.shape[0]
# 	height = image.shape[1]
#
# 	for y in xrange(0, height):
# 		for x in xrange(0, width):
# 			image[x][y] = center_func(x,y,width,height)

@cuda.jit
def center_kernel(image, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, time):
	width = image.shape[0]
	height = image.shape[1]

	startX, startY = cuda.grid(2)
	gridX = cuda.gridDim.x * cuda.blockDim.x;
	gridY = cuda.gridDim.y * cuda.blockDim.y;

	for y in range(startY, height, gridY):
		for x in range(startX, width, gridX):
			image[x][y] = center_gpu(x,y,width,height, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, time)

#get params from raw line data from arduino
buf = ""
old_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
def get_params(arduino):
	global buf
	global old_params
	while 1:
		chars = arduino.readline()
		buf += chars
		#repeat until we have some characters
		if len(buf) == 0:
			continue
		#process the line
		while 1:
			params = []
			try:
				i = buf.index('\n')
				line = buf[:i]
				buf = buf[i+1:]		#skip newline
				pstring = line.replace('\r', '').split(' ')
				if len(pstring) < 10:
					break
				for i in xrange(0, 10):
					new = float(pstring[i])
					#remove jitter
					if abs(new-old_params[i]) < 5:
						params.append(old_params[i])
					else:
						params.append(new*0.6+old_params[i]*0.4)
				break
			except ValueError:
				break
		if len(params) > 0:
			old_params = params
			return params

def serial_thread_fn():
	global params
	global running
	#get first ttyACM* serial port
	serial_port = '/dev/ttyACM0'
	for l in os.listdir('/dev/'):
		if l.startswith('ttyACM'):
			serial_port = '/dev/' + l
	with serial.Serial(serial_port, baudrate=115200, timeout=0) as arduino:
		#skip garbage data in beginning
		for i in xrange(0, 50):
			get_params(arduino)
		#start the loop
		while running:
			params = get_params(arduino)
	exit(0)

#put arduino controller on separate thread
serial_thread = Thread(target = serial_thread_fn)
running = True
serial_thread.start()
params = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#setup CUDA acceleration and image / OpenCV
gimage = np.zeros((i_width, i_height, 3))
blockdim = (32, 8)
griddim = (32,16)
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN, 1)

time = 0
while 1:
	start = timer()
	d_image = cuda.to_device(gimage)
	center_kernel[griddim, blockdim](d_image, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], time)
	d_image.to_host()
	dt = timer() - start

	print "Mandelbrot created on GPU in %f s" % dt

	cv2.imshow('window', gimage)
	cv2.waitKey(1)
	time += 1
