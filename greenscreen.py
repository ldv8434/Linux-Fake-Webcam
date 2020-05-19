import os
import cv2
import numpy as np
import pyfakewebcam
import tensorflow as tf


REAL_CAMERA = '/dev/video3'
REAL_CAMERA_WIDTH
REAL_CAMERA_HEIGHT
REAL_CAMERA_FPS
FAKE_CAMERA = '/dev/video2'

# OUTDATED
# from keras.models import load_model

# find the GPU we want to work with
devices = tf.config.list_physical_devices()
for dev in devices:
	if dev.device_type == 'GPU':
		tf.config.experimental.set_memory_growth(dev, True)

# OUTDATED
# model = load_model('deconv_bnoptimized_munet.h5', compile=False)


keras_model = tf.keras.models.load_model('deconv_bnoptimized_munet.h5', compile=True)

def get_mask(frame):
	# Preprocess
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	simg = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
	simg = simg.reshape((1, 128, 128, 3)) / 255.0

	# Predict
	out = keras_model.predict(simg)

	# Postprocess
	msk = out.reshape((128, 128, 1))
	mask = cv2.resize(msk, (frame.shape[1], frame.shape[0]))

	return  mask


def post_process_mask(mask):
	mask = cv2.dilate(mask, np.ones((10,10), np.uint8) , iterations=1)
	mask = cv2.blur(mask.astype(float), (30,30))
	return mask


def shift_image(img, dx, dy):
	img = np.roll(img, dy, axis=0)
	img = np.roll(img, dx, axis=1)
	if dy>0:
		img[:dy, :] = 0
	elif dy<0:
		img[dy:, :] = 0
	if dx>0:
		img[:, :dx] = 0
	elif dx<0:
		img[:, dx:] = 0
	return img


def hologram_effect(img):
	# add a blue tint
	holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
	
	# add a halftone effect
	bandLength, bandGap = 2, 3
	for y in range(holo.shape[0]):
		if y % (bandLength+bandGap) < bandLength:
			holo[y,:,:] = holo[y,:,:] * np.random.uniform(0.1, 0.3)

	# add some ghosting
	holo_blur = cv2.addWeighted(holo, 0.2, shift_image(holo.copy(), 5, 5), 0.8, 0)
	holo_blur = cv2.addWeighted(holo_blur, 0.4, shift_image(holo.copy(), -5, -5), 0.6, 0)

	# combine with the original color, oversaturated
	out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
	return out


def get_frame(cap, background_scaled):
	_, frame = cap.read()

	# fetch the mask with retries (the app needs to warmup and we're lazy)
	# e v e n t u a l l y c o n s i s t e n t
	mask = None
	while mask is None:
		try:
			mask = get_mask(frame)
		except requests.RequestException:
			print("mask request failed, retrying")

	# post-process mask and frame
	#mask = post_process_mask(mask)
	#frame = hologram_effect(frame)

	# composite the foreground and background
	inv_mask = 1-mask
	for c in range(frame.shape[2]):
		frame[:,:,c] = frame[:,:,c]*mask + background_scaled[:,:,c]*inv_mask
	return frame


# setup access to the *real* webcam
cap = cv2.VideoCapture(REAL_CAMERA)
height, width = REAL_CAMERA_HEIGHT, REAL_CAMERA_WIDTH
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, REAL_CAMERA_FPS)


# setup the fake camera
fake = pyfakewebcam.FakeWebcam(FAKE_CAMERA, width, height)

# load the virtual background
background = cv2.imread("chroma.jpg")
background_scaled = cv2.resize(background, (width, height))

# frames forever
while True:
	frame = get_frame(cap, background_scaled)
	# fake webcam expects RGB
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	print("Printed frame")
	fake.schedule_frame(frame)
