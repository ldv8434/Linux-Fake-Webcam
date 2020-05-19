# Linux-Fake-Webcam
Emulates webcam
**Please forgive how much of a hackjob this is. I am not an experienced programmer, and I have almost no experience in machine learning/tensorflow. This was made as a test to see how effective it would be for live-streams**

# Credits
## Ben Elder
https://elder.dev/posts/open-source-virtual-background/
Wrote the original script

## rogierhofboer on HackerNews
https://news.ycombinator.com/item?id=22823070
Translated the get_mask function to Python

## Anil Sathyan aka anilsathyan7
https://github.com/anilsathyan7/Portrait-Segmentation
Model file used

## Fangfufu on Github
https://github.com/fangfufu/Linux-Fake-Background-Webcam
Guidance on getting v4l2loopback working

## Lukas Vacula aka ldv8434 aka me
The original get_mask function from rogierhofboer only worked under Tensorflow v1. I've ported it to v2, and made some other adjustments to the original script from Ben Elder.

# Requirements (based on arch repo names)
python-tensorflow-cuda 2.2.0-1
tensorflow-cuda 2.2.0-1
pyfakewebcam 0.1.0 (installed via pip)

# usage
Please refer to https://github.com/fangfufu/Linux-Fake-Background-Webcam for setting up the virtual webcam device
You must edit greenscreen.py with the height and width of your real camera, as well as point to the real and fake cameras' locations. After that, it should be able to run on its own. 
Also uses chroma.jpg as the image replacement by default. 
