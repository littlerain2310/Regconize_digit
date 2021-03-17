# import the necessary packages
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import cv2
import numpy as np
import matplotlib.pyplot as plt

# load the example image
image = cv2.imread("pic1.jpg")


DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)


## find the non-zero min-max coords of canny
pts = np.argwhere(edged>0)
y1,x1 = pts.min(axis=0)
y2,x2 = pts.max(axis=0)

## crop the region
cropped = blurred[y1:y2, x1-10:x2-20]
output = image[y1:y2, x1-10:x2-20]

def crop1():
	(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	idx = 0
	for c in cnts:
		x,y,w,h = cv2.boundingRect(c)
		idx+=1
		new_img=image[y:y+h,x:x+w]
		cv2.imshow('image',new_img)
		cv2.waitKey(0)
thresh = cv2.threshold(cropped, 127, 255,0)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


# # find contours in the thresholded image, then initialize the
# # digit contours lists
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []
# # loop over the digit area candidates
for c in cnts:
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)
	print("x la {} , y la {} ,w la {},h la {}".format(x,y,w,h))
	# if the contour is sufficiently large, it must be a digit
	if w >= 15 and (h >= 30 and h <= 45):
		digitCnts.append(c)


# # sort the contours from left-to-right, then initialize the
# # actual digits themselves
digitCnts = contours.sort_contours(digitCnts,
	method="left-to-right")[0]
digits = []
roi1 = []
idx =0

def CV():
	for c in digitCnts:
		# extract the digit ROI
		(x, y, w, h) = cv2.boundingRect(c)
		if x == 33:
			roi = thresh[y:y + h, x:x + w]
			image_= output[y:y + h, x:x + w]
		else:
			roi = thresh[y:y + h, x-5:x + w]
			image_= output[y:y + h, x-5:x + w]
		cv2.imshow("image_",image_)
		print(image_.shape)
		# print(image_.shape)
		image_ = cv2. cvtColor(image_, cv2.COLOR_BGR2GRAY)
		image_ = np.expand_dims(image_, 2)
		cv2.waitKey(0)
		# image_ = np.expand_dims(image_, 2)
		print(image_.shape)
		# image_ = image_.reshape(38,20,1)
		

		# x= image
		print(image_.shape)
		

		# compute the width and height of each of the 7 segments
		# we are going to examine
		(roiH, roiW) = roi.shape
		(dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
		dHC = int(roiH * 0.05)
		# define the set of 7 segments
		segments = [
			((0, 0), (w, dH)),	# top
			((0, 0), (dW, h // 2)),	# top-left
			((w - dW, 0), (w, h // 2)),	# top-right
			((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
			((0, h // 2), (dW, h)),	# bottom-left
			((w - dW, h // 2), (w, h)),	# bottom-right
			((0, h - dH), (w, h))	# bottom
		]
		on = [0] * len(segments)
		for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
			# extract the segment ROI, count the total number of
			# thresholded pixels in the segment, and then compute
			# the area of the segment
			segROI = roi[yA:yB, xA:xB]
			total = cv2.countNonZero(segROI)
			area = (xB - xA) * (yB - yA)
			# if the total number of non-zero pixels is greater than
			# 50% of the area, mark the segment as "on"
			if total / float(area) > 0.5:
				on[i]= 1
		# lookup the digit and draw it on the image
		digit = DIGITS_LOOKUP[tuple(on)]
		digits.append(digit)
		print(digit)
		cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
		cv2.putText(output, str(digit), (x , y  +20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 2)
	# display the digits
	# cv2.putText(output, u"{}{} \u00b0C".format(*digits), (x -10 , y  -10),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
	print(u"{}{} \u00b0C".format(*digits))
	cv2.imshow("Output", image)
	# cv2.imshow("Output", output)
	cv2.waitKey(0)

		


CV()