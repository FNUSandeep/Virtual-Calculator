import cv2
import numpy as np
import pickle

def squares_in_image(image):
	x_co, y_co, width, height = 420, 140, 10, 10
	d = 10
	crop_image = None
	for i in range(10):
		for j in range(5):
			if np.any(crop_image == None):
				t= y_co+height
				q= x_co+width
				crop_image = image[y_co:t, x_co:q]
			else:
				t= y_co+height
				q= x_co+width
				crop_image = np.vstack((crop_image, image[y_co:t, x_co:q]))
			cv2.rectangle(image, (x_co,y_co), (q, t), (0,255,0), 2)
			x_co+=width+d
		x_co = 420
		y_co+=height+d
	return crop_image

def get_histogram_value():
	cam = cv2.VideoCapture(0)
	x_co, y_co, width, height = 300, 100, 300, 300
	flag_c, flag_s = False, False
	while True:
		image = cam.read()[1]
		image = cv2.flip(image, 1)
		image_hsv_b2w = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		
		keypress = cv2.waitKey(1)
		if keypress == ord('c'):		
			image_hsv_b2wCrop = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
			flag_c = True
			histogram_value = cv2.calcHist([image_hsv_b2wCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
			cv2.normalize(histogram_value, histogram_value, 0, 255, cv2.NORM_MINMAX)
		elif keypress == ord('s'):
			flag_s = True	
			break
		if flag_c:	
			dst_image = cv2.calcBackProject([image_hsv_b2w], [0, 1], histogram_value, [0, 180, 0, 256], 1)
			disc_image = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
			cv2.filter2D(dst_image,-1,disc_image,dst_image)
			blured_image = cv2.GaussianBlur(dst_image, (11,11), 0)
			blured_image = cv2.medianBlur(blured_image, 15)
			ret,thresh_image = cv2.threshold(blured_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			thresh_image = cv2.merge((thresh_image,thresh_image,thresh_image))
			#thresh = cv2.merge((thresh,thresh,thresh))
			res = cv2.bitwise_and(image,thresh_image)
			#cv2.imshow("res", res)
			cv2.imshow("Thresh", thresh_image)
		if not flag_s:
			crop_image = squares_in_image(image)
		#cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
		cv2.imshow("Set hand histogram_valueogram", image)
	cam.release()
	cv2.destroyAllWindows()
	with open("hist", "wb") as f:
		pickle.dump(histogram_value, f)


get_histogram_value()