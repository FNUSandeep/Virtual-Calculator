import cv2
import numpy as np
import pickle
import math
import sqlite3
from collections import Counter, deque
from keras.models import load_model

model = load_model('cnn_model_keras2.h5')

def get_size_of_image():
	image = cv2.imread('gestures/0/100.jpg', 0)
	return image.shape

image_x_co, image_y_co = get_size_of_image()

def get_hand_histogram_value():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

def process_image_through_keras(image):
	image = cv2.resize(image, (image_x_co, image_y_co))
	image = np.array(image, dtype=np.float32)
	image = np.reshape(image, (1, image_x_co, image_y_co, 1))
	return image

def keras_predict(model, image):
	processed_image = process_image_through_keras(image)
	pred_probab_of_image = model.predict(processed_image)[0]
	predict_class_of_image = list(pred_probab_of_image).index(max(pred_probab_of_image))
	return max(pred_probab_of_image), predict_class_of_image

def get_predicted_text_from_database(predict_class_of_image):
	conn = sqlite3.connect("gesture_db.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(predict_class_of_image)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

def get_operator_for_calculation(predicted_text):
	try:
		predicted_text = int(predicted_text)
	except:
		return ""
	operator = ""
	if predicted_text == 1:
		operator = "+"
	elif predicted_text == 2:
		operator = "-"
	elif predicted_text == 3:
		operator = "*"
	elif predicted_text == 4:
		operator = "/"
	elif predicted_text == 5:
		operator = "%"
	elif predicted_text == 6:
		operator = "**"
	elif predicted_text == 7:
		operator = ">>"
	elif predicted_text == 8:
		operator = "<<"
	elif predicted_text == 9:
		operator = "&"
	elif predicted_text == 0:
		operator = "|"
	return operator

def start_cal():
	x_co, y_co, width, height = 300, 100, 300, 300
	hist = get_hand_histogram_value()
	flag_calc = {"first": False, "operator": False, "second": False, "clear": False}
	number_of_same_frame = 0
	first, operator, second = "", "", ""
	predicted_text = ""
	calc_text = ""
	information = "Enter first number"
	count_clear_frames = 0
	cam = cv2.VideoCapture(0)
	while True:
		_, image = cam.read()
		image = cv2.flip(image, 1)
		imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		dst_image = cv2.calcBackProject([imageHSV], [0, 1], hist, [0, 180, 0, 256], 1)
		disc_image = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
		cv2.filter2D(dst_image,-1,disc_image,dst_image)
		blured_image = cv2.GaussianBlur(dst_image, (11,11), 0)
		blured_image = cv2.medianBlur(blured_image, 15)
		#ret1,thresh = cv2.threshold(blured_image,127,255,cv2.THRESH_BINARY)
		thresh_image = cv2.threshold(blured_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		thresh_image = cv2.merge((thresh_image,thresh_image,thresh_image))
		thresh_image = cv2.cvtColor(thresh_image, cv2.COLOR_BGR2GRAY)
		t=y_co+height
		q=x_co+width
		thresh_image = thresh_image[y_co:t, x_co:q]
		
		contours_image = cv2.findContours(thresh_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
		old_predicted_text = predicted_text
		if len(contours_image) > 0:
			contour = max(contours_image, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				x1_co, y1_co, width_co, height_co = cv2.boundingRect(contour)
				save_image = thresh_image[y1_co:y1_co+height_co, x1_co:x1_co+width_co]
				if width_co > height_co:
					save_image = cv2.copyMakeBorder(save_image, int((width_co-height_co)/2) , int((width_co-height_co)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
				elif height_co > width_co:
					save_image = cv2.copyMakeBorder(save_image, 0, 0, int((height_co-width_co)/2) , int((height_co-width_co)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
				pred_probab_of_image, predict_class_of_image = keras_predict(model, save_image)
				if pred_probab_of_image*100 > 70:
					predicted_text = get_predicted_text_from_database(predict_class_of_image)

				if old_predicted_text == predicted_text:
					number_of_same_frame += 1
				else:
					number_of_same_frame = 0


				if predicted_text == "C":
					if number_of_same_frame > 5:
						number_of_same_frame = 0
						first, second, operator, predicted_text, calc_text = '', '', '', '', ''
						flag_calc['first'], flag_calc['operator'], flag_calc['second'], flag_calc['clear'] = False, False, False, False
						information = "Enter first number"

				elif predicted_text == "Confirm" and number_of_same_frame > 15:
					number_of_same_frame = 0
					if flag_calc['clear']:
						first, second, operator, predicted_text, calc_text = '', '', '', '', ''
						flag_calc['first'], flag_calc['operator'], flag_calc['second'], flag_calc['clear'] = False, False, False, False
						information = "Enter first number"
					elif second != '':
						flag_calc['second'] = True
						information = "Clear screen"
						second = ''
						flag_calc['clear'] = True
						calc_text += "= "+str(eval(calc_text))
					elif first != '':
						flag_calc['first'] = True
						information = "Enter operator"
						first = ''

				elif predicted_text != "Confirm":
					if flag_calc['first'] == False:
						if number_of_same_frame > 15:
							number_of_same_frame = 0
							first += predicted_text
							calc_text += predicted_text
					elif flag_calc['operator'] == False:
						operator = get_operator_for_calculation(predicted_text)
						if number_of_same_frame > 15:
							number_of_same_frame = 0
							flag_calc['operator'] = True
							calc_text += operator
							information = "Enter second number"
							operator = ''
					elif flag_calc['second'] == False:
						if number_of_same_frame > 15:
							second += predicted_text
							calc_text += predicted_text
							number_of_same_frame = 0	



		if count_clear_frames == 30:
			first, second, operator, predicted_text, calc_text = '', '', '', '', ''
			flag_calc['first'], flag_calc['operator'], flag_calc['second'], flag_calc['clear'] = False, False, False, False
			information = "Enter first number"
			count_clear_frames = 0

		blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		cv2.putText(blackboard, "Predicted text - " + predicted_text, (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
		cv2.putText(blackboard, "Operator " + operator, (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 127))
		cv2.putText(blackboard, calc_text, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		cv2.putText(blackboard, information, (30, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255) )
		cv2.rectangle(image, (x_co,y_co), (x_co+width, y_co+height), (0,255,0), 2)
		res = np.hstack((image, blackboard))
		cv2.imshow("Calculator", res)
		cv2.imshow("thresh", thresh_image)
		if cv2.waitKey(1) == ord('q'):
			break


keras_predict(model, np.zeros((50, 50), dtype = np.uint8))
start_cal()
