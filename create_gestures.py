import cv2
import numpy as np
import pickle, os, sqlite3

image_x_coordinate, image_y_coordinate = 50, 50

def get_hand_histogram_valueogram_value():
	with open("hist", "rb") as f:
		histogram_value = pickle.load(f)
	return histogram_value

def initialize_creation_of_database():
	# create the folder and database if not exist
	if not os.path.exists("gestures"):
		os.mkdir("gestures")
	if not os.path.exists("gesture_db.db"):
		conn = sqlite3.connect("gesture_db.db")
		create_table_command_query = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
		conn.execute(create_table_command_query)
		conn.commit()

def create_folder_for_gestures(folder_name_for_gestures):
	if not os.path.exists(folder_name_for_gestures):
		os.mkdir(folder_name_for_gestures)

def create_images_can_be_empty(folder_name_for_gestures, n_images):
	create_folder_for_gestures("gestures/"+folder_name_for_gestures)
	black_images_in_gestures = np.zeros(shape=(image_x_coordinate, image_y_coordinate, 1), dtype=np.uint8)
	for i in range(n_images):
		cv2.imwrite("gestures/"+folder_name_for_gestures+"/"+str(i+1)+".jpg", black_images_in_gestures)

def store_in_database(g_id, g_name):
	conn = sqlite3.connect("gesture_db.db")
	command_query = "INSERT INTO gesture (g_id, g_name) VALUES (%s, \'%s\')" % (g_id, g_name)
	try:
		conn.execute(command_query)
	except sqlite3.IntegrityError:
		choice_option_y_n = input("g_id already exists. Want to change the record? (y/n): ")
		if choice_option_y_n.lower() == 'y':
			command_query = "UPDATE gesture SET g_name = \'%s\' WHERE g_id = %s" % (g_name, g_id)
			conn.execute(command_query)
		else:
			print("Doing nothing... !! SAVE COMPUTATIONS")
			return
	conn.commit()
	
def store_images(g_id):
	total_pics = 1200
	if g_id == str(0):
		create_images_can_be_empty("0", total_pics)
		return
	histogram_value = get_hand_histogram_valueogram_value()

	cam = cv2.VideoCapture(0)
	x, y, w, h = 300, 100, 300, 300

	create_folder_for_gestures("gestures/"+str(g_id))
	picture_number = 0
	flag_start_capturing = False
	frames = 0
	
	while True:
		ret,image = cam.read()
		image = cv2.flip(image, 1)
		t=y+h
		q=x+w
		crop_image = image[y:t, x:q]
		imageHSV_B2W = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		dst_image = cv2.calcBackProject([imageHSV_B2W], [0, 1], histogram_value, [0, 180, 0, 256], 1)
		disc_image = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
		cv2.filter2D(dst_image,-1,disc_image,dst_image)
		blured_image = cv2.GaussianBlur(dst_image, (11,11), 0)
		blured_image = cv2.medianBlur(blured_image, 15)
		threshold_image = cv2.threshold(blured_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		threshold_image = cv2.merge((threshold_image,threshold_image,threshold_image))
		threshold_image = cv2.cvtColor(threshold_image, cv2.COLOR_BGR2GRAY)
		threshold_image = threshold_image[y:y+h, x:x+w]
		contour_image = cv2.findContours(threshold_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

		if len(contour_image) > 0:
			contour = max(contour_image, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000 and frames > 50:
				x_co, y_co, width_1, height = cv2.boundingRect(contour)
				picture_number += 1
				saveed_image = threshold_image[y_co:y_co+height, x_co:x_co+width_1]
				if width_1 > height:
					saveed_image = cv2.copyMakeBorder(saveed_image, int((width_1-height)/2) , int((width_1-height)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
				elif height > width_1:
					saveed_image = cv2.copyMakeBorder(saveed_image, 0, 0, int((height-width_1)/2) , int((height-width_1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
				saveed_image = cv2.resize(saveed_image, (image_x_coordinate, image_y_coordinate))
				cv2.putText(image, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
				cv2.imwrite("gestures/"+str(g_id)+"/"+str(picture_number)+".jpg", saveed_image)

		cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
		cv2.putText(image, str(picture_number), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
		cv2.imshow("Capturing gesture !! Do not disturb while capturing",image)
		cv2.imshow("threshold image", threshold_image)
		keypress = cv2.waitKey(1)
		if keypress == ord('c'):
			if flag_start_capturing == False:
				flag_start_capturing = True
			else:
				flag_start_capturing = False
				frames = 0
		if flag_start_capturing == True:
			frames += 1
		if picture_number == total_pics:
			break

initialize_creation_of_database()
g_id = input("Enter gesture no.: ")
g_name = input("Enter gesture name/text: ")
store_in_database(g_id, g_name)
store_images(g_id)