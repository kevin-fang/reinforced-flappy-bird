import cv2

def bw_shrink(img):
	image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
	resized = cv2.resize(image, (0, 0), fx=0.2 , fy=0.2)
	return resized