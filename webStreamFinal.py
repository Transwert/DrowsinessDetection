# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from flask import Response
from flask import Flask
from flask import render_template
import numpy as np
from scipy.spatial import distance as dist
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import threading
from threading import Thread
import datetime
import time
import cv2
import dlib
import playsound
import argparse

#Threshold constants required to define: 
eyeArThreshold = 0.25
eyeArConsecFrames = 45
pathOfPredictor = "shape_predictor_68_face_landmarks.dat"
WebcamIndex = 0
pathOfSound = "alarm.wav"

app = Flask(__name__)

@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')

def DrowsinessDetectorMain(pathOfPredictor, WebcamIndex, pathOfSound):

	#Threshold constants required to define: 
	eyeArThreshold = 0.3
	eyeArConsecFrames = 48

	def sound_alarm( pathOfSound ):
		playsound.playsound(pathOfSound)

	def eye_aspect_ratio( eyeLandmarks ):
		# compute the euclidean distances between the two sets of
		# vertical eye landmarks (x, y)-coordinates
		A = dist.euclidean(eyeLandmarks[1], eyeLandmarks[5])
		B = dist.euclidean(eyeLandmarks[2], eyeLandmarks[4])

		# compute the euclidean distance between the horizontal
		# eye landmark (x, y)-coordinates
		C = dist.euclidean(eyeLandmarks[0], eyeLandmarks[3])

		# compute the eye aspect ratio
		ear = (A + B) / (2.0 * C)

		# return the eye aspect ratio
		return ear

	def detectMain(frame):
		# initialize the frame counter as well as a boolean used to
		# indicate if the alarm is going off
		COUNTER = 0
		ALARM_ON = False
		ear = 0
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale frame
		rects = detector(gray, 0)

		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			
			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0

			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			frame = cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			frame = cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		return frame, ear

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(pathOfPredictor)

	# grab the indexes of the facial landmarks for the left and
	# right eye, respectively
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	# start the video stream thread
	print("[INFO] starting video stream thread...")
	print("[INFO] press 'q' to close the video stream")

	vs = VideoStream(src= WebcamIndex).start()
	time.sleep(1.0)

	COUNTER = 0

	# loop over frames from the video stream
	while True:

		ALARM_ON = False

		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)
		frame = vs.read()
		frame, ear = detectMain(frame)
		# print(ear)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < eyeArThreshold:
			COUNTER += 1

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= eyeArConsecFrames:

				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					if pathOfSound != "":
						# print(pathOfSound)
						t = Thread(target= sound_alarm, args=(pathOfSound,))
						t.deamon = True
						t.start()

					# draw an alarm on the frame
					frame = cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			ALARM_ON = False

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		frame = cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# show the frame
		# cv2.imshow("Frame", frame)
		ret, jpeg = cv2.imencode('.jpg', frame)

		if not ret:
			continue

		yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(jpeg) + b'\r\n\r\n')
		# key = cv2.waitKey(1) & 0xFF		 
		# # if the `q` key was pressed, break from the loop
		# if key == ord("q"):
		# 	print("[INFO] Shutting down the stream.")
		# 	break

		# do a bit of cleanup
	# cv2.destroyAllWindows()
	# vs.stop()

# def gen(camera):
#     while True:
#         #get camera frame
#         frame = camera.get_frame()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    # return Response(gen(VideoCamera()),
    #                 mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(DrowsinessDetectorMain(pathOfPredictor, WebcamIndex, pathOfSound),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True, help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True, help="ephemeral port number of the server (1024 to 65535)")
	args = vars(ap.parse_args())

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)
