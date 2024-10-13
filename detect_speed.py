#Plotting Tracks Over Time

#Import All the Required Libraries
import cv2
from ultralytics import YOLO,solutions
from collections import defaultdict
import numpy as np

#Load the YOLO Model
model = YOLO("yolo11n.pt")
names=model.model.names
#Create a Video Capture Object
cap = cv2.VideoCapture("Resources/Videos/video7.mp4")
w,h,fps=(int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS))
video_writer= cv2.VideoWriter("ew video after detect speed.avi",cv2.VideoWriter.fourcc(*"mp4v"),fps,(w,h)) 
line_pts=[(0,360),(1200,360)]

speed_obj=solutions.SpeedEstimator(reg_pts=line_pts,names=names,view_img=True)
#Loop through the Video Frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("frame empty or procces")
        #Run YOLO11 tracking on the frame
    results = model.track(frame, persist=True)
    frame=speed_obj.estimate_speed(frame,results)
    video_writer.write(frame)
    #cv2.imshow("YOLO11 Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        break
    
cap.release()
#video_writer.release()
cv2.destroyAllWindows()
