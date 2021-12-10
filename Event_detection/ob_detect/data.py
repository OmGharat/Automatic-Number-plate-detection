import cv2
import datetime
import numpy as np

weight="D:\python1\ob_detect\\yolov3.weights"
cfg='D:\python1\ob_detect\\yolo.cfg'


net = cv2.dnn.readNet(weight,cfg)
classes = []
with open("coco.names", "r") as f:
        classes = f.read().splitlines()
        
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# Loading image
#img = cv2.imread("room_ser.jpg")
#img = cv2.resize(img, None, fx=0.4, fy=0.4)

# Enter file name for example "ak47.mp4" or press "Enter" to start webcam

def value():
    val = input("Enter file name or press enter to start webcam : \n")
    if val == "":
        val = 0
    return val
    
# image loading

    




# for video capture
cap = cv2.VideoCapture(value())

#val = cv2.VideoCapture()
start=datetime.datetime.now()
fps=0
total=0

font=cv2.FONT_HERSHEY_COMPLEX
frame_no = 0
while True:
    ret, img = cap.read()
    if ret==True:
        dt = str(datetime.datetime.now())
        print("for frame : " + str(frame_no) + "   timestamp is: ", str(cap.get(cv2.CAP_PROP_POS_MSEC)))
        frame = cv2.putText(img, dt,(5, 80),font, 1,(0, 0, 255),1, cv2.LINE_8)
        total=total+1
        end_time=datetime.datetime.now()
        time_diff=end_time - start
        
        if time_diff.seconds==0:
            fps=0.0
        else: 
            fps=(total/time_diff.seconds)
        
        fps_text="FPS : {:.2f}".format(fps)
        cv2.putText(img,fps_text,(5,30),cv2.FONT_HERSHEY_COMPLEX,1 ,(0,0,255),1)
    
        height, width, channels = img.shape
        # width = 512
        # height = 512

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
     
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

        # frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key==ord('q'):
            break
    else:
        print("video is end")
        break
    frame_no+=1
        

cap.release()
cv2.destroyAllWindows()