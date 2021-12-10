import cv2
import requests
import clx.xms
import numpy as np 
import argparse
from pygame import mixer
import smtplib,ssl
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--play_video', help="Tue/False", default=False)
parser.add_argument('--image', help="Tue/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default=False)
parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()


weight="D:\python1\object\\yolov3.weights"
cfg='D:\python1\object\\yolo.cfg'
#Load yolo
mixer.init()
sound = mixer.Sound('alert.mpeg')

  
# establishing connection

smtp_server = "smtp.gmail.com"
port = 587  # For starttls
sender_email = "000testdemo@gmail.com"
password = "Ritesh@1234"
receiver_email=["mishra3618@gmail.com","rm309295@gmail.com"]
msg = MIMEMultipart()

msg['From'] = "000testdemo@gmail"
msg['To'] = ", ".join(receiver_email)
msg['Subject'] = "alert"




# Create a secure SSL context
context = ssl.create_default_context()

def load_yolo():
    net = cv2.dnn.readNet(weight,cfg)
    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

def load_image(img_path):
    # image loading
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels

def start_webcam():
    cap = cv2.VideoCapture(0)

    return cap


def display_blob(blob):
    '''
        Three images each for RED, GREEN, BLUE channel
    '''
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)

def detect_objects(img, net, outputLayers):			
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
                
    return boxes, confs, class_ids
            

    

def image_detect(img_path): 
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, colors, class_ids, classes, image)
    while True:
        key = cv2.waitKey(1)
        if key == 10:
            break
   

def webcam_detect():
    model, classes, colors, output_layers = load_yolo()
    cap = start_webcam()
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()


def start_video(video_path):
    
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    start=datetime.datetime.now()
    fps=0
    total=0
    
    font=cv2.FONT_HERSHEY_COMPLEX
    frame_no = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret==True:
            dt = str(datetime.datetime.now())
            print("for frame : " + str(frame_no) + "   timestamp is: ", str(cap.get(cv2.CAP_PROP_POS_MSEC)))
            frame = cv2.putText(frame, dt,(5, 80),font, 1,(0, 0, 255),1, cv2.LINE_8)
            total=total+1
            end_time=datetime.datetime.now()
            time_diff=end_time - start
            
            if time_diff.seconds==0:
                fps=0.0
            else: 
                fps=(total/time_diff.seconds)
            
            fps_text="FPS : {:.2f}".format(fps)
            cv2.putText(frame,fps_text,(5,30),cv2.FONT_HERSHEY_COMPLEX,1 ,(0,0,255),1)
            height, width, channels = frame.shape
            blob, outputs = detect_objects(frame, model, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
            draw_labels(boxes, confs, colors, class_ids, classes, frame)

            key = cv2.waitKey(1)
            if key==ord('q'):
                break
        else:
            print("video is end")
            break
        frame_no+=1
       
            
       
        
    cap.release()
    cv2.destroyAllWindows()
    print("video end")

def draw_labels(boxes, confs, colors, class_ids, classes, img): 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
   
        
        
        
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            date = str(datetime.datetime.now())
            print(label,"detected in frame")
            
            body="""hello this is an"""+str(label)+""" alert message a """+str(label)+""" was detected """+str(date)+""" that sit """
            msg.attach(MIMEText(body))
            text = msg.as_string()
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1.5, color, 2)
            try:
                server = smtplib.SMTP(smtp_server,port)
                server.ehlo() # Can be omitted
                server.starttls(context=context) # Secure the connection
                server.ehlo() # Can be omitted
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, text)
                # TODO: Send email here
            except Exception as e:
                # Print any error messages to stdout
                print(e)
            finally:
                server.quit() 
            sound.play() 
    img=cv2.resize(img, (1000,800))
    cv2.imshow("Image", img)
    
    
    
    


if __name__ == '__main__':
    webcam = args.webcam
    video_play = args.play_video
    image = args.image
    if webcam:
        if args.verbose:
            print('---- Starting Web Cam object detection ----')
        webcam_detect()
    if video_play:
        video_path = args.video_path
        if args.verbose:
            print('Opening '+video_path+" .... ")
        start_video(video_path)
    if image:
        image_path = args.image_path
        if args.verbose:
            print("Opening "+image_path+" .... ")
        image_detect(image_path)
    

    cv2.destroyAllWindows()