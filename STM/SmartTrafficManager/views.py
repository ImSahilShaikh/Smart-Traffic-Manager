from django.shortcuts import render
from django.contrib import messages
from django.http import HttpResponse

# DATABASE
from .models import *
#

import os

# Speed Check
import cv2
import dlib
import time
import threading
import math
# Speed Check

# OCR
import pytesseract
import imutils
# OCR

#DETECTOR LICENSE PLATE

import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import SmartTrafficManager.core.utils as utils
from SmartTrafficManager.core.yolov4 import filter_boxes
from SmartTrafficManager.core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#DETECTOR LICENSE PLATE

# DETECTOR FLAGS



# DETECTOR FLAGS

#SEND EMAIL
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 

import datetime



#SEND EMAIL

# Create your views here.


def login(request):
    return render(request,"SmartTrafficManager/login.html")


def home(request):
    # x = datetime.datetime.now()
    # myobject = violators.objects.create(license_plate="lasjdsad",owner_name="Sahil",owner_email="sahil@gmail.com",date_time=x,email_status=1)

    return render(request, "SmartTrafficManager/index.html")


def SpeedLimitDetection(request):
    print("loading haar and video")
    carCascade = cv2.CascadeClassifier(
        './input_data/cascade_classifiers/myhaar.xml')
    video = cv2.VideoCapture('./input_data/input_videos/indian_vid_c1.mp4')
    print("after loading haar and video")

    WIDTH = 1280
    HEIGHT = 720

    carLocation1 = {}
    carLocation2 = {}

    def estimateSpeed(location1, location2):
        d_pixels = math.sqrt(math.pow(
            location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
        # ppm = location2[2] / carWidht
        ppm = 5.5
        d_meters = d_pixels / ppm
        #print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
        fps = 18
        speed = d_meters * fps * 3.6
        return speed

    def trackMultipleObjects():
        print("track multiple objects")
        rectangleColor = (0, 255, 0)
        frameCounter = 0
        currentCarID = 0
        fps = 0

        carTracker = {}
        carNumbers = {}

        speed = [None] * 1000

        storeKeys = {}

        counter = 0
        # Write output to video file
        out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 10, (WIDTH, HEIGHT))

        while True:
            print("while loop of track multiple objects")
            start_time = time.time()
            rc, image = video.read()

            if type(image) == type(None):
                print("breaking")
                break

            image = cv2.resize(image, (WIDTH, HEIGHT))
            resultImage = image.copy()

            frameCounter = frameCounter + 1

            carIDtoDelete = []

            for carID in carTracker.keys():
                trackingQuality = carTracker[carID].update(image)

                if trackingQuality < 7:
                    carIDtoDelete.append(carID)

            for carID in carIDtoDelete:
                print('Removing carID ' + str(carID) +
                      ' from list of trackers.')
                print('Removing carID ' + str(carID) + ' previous location.')
                print('Removing carID ' + str(carID) + ' current location.')
                carTracker.pop(carID, None)
                carLocation1.pop(carID, None)
                carLocation2.pop(carID, None)

            if not (frameCounter % 10):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

                for (_x, _y, _w, _h) in cars:
                    x = int(_x)
                    y = int(_y)
                    w = int(_w)
                    h = int(_h)

                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h

                    matchCarID = None

                    for carID in carTracker.keys():
                        trackedPosition = carTracker[carID].get_position()

                        t_x = int(trackedPosition.left())
                        t_y = int(trackedPosition.top())
                        t_w = int(trackedPosition.width())
                        t_h = int(trackedPosition.height())

                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                            matchCarID = carID

                    if matchCarID is None:
                        print('Creating new tracker ' + str(currentCarID))

                        tracker = dlib.correlation_tracker()
                        tracker.start_track(
                            image, dlib.rectangle(x, y, x + w, y + h))

                        carTracker[currentCarID] = tracker
                        carLocation1[currentCarID] = [x, y, w, h]

                        currentCarID = currentCarID + 1

            # cv2.line(resultImage,(0,480),(1280,480),(255,0,0),5)

            for carID in carTracker.keys():
                trackedPosition = carTracker[carID].get_position()

                t_x = int(trackedPosition.left())
                t_y = int(trackedPosition.top())
                t_w = int(trackedPosition.width())
                t_h = int(trackedPosition.height())

                cv2.rectangle(resultImage, (t_x, t_y),
                              (t_x + t_w, t_y + t_h), rectangleColor, 4)

                # speed estimation
                carLocation2[carID] = [t_x, t_y, t_w, t_h]

            end_time = time.time()

            if not (end_time == start_time):
                fps = 1.0/(end_time - start_time)

            #cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            for i in carLocation1.keys():

                if frameCounter % 1 == 0:
                    [x1, y1, w1, h1] = carLocation1[i]
                    [x2, y2, w2, h2] = carLocation2[i]

                    # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
                    carLocation1[i] = [x2, y2, w2, h2]

                    # print 'new previous location: ' + str(carLocation1[i])
                    if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                        if (speed[i] == None or speed[i] == 0) and y1 >= 375 and y1 <= 385:
                            speed[i] = estimateSpeed(
                                [x1, y1, w1, h1], [x2, y2, w2, h2])

                        # if y1 > 275 and y1 < 285:
                        if speed[i] != None and y1 >= 180:
                            flag = 1

                            cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(
                                y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                            if not i in storeKeys.keys():
                                if int(speed[i]) > 70 and flag != 0:

                                    #(t_x, t_y), (t_x + t_w, t_y + t_h)
                                    SpeedCroppedImage = image[h1 +
                                                              280:y1+h1, w1+200:x1+w1]
                                    cv2.imwrite(
                                        './data/Cropped/myCropped_'+str(counter)+'.jpg', SpeedCroppedImage)

                                    #cv2.imwrite('overspeeders/overspeeding '+str(counter)+'.jpg',resultImage)

                                    # cv2.imwrite('overspeeders/crop'+str(counter)+'.jpg',resultImagecropped)
                                    storeKeys[i] = speed[i]
                                    #print("Store Keys"+str(storeKeys))

                                    counter = counter + 1

            cv2.imshow('result', resultImage)

            if cv2.waitKey(33) == 27:
                break

        cv2.destroyAllWindows()
    trackMultipleObjects()

    messages.success(request, "Images Trained Successfully.")
    return render(request, "SmartTrafficManager/index.html")

def licenseplate_detector(request):

    #flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
    # flags.DEFINE_string('weights', './checkpoints/yolov4-416','path to weights file')
    # flags.DEFINE_integer('size', 416, 'resize images to')
    #flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
    flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
    # flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
    flags.DEFINE_string('output', './detections/', 'path to output folder')
    flags.DEFINE_float('iou', 0.45, 'iou threshold')
    flags.DEFINE_float('score', 0.50, 'score threshold')
    # flags.DEFINE_boolean('count', False, 'count objects within images')
    flags.DEFINE_boolean('dont_show', False, 'dont show image output')
    flags.DEFINE_boolean('info', False, 'print info on detections')
    # flags.DEFINE_boolean('crop', False, 'crop detections from images')
    # flags.DEFINE_boolean('ocr', False, 'perform generic OCR on detection regions')
    # flags.DEFINE_boolean('plate', False, 'perform license plate recognition')

    def crop_objects(img, data, crop_path ,allowed_classes):
        boxes, scores, classes, num_objects = data
        class_names = read_class_names(cfg.YOLO.CLASSES)
        #create dictionary to hold count of objects for image name
        counts = dict()
        for i in range(num_objects):
            # get count of class for part of image name
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name in allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
                # get box coords
                xmin, ymin, xmax, ymax = boxes[i]
                # crop detection from image (take an additional 5 pixels around all edges)
                cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
                # construct image name and join it to path for saving crop properly
                img_name = class_name + '_' + str(counts[class_name]) + '.jpg'
                img_path = crop_path+"_"+img_name
                # save image
                cv2.imwrite(img_path, cropped_img)
                print("License Plate cropped Successfully")
            else:
                continue
        
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 416

    # load model
    # if FLAGS.framework == 'tflite':
    #         interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
    # else:
    
    saved_model_loaded = tf.saved_model.load("./input_data/yolov4-licenseplate-model/custom-416", tags=[tag_constants.SERVING])
    print("model loaded")
    # loop through images in list and run Yolov4 model on each
    # print("images = ",images)
    # count, image_path = images,1
    # print("count=",count)
    # print("image path = ",image_path)
    # for count, image_path in enumerate(images, 1):
    #     print(image_path)


    #images = "./data/images/car.jpg"

    images_list = os.listdir("./data/Cropped/")
    for i in images_list:
        print(i)
    for i in images_list:
        
        images = "./data/Cropped/"+i
        original_image = cv2.imread(images)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.
        
        # get image name by using split method
        image_name = images.split('/')[-1]
        image_name = image_name.split('.')[0]

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        # if FLAGS.framework == 'tflite':
        #     interpreter.allocate_tensors()
        #     input_details = interpreter.get_input_details()
        #     output_details = interpreter.get_output_details()
        #     interpreter.set_tensor(input_details[0]['index'], images_data)
        #     interpreter.invoke()
        #     pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        #     # if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
        #     #     boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        #     # else:
        #     boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        # else:
        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # run non max suppression on detections
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.50
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = original_image.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
        
        # hold all detection data in one variable
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to allow detections for only people)
        #allowed_classes = ['person']

        # if crop flag is enabled, crop each detection and save it as new image
        # if FLAGS.crop:
        crop_path = os.path.join(os.getcwd(),'data', 'LicensePlates',image_name)
        print("Cropping path: ",crop_path)
        crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes)


        # if ocr flag is enabled, perform general text extraction using Tesseract OCR on object detection bounding box
        # if FLAGS.ocr:
        #     ocr(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox)

        # if count flag is enabled, perform counting of objects
        # if FLAGS.count:
        #     # count objects found
        #     counted_classes = count_objects(pred_bbox, by_class = False, allowed_classes=allowed_classes)
        #     # loop through dict and print
        #     for key, value in counted_classes.items():
        #         print("Number of {}s: {}".format(key, value))
        #     image = utils.draw_bbox(original_image, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes, read_plate = FLAGS.plate)
        # else:
        image = utils.draw_bbox(original_image, pred_bbox, False, allowed_classes=allowed_classes, read_plate = False)
        
        image = Image.fromarray(image.astype(np.uint8))
        #if not FLAGS.dont_show:
        #    image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        dirpath = os.getcwd()
        print(dirpath)
        cv2.imwrite(dirpath+'/detections' + image_name + '.jpg', image)
        print("License plate detection image written Successfully")


    return render(request,"SmartTrafficManager/index.html")


def ocr_licenseplate(request):

    images = os.listdir("./data/LicensePlates")

    #file = open("numbers.txt","a")

    violators_license_numbers = set()

    for i in images:
        print(i)

    for i in images:

        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
        imagepath = "./data/LicensePlates/"+i

        image = cv2.imread(imagepath)

        image = imutils.resize(image, width=500)

        #cv2.imshow("Original image", image)
        #cv2.waitKey(0)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Gray scale image", gray)
        #cv2.waitKey(0)

        gray = cv2.bilateralFilter(gray, 11, 17, 17)
    #cv2.imshow("Smooth image", gray)
    #cv2.waitKey(0)

        edged = cv2.Canny(gray, 170, 200)
    #cv2.imshow("Canny image", edged)
    #cv2.waitKey(0)

        cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        image1 = image.copy()
        cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
        #cv2.imshow("Canny after Contouring", image1)
        #cv2.waitKey(0)

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:200]
        NumberPlateCount = None

        image2 = image.copy()
        cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
        #cv2.imshow("Top 200 Contours", image2)
    #cv2.waitKey(0)

        count = 0
        name = 1

        for i in cnts:
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*perimeter, True)
            if(len(approx) == 4):
                NumberPlateCount = approx
                x, y, w, h = cv2.boundingRect(i)
                crp_img = image[y:y+h, x:h+w]

                cv2.imwrite(str(name) + '.png', crp_img)
                name += 1

                break
        cv2.drawContours(image, [NumberPlateCount], -1, (0, 255, 0), 3)

        text = pytesseract.image_to_string(image, lang='eng')
        text = ''.join(e for e in text if e.isalnum())
        #print("Number:" + text)
        if not len(text) == 0:
            violators_license_numbers.add(text)
        #file.write(text)
        #file.write("\n")
        #cv2.imshow("Final Image", image)
        #cv2.waitKey(0)
        
    for i in violators_license_numbers:
            
        rto = RTO_Details.objects.filter(license_plate=i).values_list('license_plate','owner_name','owner_email')
        
        for j in rto:
            cur_date_time = datetime.datetime.now()    
            myobject = violators.objects.create(license_plate=j[0],owner_name=j[1],owner_email=j[2],date_time=cur_date_time,email_status=0)    

    
    violators_render = violators.objects.all

    context = {'violators_render':violators_render}
    return render(request, "SmartTrafficManager/index.html",context)

def sendEmail(request):
    fromaddr = "projecttms4@gmail.com"
    toaddr = "niranjanp9192@gmail.com"

    # MIMEMultipart 
    msg = MIMEMultipart() 

    # senders email address 
    msg['From'] = fromaddr 

    # receivers email address 
    msg['To'] = toaddr 

    # the subject of mail
    msg['Subject'] = "Mailer test with attachment"

    # the body of the mail 
    body = "Hello there, this is test email"

    # attaching the body with the msg 
    msg.attach(MIMEText(body, 'plain')) 

    #------------------Uncomment following code if you want to send attachment

    # open the file to be sent
    # rb is a flag for readonly 
    # filename = "demo.jpg"
    # attachment = open("./demo.jpg", "rb") 

    # # MIMEBase
    # attac= MIMEBase('application', 'octet-stream') 

    # # To change the payload into encoded form 
    # attac.set_payload((attachment).read()) 

    # # encode into base64 
    # encoders.encode_base64(attac) 

    # attac.add_header('Content-Disposition', "attachment; filename= %s" % filename) 

    # # attach the instance 'p' to instance 'msg' 
    # msg.attach(attac) 

    #----------------------------End of attachment---------------------------------------

    # creates SMTP session 
    email = smtplib.SMTP('smtp.gmail.com', 587) 

    # TLS for security 
    email.starttls() 

    # authentication 
    email.login(fromaddr, "tms@1234") 

    # Converts the Multipart msg into a string 
    message = msg.as_string() 

    # sending the mail 
    email.sendmail(fromaddr, toaddr, message) 

    print("Mail Sent")

    # terminating the session 
    email.quit()
    return render(request,"SmartTrafficManager/index.html")