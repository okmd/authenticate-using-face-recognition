# import
import cv2 as cv
import time
import os
import math
from PIL import Image
import pickle
import numpy as np
import random
# some variable
directory = "datasets"
classifier = cv.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")
# generate date-time
def datetime():
    currenttime = []
    localtime = time.localtime(time.time())
    for val in localtime:
        currenttime.append(str(val))

    return "".join(currenttime[:6])

# generate datasets 20 faces
def generate():
    name =input("Enter Your Name: ")
    while not name:
        name =input("Enter Your Name: ")
    font = cv.FONT_HERSHEY_SIMPLEX
    offset = 50
    personId = int(random.random()*10**3)
    camera = cv.VideoCapture(0)
    if not os.path.exists(directory):
        os.mkdir(directory)
    if not os.path.exists(directory +"/"+name+"_"+str(personId)):
        os.mkdir(directory+ "/" +name+"_"+str(personId))
    i = 0
    while i < 50:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        ret, image = camera.read()
        if ret:
            grayImage = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
            faces = classifier.detectMultiScale(
            grayImage,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (100,100)
            )

            if len(faces) == 1:
                i +=1
                x, y, w, h = faces[0][0],faces[0][1],faces[0][2],faces[0][3]
                cv.imwrite(directory+ "/" +name+"_"+str(personId)+ "/" +str(i)+ ".jpg", grayImage[y:y+h,x:x+w])
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #radius = math.sqrt(abs(w**2 + h**2))/2
                #cv.circle(image, (int(x+w/2),int(y+h/2)),int(radius-radius*.15),(0,255,0),1)
                cv.putText(image,"[ "+name+" ]",(50,25),font,1,(0,255,0),2)
            else:
                for (x, y, w, h) in faces:
                    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    #radius = math.sqrt(abs(w**2 + h**2))/2
                    #cv.circle(image, (int(x+w/2),int(y+h/2)),int(radius-radius*.15),(0,255,0),1)
                msg = "{} face detected!, only one is allowed ".format(len(faces))
                cv.putText(image,msg,(15,25),font,1,(0,0,255),2)
            cv.waitKey(50)
            cv.imshow(name, image)
        else:
            print("!ret")
    camera.release()
    cv.destroyAllWindows()


def train():
    recognizer = cv.face.LBPHFaceRecognizer_create()

    person_paths = [os.path.join(directory, path) for path in os.listdir(directory)]
    images,labels = [],[]

    for img_path in person_paths:
        pid = int(os.path.split(img_path)[1].split("_")[-1])
        allImages = [os.path.join(img_path, path) for path in os.listdir(img_path)]
        for img in allImages:
             grayImage = Image.open(img).convert('L')
             image = np.array(grayImage, 'uint8')
             faces = classifier.detectMultiScale(image)
             for (x, y, w, h) in faces:
                 images.append(image[y: y + h, x: x + w])
                 labels.append(pid)
                 cv.imshow("Adding faces to traning set...", image)
                 cv.waitKey(20)
    cv.imshow("Traning Data",images[0])

    recognizer.train(images, np.array(labels))
    if not os.path.exists("trainer"):
        os.mkdir("trainer")
    recognizer.save('trainer/trained.yml')
    cv.destroyAllWindows()

def detect(src=0):
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trained.yml')
    camera = cv.VideoCapture(src)
    idName={}
    while True:
        if cv.waitKey(10) & 0xFF==ord('q'):
            break
        ret, image =camera.read()
        if ret:
            gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
            faces=classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv.CASCADE_SCALE_IMAGE)
            for(x,y,w,h) in faces:
                nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
                cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

                ldr = os.listdir("datasets")
                for nameAndId in ldr:
                    nai=nameAndId.split("_")
                    idName[int(nai[1])]=nai[0]

                if nbr_predicted in idName:
                    msg=idName[nbr_predicted]
                else:
                    msg="Matching"
                cv.putText(image,msg+"-"+str(int(conf))+"%", (x,y+h-20),cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1) #Draw the text
            cv.imshow("Detected Image",image)
            cv.waitKey(10)
        else:
            print("Unable to read Camera!")
            
    camera.release()
    cv.destroyAllWindows()

def faceInImage(imageurl):
    image = cv.imread(imageurl)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    # for (x,y,w,h) in faces:
    #     cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # cv.imshow("Face(s) Found", image)
    # cv.waitKey(0)
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trained.yml')
    idName={}
    for(x,y,w,h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        ldr = os.listdir("datasets")
        for nameAndId in ldr:
            nai=nameAndId.split("_")
            idName[int(nai[1])]=nai[0]

        if nbr_predicted in idName:
            msg=idName[nbr_predicted]
        else:
            msg="Not in record!"
        cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv.putText(image,msg+"-"+str(int(conf))+"%", (x-20,y+h+20),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,5),1) #Draw the text

    # for (x,y,w,h) in faces:
    #     cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.imshow("Face(s) Found", image)
    cv.waitKey(0)
#function call

#generate()

#train()
#detect()
#faceInImage("ab.jpg")
