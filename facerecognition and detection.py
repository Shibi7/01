import cv2
import face_recognition
import numpy as np
import os
import urllib.request
from datetime import datetime
url = 'http://192.168.12.134/cam-hi.jpg'

cv2.namedWindow("webcam",cv2.WINDOW_AUTOSIZE)

path = 'Registering'
images = []
classnames = []
mylist = os.listdir(path)

for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])

def findencodings(ima) :
    encodelist = []
    for im in ima:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(im)[0]
        encodelist.append(encode)
    return encodelist

faceencodings = findencodings(images)

def gettimeanddate(name):
    with open('FRD.csv','r+') as f:
        myDatalist = f.readlines()
        nameList = []
        for line in myDatalist:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# gettimeanddate('shibi')
# cap = cv2.VideoCapture(0)
while True:
    # sucess,img = cap.read()
    imgres = urllib.request.urlopen(url)
    imnp = np.array(bytearray(imgres.read()),dtype=np.uint8)
    img = cv2.imdecode(imnp, -1)
    # imgs = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fcl = face_recognition.face_locations(imgs)
    encod = face_recognition.face_encodings(imgs, fcl)

    for enc,f in zip(encod, fcl):
        matches = face_recognition.compare_faces(faceencodings, enc)
        facdis = face_recognition.face_distance(faceencodings, enc)

        mathindex = np.argmin(facdis)

        if matches[mathindex]:
            name = classnames[mathindex].upper()
            print(name)
            y1,x2,y2,x1 = f
            # y1, x2, y2, x1 = 4*y1,4*x2,4*y2,4*x 1
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img, name,(x1+6,y2-6), cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)
            gettimeanddate(name)
        else :
            y1, x2, y2, x1 = f
            # y1, x2, y2, x1 = 4 * y1, 4 * x2, 4 * y2, 4 * x1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
            gettimeanddate("unknown")
    cv2.imshow('webcam',img)
    cv2.waitKey(1)
