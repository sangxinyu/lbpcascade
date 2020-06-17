import cv2
import sys
import os.path
import os

list=[]

def write_into_file(msg):
    with open("text.txt", "a") as f:
        data = f.write(msg)

def detect(filename, cascade_file="../lbpcascade_animeface.xml"):

    global xmin, ymin, xmax, ymax
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0

    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                    # detector options
                                    scaleFactor=1.1,
                                    minNeighbors=5,
                                    minSize=(24, 24))


    for (x, y, w, h) in faces:
        xmin = x
        ymin = y
        xmax = x+w
        ymax = y+h
    msg = filename[13:]+","+str(xmin)+","+str(ymin)+","+str(xmax)+","+str(ymax)+"\n"
    write_into_file(msg)

    if not msg.endswith("0,0,0,0\n"):
        list.append(0)

dirs = os.listdir("cartoon_test")
for file in dirs:
    detect("cartoon_test/"+file)
print(len(list))