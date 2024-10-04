import requests  # for send https request and interact with server
import cv2       # for Computer Vision related task
import numpy as np  # for mathematical manipulation
import imutils     # for image manipulation

url='http://192.168.101.2:8080/shot.jpg'

while True:
    img_resp=requests.get(url)
    img_arr=np.array(bytearray(img_resp.content), dtype=np.uint8)
    img=cv2.imdecode(img_arr, -1)
    img=imutils.resize(img, width=1000, height=1000)
    cv2.imshow('android_cam', img)

    if cv2.waitKey(1)==27:
        break

cv2.destroyAllWindows()