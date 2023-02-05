

import os
import cv2
import matplotlib.pyplot as plt

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    
    txt = open(dataPath, "r") #read mode
    n = 0
    while n < 2:
        info = txt.readline()
        name = info.split(" ")[0]
        faces = info.split(" ")[1]
        boxes = []
        path = "data/detect/" + name
        img = cv2.imread(path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        for i in range(int(faces)):
            boxes.append(txt.readline())
            x, y, w, h = boxes[i].split(" ")
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            img_crop = img_gray[y : y + h, x : x + w]
            img_resize = cv2.resize(img_crop, (19, 19), interpolation = cv2.INTER_NEAREST)
            start_point = (x, y)
            end_point = (x + w, y + h)
            # color
            green = (0, 255, 0) #face
            red = (0, 0, 255) #non-face
            # thickness
            if (clf.classify(img_resize) == 1):
                img = cv2.rectangle(img, start_point, end_point, green, 2)
            else:
                img = cv2.rectangle(img, start_point, end_point, red, 2)
            
        cv2.imshow(name, img)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.show()
        n += 1
            
    # raise NotImplementedError("To be implemented")
    # End your code (Part 4)
