import os
import cv2

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    
    dataset = []
    path = dataPath + "/face"
    for filename in os.listdir(path):
        img = cv2.imread(path + "/" + filename, cv2.IMREAD_GRAYSCALE)
        dataset.append((img, 1))
    path = dataPath + "/non-face"
    for filename in os.listdir(path):
        img = cv2.imread(path + "/" + filename, cv2.IMREAD_GRAYSCALE)
        dataset.append((img, 0))
    
    # raise NotImplementedError("To be implemented")
    # End your code (Part 1)
    return dataset
