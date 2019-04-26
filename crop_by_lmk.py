import numpy as np
from cv2 import warpAffine
import cv2
from skimage import transform as trans
import os

def face_align_demo():
    lmk_path = './celebrity_lmk/'
    data_path = './'
    aligned_path = './aligned/'
    img_size = np.array([112, 96])
    coord5point = np.array([[30.2946, 51.6963],
               [65.5318, 51.5014],
               [48.0252, 71.7366],
               [33.5493, 92.3655],
               [62.7299, 92.2041]])
    with open(lmk_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        row_data = line.split()
        img_path = os.path.join(data_path, row_data[0])
        img_label = row_data[1]
        facial5point = np.array([[float(row_data[2]), float(row_data[3])]
                        [float(row_data[4]), float(row_data[5])],
                        [float(row_data[6]), float(row_data[7])],
                        [float(row_data[8]), float(row_data[9])],
                        [float(row_data[10]), float(row_data[11])]])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tform = trans.SimilarityTransform()                                                                                                                                                  
        tform.estimate(facial5point, coord5point)
        M = tform.params[0:2,:]
        img_aligned = warpAffine(img,M,(img_size[1],img_size[0]), borderValue = 0.0)
        img_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.join(aligned_path, row_data[0]), exist_ok=True)
        cv2.imwrite(os.path.join(aligned_path, row_data[0]), img_aligned)


if __name__ == '__main__':
    face_align_demo()