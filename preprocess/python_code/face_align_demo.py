import numpy as np
from cv2 import warpAffine
import cv2
from skimage import transform as trans
import os

def face_align_demo():
    img_size = np.array([112, 96])
    coord5point = np.array([[30.2946, 51.6963],
               [65.5318, 51.5014],
               [48.0252, 71.7366],
               [33.5493, 92.3655],
               [62.7299, 92.2041]])
    data_list = np.load('./data_list.npy')
    result_dir = '../result'
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
        
    # face alignment
    for i, data in enumerate(data_list):
        print('aligning the %dth image'%(i))
        if data_list[-1] == []:
            continue
        print(result_dir + '/' + data[2])
        if not os.path.isdir(result_dir + '/' + data[2]):
            os.makedirs(result_dir + '/' + data[2])
        
        img = cv2.imread(data[1] + '/' + data[0])
        print(data[1] + '/' + data[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        facial5point = data[-1]
        
        tform = trans.SimilarityTransform()                                                                                                                                                  
        tform.estimate(facial5point, coord5point)
        M = tform.params[0:2,:]
        img_aligned = warpAffine(img,M,(img_size[1],img_size[0]), borderValue = 0.0)

        img_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_RGB2BGR)
        write_to_subfolder(result_dir + '/' + data[2], data[1].split('/')[-1], data[0], img_aligned)

def write_to_subfolder(folder, subfolder, name, img):
    if not os.path.isdir(folder + '/' + subfolder):
        os.makedirs(folder + '/' + subfolder)
    cv2.imwrite(folder + '/' + subfolder + '/' + name, img)

if __name__ == '__main__':
    face_align_demo()