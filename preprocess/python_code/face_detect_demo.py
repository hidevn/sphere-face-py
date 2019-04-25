import os
import caffe
import numpy as np
from MTCNNv1.detect_face import detect_face
import cv2
def face_detect_demo():
    #train_list = read_folder('../data/CASIA-WebFace', 'CASIA')
    #test_list = read_folder('../data/lfw', 'lfw')
    
    # mtcnn settings
    min_size = 20
    factor = 0.85
    threshold = [0.6, 0.7, 0.9]

    gpu = False
    if gpu:
        gpu_id = 0
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
    else:
        caffe.set_mode_cpu()
    
    model_path = './MTCNNv1/model/'
    PNet = caffe.Net(model_path+'det1.prototxt', model_path+'det1.caffemodel', caffe.TEST)
    RNet = caffe.Net(model_path+'det2.prototxt', model_path+'det2.caffemodel', caffe.TEST)
    ONet = caffe.Net(model_path+'det3.prototxt', model_path+'det3.caffemodel', caffe.TEST)

    #data_list = train_list + test_list
    data_list = read_folder('../data/lfw', 'lfw')
    for i, data in enumerate(data_list):
        print('Detecting the', i, 'image')
        # load image
        img = cv2.imread(data[1] + '/' + data[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[2] == 1:
            img = np.matlib.tile(img, (1, 1, 3))
        # detection
        bboxes, landmarks = detect_face(img, min_size, PNet, RNet, ONet, threshold, factor)
        if np.shape(bboxes)[0] > 1:
            # pick the face closed to the center
            center = [np.shape(img)[0]/2, np.shape(img)[1]/2]
            distance = (np.mean(bboxes[:, [1, 3]], axis=1) - center[1])**2+\
                       (np.mean(bboxes[:, [0, 2]], axis=1) - center[0])**2
            ix = np.argmin(distance)
            data_list[i].append(bboxes[ix, :4].reshape(4,))
            data_list[i].append(landmarks[ix, :].reshape(5,2))
        elif np.shape(bboxes)[0] == 1:
            data_list[i].append(bboxes[0, :4].reshape(4,))
            data_list[i].append(landmarks.reshape(5,2))
        else:
            data_list[i].append([])
            data_list[i].append([])
    np.save('data_list', np.array(data_list))

def read_folder(folder, name):
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    list_img = []
    for sf in subfolders:
        imgs = os.listdir(sf)
        for img in imgs:
            list_img.append([img, sf, name])
    return list_img

if __name__ == '__main__':
    face_detect_demo()