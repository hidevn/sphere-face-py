import numpy as np
import cv2
import os
import caffe
from scipy.spatial.distance import cosine

image_folder = './images'
output_folder = './features'
model = './train/code/sphereface_deploy.prototxt'
weights = './train/result/sphereface_model.caffemodel'
net = caffe.Net(model, weights, caffe.TEST)

def extract_deep_feature(filename, net):
    img = cv2.imread(filename)
    
    if img is None:
        return None
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img - 127.5)/128
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    img = transformer.preprocess('data', img)
    img = np.transpose(img, [2, 0, 1]) 
    data = np.concatenate([np.expand_dims(img, axis=0), np.expand_dims(np.flip(img, axis=0), axis=0)], axis=0)
    net.blobs['data'].reshape(2, 3, 112, 96)
    net.blobs['data'].data[...] = data
    res = net.forward()['fc5']
    feature = np.concatenate([res[0], res[1]])
    return feature

def save_feature_vectors():
    list_images = os.listdir(image_folder)
    os.makedirs(output_folder, exist_ok=True)
    for image_name in list_images:
        feature_vector = extract_deep_feature(os.path.join(image_folder, image_name), net)
        np.savetxt(os.path.join(output_folder, image_name.split('.')[0]), feature_vector)

def detect(feature):
    list_features = os.listdir(output_folder)
    scores = []
    for image_name in list_features:
        feature2 = np.loadtxt(os.path.join(output_folder, image_name))
        score = 1 - cosine(feature,feature2)
        scores.append(score)
    scores = np.array(scores)
    return list_features[np.argmax(scores)]

def detect_from_img(img_path):
    img_feature = extract_deep_feature(img_path, net)
    return detect(img_feature)

if __name__ == '__main__':
    #save_feature_vectors()
    print(detect_from_img('./Aaron_Peirsol_0003.jpg'))
    #img_feature = extract_deep_feature('./Aaron_Peirsol_0003.jpg', net)
    