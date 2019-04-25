import caffe
import cv2
import numpy as np
from scipy.spatial.distance import cosine

def extract_deep_feature(filename, net):
    img = cv2.imread(filename)
    
    if img is None:
        return None
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img - 127.5)/128
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    img = transformer.preprocess('data', img)
    img = np.transpose(img, [2, 0, 1]) 
    net.blobs['data'].reshape(1, 3, 112, 96)
    net.blobs['data'].data[...] = np.expand_dims(img, axis=0)
    res1 = net.forward()['fc5']
    net.blobs['data'].data[...] = np.expand_dims(np.flip(img, axis=0), axis=0)
    res2 = net.forward()['fc5']
    feature = np.concatenate([res1, res2], axis=1)
    return feature

def parse_list(pairs, test_folder_path):
    tests = []
    with open(pairs) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        l = line.split()
        if len(l) == 3:
            f1 = test_folder_path + l[0] + '/' + l[0] + '_%04i.jpg'%(int(l[1]))
            f2 = test_folder_path + l[0] + '/' + l[0] + '_%04i.jpg'%(int(l[2]))
            fold = np.ceil(i/600) - 1
            flag = 1
            tests.append([f1, f2, fold, flag])
        elif len(l) == 4:
            f1 = test_folder_path + l[0] + '/' + l[0] + '_%04i.jpg'%(int(l[1]))
            f2 = test_folder_path + l[2] + '/' + l[2] + '_%04i.jpg'%(int(l[3]))
            fold = np.ceil(i/600) - 1
            flag = 0
            tests.append([f1, f2, fold, flag])
    return tests
            
def get_threshold(scores, flags, thr_num):
    accuracies = np.zeros(2*thr_num)
    thresholds = np.arange(-thr_num, thr_num)/thr_num
    for i in range(len(accuracies)):
        accuracies[i] = get_accuracy(scores, flags, thresholds[i])
    best_threshold = np.mean(thresholds[accuracies == np.max(accuracies)])
    return best_threshold

def get_accuracy(scores, flags, threshold):
    accuracy = (np.sum(scores[flags == 1] > threshold) + np.sum(scores[flags != 1] <= threshold))/np.shape(scores)[0]
    return accuracy

def evaluation():
    gpu = True
    if gpu:
        gpu_id = 0
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
    else:
        caffe.set_mode_cpu()
    
    model = '../../train/code/sphereface_deploy.prototxt'
    weights = '../../train/result/sphereface_model.caffemodel'
    net = caffe.Net(model, weights, caffe.TEST)

    pairs = parse_list('./pairs.txt', '../data/result/lfw/')
    for i, pair in enumerate(pairs):
        print('Extracting deep features from the %dth face pair...'%(i))
        df1 = extract_deep_feature(pair[0], net)
        df2 = extract_deep_feature(pair[1], net)
        scores = 1 - cosine(df1,df2)
        pair.extend([df1, df2, scores])
    
    pairs = np.array(pairs)
    np.save('pairs.npy', pairs)
    accuracies = np.zeros(10)
    print('\n\n\nfold\tACC')
    print('----------------')
    for i in range(10):
        list_i = np.array([pair[2] for pair in pairs]).astype(np.uint16)
        scores = np.array([float(pair[-1]) for pair in pairs])
        flags = np.array([int(pair[3]) for pair in pairs])
        idx = list_i == i
        val_fold = pairs[np.logical_not(idx)]
        test_fold = pairs[idx]
        threshold = get_threshold(scores[np.logical_not(idx)], flags[np.logical_not(idx)], 10000)
        accuracies[i] = get_accuracy(scores[idx], flags[idx], threshold)*100
        print('%d\t%2.2f%%'%( i, accuracies[i]))
    print('----------------')
    print('AVE\t%2.2f%%'%(np.mean(accuracies)))

    

if __name__ == '__main__':
    evaluation()
