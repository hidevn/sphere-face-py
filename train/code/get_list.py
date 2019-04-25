import os

folder = '/media/hide/la/bump/sphereface-master/preprocess/data/small'
sub_folder = os.listdir(folder)

# create the list for training
with open('../data/CASIA-WebFace-112X96.txt', 'w') as f:
    for i, fol in enumerate(sub_folder):
        print('Collecting the %dth folder (total %d) ...' % (i, len(sub_folder)))
        file_names = os.listdir(os.path.join(folder, fol))
        file_names = [os.path.abspath(os.path.join(folder, fol, name)) for name in file_names]
        for name in file_names:
            f.write('%s %d\n'%(name, i))
        