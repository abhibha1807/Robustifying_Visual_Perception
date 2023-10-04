import os
import numpy as np
import torch
import csv
import random
import cv2
import imageio.v2 as imageio
from torch.utils.data import Dataset

class gtsrbLoader(Dataset):

  def __init__(self, exp, split='train', is_transform=False, img_size=None, augmentations=None, prototype_sampling_rate=0.005):
    super().__init__()

    if split == 'train':
        self.proto_rate = prototype_sampling_rate
    else:
        self.proto_rate = 0.0
        
    self.inputs = []
    self.targets = []
    self.class_names = []
    self.split = split
    self.img_size = img_size
    self.is_transform = is_transform
    self.augmentations = augmentations
    self.mean = np.array([125.00, 125.00, 125.00]) # average intensity
    self.n_classes = 43
    self.tr_class = torch.LongTensor([i for i in (0,self.n_classes)]) #number of classes
    self.te_class = torch.LongTensor([i for i in (0,self.n_classes)])
    self.org_dataset_path = '/GTSRB/org_dataset/'
    self.aug_dataset_path = '/GTSRB/aug_dataset/'

    self.experiment_list = {'3':{'train': [self.org_dataset_path, self.aug_dataset_path], 'test':self.org_dataset_path},
                      '4':{'train': [self.org_dataset_path, self.aug_dataset_path], 'test':self.aug_dataset_path},
                      '5':{'train': [self.aug_dataset_path], 'test':self.org_dataset_path},
                      '6':{'train': [self.aug_dataset_path], 'test':self.aug_dataset_path},
                      '2':{'train': [self.org_dataset_path], 'test':self.aug_dataset_path},
                      '1':{'train': [self.org_dataset_path], 'test':self.org_dataset_path},}
  

  
    classnamesPath =  './GTSRB/classnames.txt'
    f_classnames = open(classnamesPath, 'r')
    data_lines = f_classnames.readlines()
    
    for i in range(len(data_lines)):
        self.class_names.append(data_lines[i][0:-1])
  

    def load_exp_dataset(self):
        for i in range(0, 43):
            direc = str(i).zfill(5)
            #train = train_file_paths(train , '/GTSRB/' + direc, i)
            for x in self.experiment_list[exp][split]:
                for root, dirs, files in os.walk(x + split + direc):
                    for file in files:
                        if '.ppm' in file or '.png' in file:
                            file_path = os.path.join(root, file)
                            self.inputs.append(file_path)
                            self.targets.append(i)
   

    # with open('/GTSRB/GT-final_test.csv', 'r') as file:
    #     csv_reader = csv.reader(file)
    #     for row in csv_reader:
    #         if row[0].split(';')[7] in test_labels:
    #                test_file.append([row[0].split(';')[0], int(row[0].split(';')[7])])

    # final_test  = []
    # for i in test:
    #     for j in test_file:
    #         if j[0] in i.split('/')[6]:
    #             final_test.append([i, j[1]])
    #             break

    # print(final_test[0:10])
    # print(len(final_test))


    assert(self.n_classes == len(self.class_names))

    print('GTSRB %d classes'%(len(self.class_names)))
    print('Load GTSRB %s: %d samples'%(split, len(self.targets)))

        

  def __len__(self):
    return len(self.inputs)


  def __getitem__(self, index):
    img_path = self.inputs[index]
    gt = self.targets[index]
    gt = torch.ones(1).type(torch.LongTensor)*gt

    # Load images and templates. perform augmentations
    img = imageio.imread(img_path)
    img = np.array(img, dtype=np.uint8)
    template = imageio.imread('/GTSRB/' + 'template_ordered/%02d.jpg'%(gt+1))
    template = np.array(template, dtype=np.uint8)

    if random.random() < self.proto_rate:
        img = np.copy(template)

    if self.augmentations is not None:
        img, template = self.augmentations(img, template)

    if self.is_transform:
        img = self.transform(img)
        template = self.transform(template)

    return img, gt, template
    
  def transform(self, img):
    img = img.astype(np.float64)
    img -= self.mean
    if self.img_size is not None:
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
    # Resize scales images from 0 to 255, thus we need
    # to divide by 255.0
    img = img.astype(float) / 255.0
    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    
    return img


  def load_template(self, target, augmentations=None):

    # if augmentation is not specified, use self.augmentations. Unless use input augmentation option.
    if augmentations is None:
        augmentations = self.augmentations
    img_paths = []
    
    for id in target:
        img_paths.append('/GTSRB/' + '/template_ordered/%02d.jpg'%(id+1))

    target_img = []
    for img_path in img_paths:
        img = imageio.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        if augmentations is not None:
            img, _ = augmentations(img, img)
        if self.transform:
            img = self.transform(img)

        target_img.append(img)

    return torch.stack(target_img, dim=0)

