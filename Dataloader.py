import os

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


  
    classnamesPath =  './GTSRB/classnames.txt'
    f_classnames = open(classnamesPath, 'r')
    data_lines = f_classnames.readlines()
    
    for i in range(len(data_lines)):
        self.class_names.append(data_lines[i][0:-1])
  


    train= []
    test = []

    if exp == '1':
        train_org = load_original_dataset(self.org_dataset_path + self.split)
        train_aug = load_augmented_dataset(self.org_dataset_path + self.split)
        self.inputs = train_org + train_aug
        test_org = load_original_dataset('/GTSRB/org_dataset/test')


    if exp == '2':
        train_org = load_original_dataset('/GTSRB/org_dataset/train')
        train_aug = load_augmented_dataset('/GTSRB/aug_dataset/train')
        test_aug = load_augmented_dataset('/GTSRB/aug_dataset/test')

    if exp == '3':
        train_aug = load_augmented_dataset('/GTSRB/aug_dataset/train')
        test_org = load_original_dataset('/GTSRB/org_dataset/test')

    if exp == '4':
        train_aug = load_augmented_dataset('/GTSRB/aug_dataset/train')
        test_aug = load_augmented_dataset('/GTSRB/aug_dataset/test')

    if exp == '5':
        train_org = load_original_dataset('/GTSRB/org_dataset/train')
        test_aug = load_augmented_dataset('/GTSRB/aug_dataset/test')

    if exp == '6':
        train_org = load_original_dataset('/GTSRB/org_dataset/train')
        test_org = load_original_dataset('/GTSRB/org_dataset/test')
   

            

  

  def load_original_dataset(path):

    def train_file_paths(train, path, label):
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if '.ppm' in file or '.png' in file:
                    file_path = os.path.join(root, file)
                    train.append([file_path, label ])
            
        return train

    def test_file_paths(path):
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if '.ppm' in file or '.png' in file:
                    file_path = os.path.join(root, file)
                    test.append(file_path)

    train_classes = []

    for i in range(43):
        number = f"{i:05d}"
        train_classes.append(number)

    print(train_classes)


    train = []

    for i in range(0, 43):
        direc = str(i).zfill(5)
        train = train_file_paths(train , '/GTSRB/' + direc, i)


    for i in range(0, 43):
        direc = str(i).zfill(5)
        train = train_file_paths(train, '/GTSRB/aug_db/train/' + direc, i)
        
    directory_path = '/GTSRB/GTSRB_Test'
    # directory_path = '/home/abg96/vpe/drive0-storage/VPE/aug_db/test'
    test_file_paths(directory_path)

  

    import csv
    test_file = []

    test_labels = [str(i) for i in range(0, 43)]
    print(test_labels)

    with open('/GTSRB/GTSRB_Test/GT-final_test.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0].split(';')[7] in test_labels:
                   test_file.append([row[0].split(';')[0], int(row[0].split(';')[7])])

    final_test  = []
    for i in test:
        for j in test_file:
            if j[0] in i.split('/')[6]:
                final_test.append([i, j[1]])
                break

    print(final_test[0:10])
    print(len(final_test))




         
    # aug test

    # final_test= []
    # for i in test:
    #     #print(i)
    #     c = i.split('/')[8]
    #     c = c.split('.')[0]
    #     c = c.split('_')[1]
    #     final_test.append([i, int(c)])


    # print(final_test[0:10])





    # Splitting the data into training and testing sets
    # train_data, test_data = train_test_split(total, test_size=0.1, random_state=42)
    # print(type(train_data))
    # print(type(train_data[0]))
    # print(train_data[0])


    # Printing the results

    train_data = train
    test_data = final_test

    print("Training data:", len(train_data))

    print("Testing data:", len(test_data))

    if split == 'train':
        for pair in train_data:
            self.inputs.append(pair[0])
            self.targets.append(pair[1])
    else:
        for pair in test_data:
            self.inputs.append(pair[0])
            self.targets.append(pair[1])

    assert(self.n_classes == len(self.class_names))

    print('GTSRB %d classes'%(len(self.class_names)))
    print('Load GTSRB %s: %d samples'%(split, len(self.targets)))

        
        
    
    return

  def load_augmented_dataset():
    return




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

