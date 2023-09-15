import os
import random
import matplotlib
import torch
import cv2
import math
import numpy as np
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import imageio.v2 as imageio
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from pathlib import Path
from torch import optim, nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import scipy
import csv
from models import get_model
from augmentations import *
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score


# torchvision==0.2.1 -
# torch==1.0.0 - 
# matplotlib==2.2.2 - 
# scipy==1.1.0 - 
# numpy==1.15.4 - 
# Pillow==6.0.0




def write_conf_mat(log_save_dir, filename, data, epoch):
with open(log_save_dir + filename, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows([str(epoch)])
    csv_writer.writerows(['\n'])
    csv_writer.writerows(data)
    csv_writer.writerows(['\n'])


# Setup
parser = ArgumentParser(description='Variational Prototyping Encoder (VPE)')
parser.add_argument('--seed',       type=int,   default=42,             help='Random seed')
parser.add_argument('--arch',       type=str,   default='vaeIdsiaStn',  help='network type: vaeIdsia, vaeIdsiaStn')
parser.add_argument('--exp',        type=str,   default='1',     help='experiment setting')
parser.add_argument('--resume',     type=str,   default=None,           help='Resume training from previously saved model')
parser.add_argument('--epochs',     type=int,   default=1000,           help='Training epochs')
parser.add_argument('--lr',         type=float, default=1e-4,           help='Learning rate')
parser.add_argument('--batch_size', type=int,   default=128,            help='Batch size')
parser.add_argument('--img_cols',   type=int,   default=64,             help='resized image width')
parser.add_argument('--img_rows',   type=int,   default=64,             help='resized image height')
parser.add_argument('--workers',    type=int,   default=0,              help='Data loader workers')
parser.add_argument('--image_save_dir',    type='str',   default='/images/',              help='')
parser.add_argument('--log_save_dir',    type='str',   default='/logs/',              help='')
args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)
plt.switch_backend('agg')  # Allow plotting when running remotely

#initialise tensorboard
writer = SummaryWriter()

save_epoch = 200 # save log images per save_epoch



result_path = args.log_save_dir
if not os.path.exists(result_path):
  os.makedirs(result_path)

outimg_path =  args.image_save_dir
if not os.path.exists(outimg_path):
  os.makedirs(outimg_path)

f_loss = open(os.path.join(result_path, "log_loss.txt"),'w')
f_loss.write('Experiment setting: %s\n'%args.exp)
f_loss.write('Image save directory: %s\n'%args.image_save_dir)
f_loss.write('Logs save directory: %s\n'%args.log_save_dir)
f_loss.write('Network type: %s\n'%args.arch)
f_loss.write('Learning rate: %05f\n'%args.lr)
f_loss.write('batch-size: %s\n'%args.batch_size)
f_loss.write('img_cols: %s\n'%args.img_cols)
f_loss.write('Augmentation type: flip, centercrop\n\n')
f_loss.close()

f_iou = open(os.path.join(result_path, "log_acc.txt"),'w')
f_iou.close()

# set up GPU

# we could do os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

import gc
gc.collect()
torch.cuda.empty_cache()



data_aug_tr= Compose([Scale(args.img_cols), # resize longer side of an image to the defined size
                      CenterPadding([args.img_rows, args.img_cols]), # zero pad remaining regions
                      RandomHorizontallyFlip(), # random horizontal flip
                      RandomRotate(180)])  # ramdom rotation

data_aug_te= Compose([Scale(args.img_cols), 
                     CenterPadding([args.img_rows, args.img_cols])])



tr_loader = gtsrbLoader(args.exp, is_transform=True, split='train', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_tr)
te_loader = gtsrbLoader(args.exp, is_transform=True, split='test', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_te)
trainloader = DataLoader(tr_loader, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
testloader = DataLoader(te_loader, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)


# define model or load model
net = get_model(args.arch, n_classes=None)
# net.load_state_dict(torch.load('gtsrb_testBest_net_aug.pth'))
net.cuda()

if args.resume is not None:
  pre_params = torch.load(args.resume)
  net.init_params(pre_params)


reconstruction_function = nn.BCELoss()
reconstruction_function.reduction = 'sum'
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD

# Construct optimiser
optimizer = optim.Adam(net.parameters(), lr=args.lr) # 1e-4

num_train = len(tr_loader.targets)
num_test = len(te_loader.targets)
batch_iter = math.ceil(num_train/args.batch_size)
batch_iter_test = math.ceil(num_test/args.batch_size)

def train(e):
  n_classes = tr_loader.n_classes
  n_classes_te = te_loader.n_classes
  print('start train epoch: %d'%e)
  net.train()
  epoch_loss = 0
  for i, (input, target, template) in enumerate(trainloader):

    optimizer.zero_grad()
    target = torch.squeeze(target)
    input, template = input.cuda(), template.cuda()

    recon, mu, logvar, input_stn = net(input)
    loss = loss_function(recon, template, mu, logvar) # reconstruction loss
    print('Epoch:%d  Batch:%d/%d  loss:%08f'%(e, i, batch_iter, loss/input.numel()))
    
    epoch_loss += loss/input.numel()
    
    f_loss = open(os.path.join(result_path, "log_loss.txt"),'a')


    f_loss.write('Epoch:%d  Batch:%d/%d  loss:%08f\n'%(e, i, batch_iter, loss/input.numel()))
    f_loss.close()
    
    loss.backward()
    optimizer.step()

    if (e%save_epoch == 0):
      out_folder =  "%s/Epoch_%d_train"%(outimg_path, e)
      out_root = Path(out_folder)
      if not out_root.is_dir():
        os.mkdir(out_root)

      torchvision.utils.save_image(input, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(input_stn, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2) 
      torchvision.utils.save_image(recon, '{}/batch_{}_recon.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(template, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2)
  
  writer.add_scalar('Loss/Train', epoch_loss, e)

  if e%save_epoch == 0:
    class_target = torch.LongTensor(list(range(n_classes)))
    class_template = tr_loader.load_template(class_target)
    class_template = class_template.cuda()
    with torch.no_grad():
      class_recon, class_mu, class_logvar, _ = net(class_template)
    
    torchvision.utils.save_image(class_template, '{}/templates.jpg'.format(out_folder), nrow=8, padding=2)
    torchvision.utils.save_image(class_recon, '{}/templates_recon.jpg'.format(out_folder), nrow=8, padding=2)
  
def get_predictions(pred, class_feature, label, n_classes):

  sample_distance = torch.ones(pred.shape[0], n_classes)*math.inf

  pred = pred.cpu() # batch x latent size
  class_feature = class_feature.cpu() # n_classes x latent size
  
  for i in range(n_classes):
    cls_feat = class_feature[i,:]
    cls_mat = cls_feat.repeat(pred.shape[0],1)
    # euclidean distance
    sample_distance[:,i] = torch.norm(pred - cls_mat,p=2, dim=1)
  
  sample_distance = sample_distance.cpu().numpy()
  indices = np.argsort(sample_distance, axis=1) # sort ascending order\
 
  predictions = indices[:,0].tolist()
  true_labels = label.tolist()

  return  predictions, true_labels


best_f1 = 0
best_acc = 0 
best_conf_mat = []



def test(e, best_acc, val_trigger):
  predictions = []
  true_labels = []
  n_classes = te_loader.n_classes
  print('start test epoch: %d'%e)
  net.eval()
  # get template latent z
  class_target = torch.LongTensor(list(range(n_classes)))
  class_template = te_loader.load_template(class_target)
  class_template = class_template.cuda()
  
  with torch.no_grad():
    class_recon, class_mu, class_logvar, _ = net(class_template)
  

  predictions = []
  true_labels = []

  test_epoch_loss = 0
  for i, (input, target, template) in enumerate(testloader):

    target = torch.squeeze(target)
    input, template = input.cuda(), template.cuda()
    with torch.no_grad():
      recon, mu, logvar, input_stn  = net(input)
    
    test_loss = loss_function(recon, template, mu, logvar)
    test_epoch_loss += test_loss/input.numel()
    preds, true = get_predictions(mu, class_mu, target, n_classes)

    for p in preds:
        predictions.append(p)
    for t in true:
        true_labels.append(t)
 
    
    print('Epoch:%d  Batch:%d/%d  processing...'%(e, i, batch_iter_test))

    if (e%save_epoch == 0):
      out_folder =  "%s/Epoch_%d_test"%(outimg_path, e)
      out_root = Path(out_folder)
      if not out_root.is_dir():
        os.mkdir(out_root)

      torchvision.utils.save_image(input, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(input_stn, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2) 
      torchvision.utils.save_image(recon, '{}/batch_{}_recon.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(template, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2)

  if e%save_epoch == 0:
      torchvision.utils.save_image(class_template, '{}/templates.jpg'.format(out_folder), nrow=8, padding=2)  
      torchvision.utils.save_image(class_recon, '{}/templates_recon.jpg'.format(out_folder), nrow=8, padding=2) 
  
  
  writer.add_scalar('Loss/test', test_epoch_loss, e)

 

  print('========epoch(%d)========='%e)
 
  class_report = classification_report(true_labels, predictions, labels = [i for i in range(0, 43)])
  f1_score = f1_score(true_labels, predictions, labels = [i for i in range(0, 43)], average='macro')
  conf_mat = confusion_matrix(true_labels, predictions)

  if f1_score > best_f1:
    best_f1 = f1_score
    best_conf_mat = conf_mat.tolist()

  conf_mat = np.array(conf_mat)
 
  FP = conf_mat.sum(axis=0) - np.diag(conf_mat)  
  FN = conf_mat.sum(axis=1) - np.diag(conf_mat)
  TP = np.diag(conf_mat)
  TN = conf_mat.sum() - (FP + FN + TP)

  # Sensitivity, hit rate, recall, or true positive rate
  TPR = TP/(TP+FN)
  print('\ntrue postive rate', TPR)
  # Specificity or true negative rate
  TNR = TN/(TN+FP) 
  print('\ntrue negative rate', TNR)
  # Fall out or false positive rate
  FPR = FP/(FP+TN)
  print('\nfalse postive rate', FPR)
  # False negative rate
  FNR = FN/(TP+FN)
  print('\nfalse negative rate', FNR)
  print('====================================')


  f_iou = open(os.path.join(result_path, "log_acc.txt"),'a')
  f_iou.write('\n' + 'classification report\n' + str(class_report) + '\n' + 'TPR\n' + str(TPR) + '\n' + 'TNR\n' + str(TNR) + '\n' +  'FPR\n' + str(FPR) + '\n' + 'FNR\n' + str(FNR) + '\n' + str(conf_mat))
  f_iou.close()
  torch.save(net.state_dict(), os.path.join('%s_whole_dataset_exp1_e1000_final.pth'%args.dataset)) 

  return best_acc, best_f1, best_conf_mat


if __name__ == "__main__":
  out_root = Path(outimg_path)
  if not out_root.is_dir():
    os.mkdir(out_root)


  for e in range(1, args.epochs + 1):
    val_trigger = False
    train(e)
    best_acc, best_f1, best_conf_mat = test(e, best_acc, val_trigger)
    print('Accuracy:', best_acc)
    print('F1 score:', best_f1)
    csv_filename = args.exp + '_confusion_matrix_'
    write_conf_mat(args.log_save_dir, csv_filename, best_conf_mat, e)
    
writer.close()

