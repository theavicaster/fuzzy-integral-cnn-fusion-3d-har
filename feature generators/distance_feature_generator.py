import numpy as np
import os
from scipy.misc import imresize
import scipy.io as sio
import glob
from PIL import Image
import tqdm
import math
import time

data_path = r'C:\Users\AVI\Desktop\UTD MHAD\Skeleton'
skel_paths = glob.glob(os.path.join(data_path, '*.mat'))

for path in tqdm.tqdm(skel_paths):
  
  filename = path.split('\\')[6]
  filenamesplit = filename.split('_') 
  action = filenamesplit[0]
  subject = filenamesplit[1]
  
  current = sio.loadmat(path)['d_skel']

  current = np.transpose(current , (2,0,1))
  print(current.shape)
  
  numframes = current.shape[0]
  
  featurelist = []
  
  for a in range(numframes):
  
    xyz = current[a,:,:]
    feature = []
    for i in range(0,20):
      for j in range (0,i):
          
          dij = (np.linalg.norm(xyz[i,:]-xyz[j,:]))
          feature.append( dij)

         
    featurelist.append(feature)
    
  
  featurelist = np.array(featurelist)

  
  for i in range(featurelist.shape[0]):
    maximum = np.max(featurelist[i])
    minimum = np.min(featurelist[i])  
    
    featurelist[i,:] = np.floor( (featurelist[i,:] - minimum) / (maximum -minimum)  * (255.0-0) )
  
  
  featurelist = imresize(featurelist,(70,featurelist.shape[1]),interp='bicubic')
  im = Image.fromarray(featurelist)
  
  odd = ['s1','s3','s5','s7']
  
  if subject in odd:
    filepath = r'C:/Users/AVI/Desktop/UTD MHAD/train/'
  else:
    filepath = r'C:/Users/AVI/Desktop/UTD MHAD/val/'
        
  filedir = filepath + action
  if not os.path.exists(filedir):
    os.mkdir(filedir)
  
  filepath = filepath + action + r'/' + filename.replace('.mat','.jpg')
  
  im.save(filepath)
    

