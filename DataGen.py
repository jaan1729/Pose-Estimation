import tensorflow as tf
import os
import glob
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import get_file
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
from math import sin, cos
import math
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
#from keras.backend.tensorflow_backend import set_session, clear_session
from operator import itemgetter
from scipy.ndimage.filters import maximum_filter
from scipy.spatial.transform import Rotation as R
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure
from math import floor
import random
import shutil
import argparse



test = pd.read_csv("data/test")
img_h, img_w = 1355, 3384
iimg_h, iimg_w = 512, 1280
ip_h, ip_w = 512, 2560
op_h, op_w = 128, 640
batch_size = 1
k = np.array([[2304.5479, 0,  1686.2379],
           [0, 2305.8757, 1354.9849],
           [0, 0, 1]], dtype=np.float32)

def read_img(path, idx):
    img = cv2.imread(path + '_images/%s.jpg'%train.iloc[idx].ImageId)[:,:,::-1]
    return img

def str_to_coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords

def str_to_arrays(s):
    rot = []
    trx = []
    for l in np.array(s.split()).reshape([-1, 7]):
        r = []
        t = []
        r.append(l[2].astype('float'))
        r.append(l[1].astype('float'))
        r.append(l[3].astype('float'))
        t.append(l[4].astype('float'))
        t.append(l[5].astype('float'))
        t.append(l[6].astype('float'))
        rot.append(r)
        trx.append(t)
    return rot, trx

def pixel_coords(coords):
    xc = [c['x'] for c in coords]
    yc = [c['y'] for c in coords]
    zc = [c['z'] for c in coords]
    P = np.array(list(zip(xc, yc, zc))).T
    img_p = np.dot(k, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    u = img_p[:, 0]
    v = img_p[:, 1]-1355
    zc = img_p[:, 2]
    return u,v

def get_heatmap(p_x, p_y, output_height, output_width, sigma = 1):
    X1 = np.linspace(0, output_width-1, output_width)
    Y1 = np.linspace(0, output_height-1, output_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - floor(p_x)
    Y = Y - floor(p_y)
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma ** 2
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    heatmap = heatmap[:, :]
    return heatmap

def pose(coords,iimg_h, iimg_w):
    u, v = pixel_coords(coords)
    #print(len(u))
    regr = np.zeros([iimg_h, iimg_w, 8], dtype='float32')
    hm = np.zeros([iimg_h, iimg_w])
    #print(u,v)
    for p_x, p_y, regr_dict in zip(u, v, coords):
        
        regr_dict['r'] = np.sqrt(regr_dict['x']**2 + regr_dict['y']**2 + regr_dict['z']**2)
        for name in ['x', 'y', 'z', 'r']:
            regr_dict[name] = regr_dict[name]/100
        regr_dict['roll'] = regr_dict['roll'] + np.pi
        if regr_dict['roll'] > np.pi: regr_dict['roll'] = regr_dict['roll'] - 2*np.pi
        if regr_dict['pitch'] > np.pi: regr_dict['pitch'] = regr_dict['pitch'] - 2*np.pi
        regr_dict['pitch_sin'] = np.sin(regr_dict['pitch'])
        regr_dict['pitch_cos'] = np.cos(regr_dict['pitch'])
        regr_dict.pop('pitch')
        regr_dict.pop('id')
        if regr_dict['yaw'] > np.pi: regr_dict['yaw'] = regr_dict['yaw'] - 2*np.pi
        p_x, p_y = int(p_x*iimg_w/img_w), int(p_y*iimg_h/img_h)
        #print(p_x, p_y)
        if p_x >= -7 and p_x < iimg_w and p_y >= -7 and p_y < iimg_h:
            regr_v = np.array([regr_dict['yaw'],regr_dict['pitch_cos'],regr_dict['pitch_sin'],
                               regr_dict['roll'],regr_dict['x'],regr_dict['y'], regr_dict['z'], regr_dict['r']])
            if p_y<7 or p_x<7:
              if p_y<7 and p_x<7:
                regr[0:p_y+7, 0:p_x+7] = regr_v
              elif p_y<7:
                regr[0:p_y+7, p_x-7:p_x+7] = regr_v
              elif p_x<7:
                regr[p_y-7:p_y+7, 0:p_x+7] = regr_v

            else:
              regr[p_y-7:p_y+7, p_x-7:p_x+7] = regr_v
            
            hm_temp = get_heatmap(p_x, p_y, iimg_h, iimg_w)
            hm[:,:] = np.maximum(hm[:,:], hm_temp[:,:])
            #print(u,v)
    return hm, regr

def RotateImage(alpha = 0, beta = 0, gamma = 0, dx = ip_w/2, dy=0):
    fx, dx = 2304.5479, dx
    fy, dy = 2305.8757, dy
    
    # Projection 2D -> 3D matrix
    A1 = np.array([[1/fx,    0, -dx/fx],
                   [0,    1/fy, -dy/fy],
                   [0,       0,      1],
                   [0,       0,      1]])
    
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([[1,          0,           0, 0],
                   [0, cos(alpha), -sin(alpha), 0],
                   [0, sin(alpha),  cos(alpha), 0],
                   [0,          0,           0, 1]])
    
    RY = np.array([[cos(beta),  0,  -sin(beta), 0],
                   [0,          1,           0, 0],
                   [sin(beta),  0,   cos(beta), 0],
                   [0,          0,           0, 1]])
    
    RZ = np.array([[cos(gamma), -sin(gamma), 0, 0],
                   [sin(gamma),  cos(gamma), 0, 0],
                   [0,          0,           1, 0],
                   [0,          0,           0, 1]])
    
    # Composed rotation matrix with (RX, RY, RZ)
    Rot = np.dot(RZ, np.dot(RX, RY))
    
    # 3D -> 2D matrix
    A2 = np.array([[fx, 0, dx, 0],
                   [0, fy, dy, 0],
                   [0, 0,   1, 0]])
    # Final transformation matrix
    trans = np.dot(A2,np.dot(Rot, A1))
    
    return trans, Rot

def get_parameters():
    pts1=np.float32([[64,4],[1280-64,4],[64,512-4],[1280-64,512-4]])
    pts2=np.float32([[64,4],[2560-64,4],[640+64,512-4],[2560-640-64,512-4]])
    M=cv2.getPerspectiveTransform(pts1,pts2)
    

    trans =[[0,0,0], [0,5,0], [0,-5,0], [0,10,3], [0,-10,-3], [2,0,0], [-5,0,0]]
    a = np.zeros((len(trans)*2, 4))
    a[:,:3] = trans+trans
    a[len(trans):,-1] = 1
    trans = a
    parameters = dict()
    parameters['pers'] = M
    parameters['test_rot'] = trans
    parameters['test_weights'] = [2,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1]
    alpha = (np.random.random()*6-3)*np.pi/180
    beta = (np.random.random()*50-25)*np.pi/180
    gamma = (np.random.random()*6-3)*np.pi/180 + beta/3
    trans = [alpha,beta,gamma]
    parameters['train_rot'] = trans
    return parameters
def add_noise(image):
    row,col,ch= image.shape
    mean = 0
    var = np.random.random()*0.01 #0.001~0.01
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy

def normalize_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return (np.float32(image) - mean) / std


lookUpTable = np.empty((1,256), np.uint8)
g = 1.45
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, g) * 255.0, 0, 255)

def get_enhanced(img):
  i = np.random.randint(3)
  if i == 0:
    img_enh = cv2.LUT(img, lookUpTable)
  elif i==1:
    a = 2.5
    b = 15
    img_enh  = cv2.convertScaleAbs(img, alpha=a, beta=b)
  elif i==2:
    img_enh = add_noise(img).astype('int')
  return img_enh


def transform_and_save(df=train, path):    
    ImgIds = []
    if os.path.isdir('train_inputs'):
      shutil.rmtree('train_inputs', ignore_errors=True)
    if os.path.isdir('train_hms'):
      shutil.rmtree('train_hms', ignore_errors=True)
    if os.path.isdir('train_regs'):
      shutil.rmtree('train_regs', ignore_errors=True)
    
    os.mkdir('train_inputs')
    os.mkdir('train_hms')
    os.mkdir('train_regs')
    print('created directories')
    for n in range(0,len(df)):
      img_file = path + '_images/%s.jpg'%df.iloc[n].ImageId
      mask_file = path + '_masks/%s.jpg'%df.iloc[n].ImageId
      img = cv2.imread(img_file)
      mask = cv2.imread(mask_file)
      
      
      parameters = get_parameters()
      if mask is not None:
          img = img*(mask<128)   
                  
      img = img[1355:,:,::-1]
      #parameters = get_parameters()
      string = train.iloc[n].PredictionString
      
      
      roti, trxi = str_to_arrays(string)
      
      #print(rot)
      resized_img = cv2.resize(img, (iimg_w,iimg_h))
      
      #print(s)
      rot = np.array(roti)
      trx = np.array(trxi)
      alpha,beta,gamma = parameters['train_rot']
      M = parameters['pers']
      
      
      RotP, RotM = RotateImage(alpha, beta, gamma, dx = img_w/2)
     
      perspected_img = cv2.warpPerspective(resized_img, M, (ip_w, ip_h))
      perspected_img_e = get_enhanced(perspected_img)
      
      rotated_img = cv2.warpPerspective(img, RotP, (img_w, img_h))
      r_rotated_img = cv2.resize(rotated_img, (iimg_w,iimg_h)) 
      p_rotated_img = cv2.warpPerspective(r_rotated_img, M, (ip_w, ip_h))
      p_rotated_img_e = get_enhanced(p_rotated_img)
      
      
      r1 = R.from_euler('yxz',[-beta,alpha,gamma], degrees = False)
      r2 = R.from_euler('yxz', rot, degrees = False)
      r = r1*r2
      rot = r.as_euler('yxz')
      trx = np.dot(RotM[:3,:3], trx.T).T
      
      pred_str = ''
      for dof_id in range(len(rot)):
        dof_str = ' 1 %f %f %f  %f %f %f'%(rot[dof_id, 1],rot[dof_id, 0],rot[dof_id, 2],trx[dof_id, 0],trx[dof_id, 1],trx[dof_id, 2])
        pred_str += dof_str
      
      perspected_img = perspected_img.astype(np.uint8)
      perspected_img_e = perspected_img_e.astype(np.uint8)
      p_rotated_img = p_rotated_img.astype(np.uint8)
      p_rotated_img_e = p_rotated_img_e.astype(np.uint8)
      np.save('train_inputs/%s_%d_n.npy'%(df.iloc[n].ImageId,0),perspected_img)
      np.save('train_inputs/%s_%d_e.npy'%(df.iloc[n].ImageId,0),perspected_img_e)
      np.save('train_inputs/%s_%d_n.npy'%(df.iloc[n].ImageId,1),p_rotated_img)
      np.save('train_inputs/%s_%d_e.npy'%(df.iloc[n].ImageId,1),p_rotated_img_e)
      
      ImgIds.append('%s_%d.npy'%(df.iloc[n].ImageId,0))
      ImgIds.append('%s_%d.npy'%(df.iloc[n].ImageId,1))

      coords = str_to_coords(string)
      hm, reg = pose(coords,iimg_h,iimg_w)
      
      perspected_hm = cv2.warpPerspective(hm,M, (ip_w, ip_h))
      perspected_reg = cv2.warpPerspective(reg,M, (ip_w, ip_h), flags = cv2.INTER_NEAREST)

      perspected_hm_tf = tf.reshape(perspected_hm, [1,ip_h,ip_w,1])
      op_hm = tf.nn.max_pool2d(perspected_hm_tf, 4, 4, padding = 'VALID')
      reg = cv2.resize(perspected_reg, (op_w,op_h), interpolation = cv2.INTER_NEAREST)
      

      op_hm = np.squeeze(op_hm.numpy())
      
      op_hm[(op_hm*(op_hm == maximum_filter(op_hm,footprint=np.ones((3,3))))>0.1)] = 1
      op_reg = np.zeros_like(reg)
      y,x = np.where(op_hm == 1)
      op_reg[y,x,:] = reg[y,x,:]

      np.save('train_hms/%s_%d.npy'%(df.iloc[n].ImageId,0),op_hm)
      np.save('train_regs/%s_%d.npy'%(df.iloc[n].ImageId,0),op_reg)

      coords = str_to_coords(pred_str)
      hm, reg = pose(coords,iimg_h,iimg_w)
      
      perspected_hm = cv2.warpPerspective(hm,M, (ip_w, ip_h))
      perspected_reg = cv2.warpPerspective(reg,M, (ip_w, ip_h), flags = cv2.INTER_NEAREST)

      perspected_hm_tf = tf.reshape(perspected_hm, [1,ip_h,ip_w,1])
      op_hm = tf.nn.max_pool2d(perspected_hm_tf, 4, 4, padding = 'VALID')
      reg = cv2.resize(perspected_reg, (op_w,op_h), interpolation = cv2.INTER_NEAREST)
      

      op_hm = np.squeeze(op_hm.numpy())
      
      op_hm[(op_hm*(op_hm == maximum_filter(op_hm,footprint=np.ones((3,3))))>0.1)] = 1
      op_reg = np.zeros_like(reg)
      y,x = np.where(op_hm == 1)
      op_reg[y,x,:] = reg[y,x,:]
      op_hm = np.reshape(op_hm, op_hm.shape + (1,))

      np.save('train_hms/%s_%d.npy'%(df.iloc[n].ImageId,1),op_hm)
      np.save('train_regs/%s_%d.npy'%(df.iloc[n].ImageId,1),op_reg)
      if n%200 == 0: print('completed %d images'%n)
    np.save('image_names.npy', np.array(ImgIds))

def train_generator(df, batch_size  ):                
  
    while True:
        xo,yo = ip_w, ip_h
        ref = np.reshape(np.arange(0, xo*yo), (yo, xo, -1))
        ref_x = ref % xo
        ref_y = ref // xo
        ref = np.dstack([(ref_x-(xo-1)/2)/100, ref_y/100])
        coor = ref[::4, ::4]
        transformed_images = []
        transformed_hms = []
        transformed_regs = []
        input_ref = []
        parameters = get_parameters()
        M = parameters['pers']
        for i in range(len(df)):

          img = np.load('train_inputs/%s_%s'%(df[i], random.choice(['n','e']))+'.npy')
          img = normalize_image(img/255.)
          s = df[i,1]
          
          op_hm = np.load('train_hms/%s.npy'%(df[i]))
          op_reg = np.load('train_regs/%s.npy'%(df[i]))
          
          img_flipped = img[:,::-1,:].copy()

          op_hm_f = op_hm[:,::-1].copy()
          op_reg_f = op_reg[:,::-1,:].copy()
          op_reg_f[:,:,2:5] = -op_reg_f[:,:,2:5]
          transformed_images.append(img)
          transformed_images.append(img_flipped)
          transformed_hms.append(op_hm)
          transformed_hms.append(op_hm_f) 
          transformed_regs.append(op_reg)
          transformed_regs.append(op_reg_f)
          input_ref.append(coor)
          input_ref.append(coor)
          if (i+1)%(batch_size//2) == 0:
            
            t_images = np.array(transformed_images)
            t_hms = np.array(transformed_hms)
            t_regs = np.array(transformed_regs)
            ip_ref = np.array(input_ref)
            transformed_images = []
            transformed_hms = []
            transformed_regs = []
            input_ref = []
            d = dict()
            d['h1'] = t_hms
            d['d1'] = t_regs
            yield ([t_images, ip_ref], d)

            
            #print(transformed_images.shape, transformed_hms.shape, transformed_regs.shape, input_ref.shape)

def test_generator(sub):

  while True:
    xo,yo = ip_w, ip_h
    ref = np.reshape(np.arange(0, xo*yo), (yo, xo, -1))
    ref_x = ref % xo
    ref_y = ref // xo
    ref = np.dstack([(ref_x-(xo-1)/2)/100, ref_y/100])
    coor = ref[::4, ::4]
    transformed_images = []
    
    parameters = get_parameters()
    M = parameters['pers']
    trans = parameters['test_rot'] 
    w = parameters['test_weights']

    input_ref = []
    for i in range(len(sub)):    
      image = cv2.imread('../input/pku-autonomous-driving/test_images/%s.jpg'%sub.iloc[i].ImageId)
      mask = cv2.imread('../input/pku-autonomous-driving/test_masks/%s.jpg'%sub.iloc[i].ImageId)
      if mask is not None:
          image = image*(mask<128)            
      image = image[1355:,:,::-1]
      
      for alpha, beta, gamma, flip in trans:
          alpha = alpha*np.pi/180.
          beta  = beta *np.pi/180.
          gamma = gamma*np.pi/180.                
          Mat, Rot = RotateImage(alpha, beta, gamma)
          
          img = cv2.warpPerspective(image.copy(), np.dot(M,Mat), (xo,yo), flags=cv2.INTER_LINEAR)

          if flip:        
              img = img[:,::-1]                

          img = normalize_image(img/255.)    

          coor = ref[::4, ::4]

          inputs.append(img)
          input_coor.append(coor)

      tmp_inputs = np.array(inputs)
      tmp_input_coor = np.array(input_coor)
      inputs = []
      input_coor = []
      
      yield [tmp_inputs, tmp_input_coor]

def main():
  path = "data/train"
  train = pd.read_csv("data/train.csv")
  transform_and_save(train, path)
  print('completed saving input files')
    
if name == '__main__':
  main()