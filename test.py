import os
import glob
from keras.models import Model
from keras.layers import *
from keras.utils import get_file
import keras.backend as K
import numpy as np
import tensorflow as tf
import pandas as pd
from math import sin, cos
import math
import cv2
import matplotlib.pyplot as plt
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session, clear_session
from operator import itemgetter
from scipy.ndimage.filters import maximum_filter
from scipy.spatial.transform import Rotation as R
from tqdm.notebook import tqdm
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure

from DataGen import test_data_gen, get_test_parameters

def sigmoid(x):
    x = np.clip(x, -50, None)
    return 1 / (1 + np.exp(-x))

def decode3(pred, trans, Mi, xs=640, ys=128, w=None, interpolation=cv2.INTER_LANCZOS4, sel=None):
    hms=0
    masks=0
    masks2=0
    ts=0
    rs=0
    
    msk = np.zeros((ys,xs))
    msk[9:-8,31:-31]=1
    msk = cv2.GaussianBlur(msk,(65,19), 21)
    msk = msk/msk.max()
    
    if sel is None:
        sel = np.arange(len(trans))
        
    if w is None:
        w = np.ones(len(trans))
        
    for i in sel:
        alpha, beta, gamma, flip = trans[i]
        alpha = alpha*np.pi/180.
        beta  = beta *np.pi/180.
        gamma = gamma*np.pi/180.

        Mat, Rot = rotateImage(alpha, beta, gamma, dx=1691.5+2000)
        Ri = np.linalg.inv(Mat)
        Roti = np.linalg.inv(Rot)
        
        Matrix = np.dot(Mf, np.dot(Ri,Mi))
                
        if flip:
            hm = sigmoid(pred[0][i,:,::-1,0])
            t = pred[-1][i,:,::-1,-4:]
            r = pred[-1][i,:,::-1,:4]
            t[:,:,0] = -t[:,:,0]
            r[:,:,2:] = -r[:,:,2:]
        else:
            hm = sigmoid(pred[0][i,:,:,0])
            t = pred[-1][i,:,:,-4:]
            r = pred[-1][i,:,:,:4]

        hm = cv2.warpPerspective(hm, Matrix, (xo//4+sxo*2,yo//4+syo+24), flags=interpolation)
        t = cv2.warpPerspective(t, Matrix, (xo//4+sxo*2,yo//4+syo+24), flags=interpolation)
        r = cv2.warpPerspective(r, Matrix, (xo//4+sxo*2,yo//4+syo+24), flags=interpolation)
        mask = cv2.warpPerspective(msk.copy(), Matrix, (xo//4+sxo*2,yo//4+syo+24))
        
        ti = np.dot(Roti, np.reshape(t, (-1,4)).T).T.reshape(t.shape)
        
        yaw = r[:,:,0]
        pitch = np.arctan2(r[:,:,2], r[:,:,1])
        roll = r[:,:,3]+np.pi
        rr = np.dstack([-pitch, -yaw, -roll])
        
        ri_ = r.copy()
        y,x = np.where(hm>0.01)
        if len(y)>0:
            r1 = R.from_euler('xyz', rr[y,x], degrees=False)
            r2 = R.from_euler('xyz', [beta, -alpha, -gamma], degrees=False).inv()
            ri = (r2*r1).as_euler('xyz')*(-1)
            #ri = ri.reshape(rr.shape)
            ri_[y,x,0] = ri[:,1]
            ri_[y,x,1] = np.cos(ri[:,0])
            ri_[y,x,2] = np.sin(ri[:,0])
            ri_[y,x,3] = ri[:,2]%(np.pi*2)-np.pi

        mask = mask*w[i]
        if i==0:
            mask2 = mask
        else:
            mask2 = mask*(hm>0.01)

        rs = ri_*mask2[...,np.newaxis]+rs
        ts = ti*mask[...,np.newaxis]+ts
        hms = hm*mask+hms
        masks = mask+masks
        masks2 = mask2+masks2
    hms[masks>0] = hms[masks>0]/masks[masks>0]
    ts[masks>0] = ts[masks>0]/masks[...,np.newaxis][masks>0]
    rs[masks2>0] = rs[masks2>0]/masks2[...,np.newaxis][masks2>0]
    return hms, ts, rs, masks

def main():
    model = model.load(...)
    test_parameters = get_test_parameters
    parameters = test_parameters['rot']
    weights = test_parameters['weights']
    ip_ref = get_ref()

    preds = model.predict(next(test_data_gen))


    sub = pd.read_csv("data/sample_submission.csv")
    count=np.zeros(len(sub))

    for idx in tqdm(range(len(sub))):

        hms = 0
        dofs = 0
        masks = 0

        for trans, w, path, Mi, xs, mw in params:

            preds = model.predict(next(test_data_gen))

            hm, tsf, rsf, mask = decode3(preds, trans, Mi, xs=xs, w=w)
            dof = np.dstack([rsf,tsf])

            if norm_mask:
                mask = mask/np.max(mask)

            mask = mask*mw

            hms = hms + hm*mask
            mask = mask[...,np.newaxis]  
            dofs = dofs + dof*mask
            masks = masks + mask

        masks[masks==0] = masks[masks==0] + 1e-7
        p = hms/masks[:,:,0]
        dofs = dofs/masks

        ih,iw = hm.shape
        reg = get_ref(iw=iw, ih=ih)-[sxo, syo]

        #p = hms
        local_maxi = p*(p == maximum_filter(p,footprint=np.ones(kernel)))>thr
        py,px = np.where(local_maxi)

        if avg:
            markers = measure.label(local_maxi)
            labels_ws = watershed(-p, markers, mask=p>min(thr,thr2))
            scores = []
            pdx = []
            pdy = []
            dof_=[]
            for i in range(1, markers.max()+1):
                y,x = np.where(labels_ws==i)
                score = p[y,x]
                scores.append(score.max())
                score = score*(score >= min(scores[-1]*r, thr2))
                score = score**pwr
                ss = score.sum()
                dof_.append((dofs[y,x]*score[:,np.newaxis]).sum(0)/ss)
                pdx.append(((reg[y,x,0])*score).sum()/ss)
                pdy.append(((reg[y,x,1])*score).sum()/ss)
        else:
            scores = p[py,px]
            pdx = reg[py,px,0]
            pdy = reg[py,px,1]
            dof_= dofs[py,px,:]

        output = np.zeros((len(dof_),7))

        for j in range(len(dof_)):        
            pp = postprocess2(dof_[j])
            output[j,:6] = pp[:6]
            output[j, 2] = (output[j,2]+np.pi)%(np.pi*2)-np.pi
            output[j,-1] = scores[j]

            #if optimize ==3:

            X,Y,Z = np.dot(np.linalg.inv(M), [pdx[j]*4, pdy[j]*4, 1])
            X=X/Z+x_shift
            Y=Y/Z+1355
            x1,y1,z1 = get_xyz_from_XYr(X, Y, pp[-1])
            x0,y0    = get_xy_from_XYz(X, Y, pp[-2])
            if optimize ==3:
                output[j,3:5] = (x0+x1)/2,(y0+y1)/2
            else:
                output[j,3:6] = (x0+x1)/2,(y0+y1)/2, (z1+pp[-2])/2

        count[idx] = len(dof_)
        sub.iloc[idx].PredictionString = ' '.join(output.reshape(-1).astype('str'))

    print(thr, thr2, r, pwr, avg, optimize, count.sum())
    return sub, count