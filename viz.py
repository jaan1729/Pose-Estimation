import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.filters import maximum_filter
import json
from math import sin, cos
from DataGen import get_parameters
# Load a 3D model of a car
parameters = get_parameters()
M = parameters['pers']
img_h, img_w = 1355, 3384
iimg_h, iimg_w = 512, 1280
ip_h, ip_w = 512, 2560
op_h, op_w = 128, 640
batch_size = 1
k = np.array([[2304.5479, 0,  1686.2379],
           [0, 2305.8757, 1354.9849],
           [0, 0, 1]], dtype=np.float32)

with open('data/car_models_json/mazida-6-2015.json') as json_file:
    data = json.load(json_file)
vertices = np.array(data['vertices'])
vertices[:, 1] = -vertices[:, 1]
triangles = np.array(data['faces']) - 1

im_color = cv2.applyColorMap(np.arange(256).astype('uint8') , cv2.COLORMAP_HSV)[:,0,:]

def draw_obj(image, vertices, triangles, color):
    for t in triangles:
        coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)
        cv2.fillConvexPoly(image, coord, color)
        #cv2.polylines(image, np.int32([coord]), 1, color)
        
def draw_car(yaw, pitch, roll, x, y, z, overlay, color=(0,0,255)):
    yaw, pitch, roll = -pitch, -yaw, -roll
    Rt = np.eye(4)
    t = np.array([x, y, z])
    Rt[:3, 3] = t
    Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
    Rt = Rt[:3, :]
    P = np.ones((vertices.shape[0],vertices.shape[1]+1))
    P[:, :-1] = vertices
    P = P.T
    img_cor_points = np.dot(k, np.dot(Rt, P))
    img_cor_points = img_cor_points.T
    img_cor_points[:, 0] /= img_cor_points[:, 2]
    img_cor_points[:, 1] /= img_cor_points[:, 2]
    img_cor_points[:, 0] = (img_cor_points[:, 0])
    img_cor_points[:, 1] = (img_cor_points[:, 1]-1355)
    #img_cor_points[:, 0] = img_cor_points[:, 0]oimg_h/img_h
    draw_obj(overlay, img_cor_points, triangles, color)
    #print(img_cor_points)
    return overlay
def vizualize(hm, reg, img):
    #plt.subplot(14,2,i*2+1)
    #img = read_img(i).astype('uint8')
    #resize and transform img acc to params
    #For given xyz, we get xyi. And a specific overlay for one rotation.
    r = reg[:,:,:4]
    t = reg[:,:,4:]
    overlay = np.zeros((img_h, img_w, 3))
    #print(img.shape, overla)
    local_maxi = hm*(hm == maximum_filter(hm,footprint=np.ones((5,5,1))))>0.01
    y,x,z = np.where(hm==1)
    yaws, pitches, rolls, xs, ys, zs = r[y,x,0], np.arctan2(r[y,x,2],r[y,x,1]), r[y,x,3],\
                                        t[y,x,0]*100, t[y,x,1]*100, t[y,x,2]*100

    rolls = rolls-np.pi
    print('after ypr', len(yaws))
    """i=0
    print(yaws, pitches, rolls, xs, ys, zs)"""
    for yaw, pitch, roll, x, y, z in zip(yaws, pitches, rolls, xs, ys, zs):
    #for yaw, pitch, roll, x, y, z in zip([0, np.pi/2], [0,0], [np.pi, np.pi//2], [-3,3], [3,3], [8,20]):
        color = im_color[np.random.randint(256)].tolist()
        overlay = draw_car(yaw, pitch, roll, x, y, z, overlay, color)
        #
        
        #print('ypr:', i)
        #plt.imshow(overlay)

        #if i==5: break
    
    overlay = cv2.resize(overlay, (iimg_w, iimg_h))
    overlay = cv2.warpPerspective(overlay, M, (ip_w, ip_h))
    #img = (np.maximum(overlay, img)//2)
    
    img[overlay>0] = img[overlay>0]//2 + overlay[overlay>0]//2
    return img
        
def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))

hm = np.load('train_hms/ID_8a6e65317_1.npy')
reg = np.load('train_regs/ID_8a6e65317_1.npy')
img = np.load('train_inputs/ID_8a6e65317_1_n.npy')
plt.figure(figsize = (20,20))
plt.imshow(np.squeeze(hm))
plt.show()
print(hm.shape, reg.shape, img.shape)

