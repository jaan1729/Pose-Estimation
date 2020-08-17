import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import get_file
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint



CTDET_COCO_WEIGHTS_PATH = (
    'https://github.com/see--/keras-centernet/'
    'releases/download/0.1.0/ctdet_coco_hg.hdf5')

HPDET_COCO_WEIGHTS_PATH = (
    'https://github.com/see--/keras-centernet/'
    'releases/download/0.1.0/hpdet_coco_hg.hdf5')

bn_train = False
bn_train2 = True

def hourglass_module(heads, bottom, cnv_dim, hgid, dims, input_ref):
    lfs = left_features(bottom, hgid, dims)
    rf1 = right_features(lfs, hgid, dims)
    rf1 = convolution(rf1, 3, cnv_dim, name='cnvs.%d' % hgid)
    heads = create_heads(heads, rf1, hgid, input_ref)
    return heads, rf1

def convolution(_x, k, out_dim, name, stride=1):
    
    padding = (k - 1) // 2
    _x = ZeroPadding2D(padding=padding, name=name + '.pad')(_x)
    _x = Conv2D(out_dim, k, strides=stride, use_bias=False, name=name + '.conv')(_x)
    if name[:6]=='cnvs.1' and bn_train2==True:
        #print(name + '.bn')
        _x = BatchNormalization(epsilon=1e-5, name=name + '.bn')(_x)
    else:
        _x = BatchNormalization(epsilon=1e-5, name=name + '.bn')(_x, training=bn_train)
    _x = Activation('relu', name=name + '.relu')(_x)
    return _x

def residual(_x, out_dim, name, stride=1):
    shortcut = _x
    num_channels = K.int_shape(shortcut)[-1]
    _x = ZeroPadding2D(padding=1, name=name + '.pad1')(_x)
    _x = Conv2D(out_dim, 3, strides=stride, use_bias=False, name=name + '.conv1')(_x)

    if (name[:9]=='kps.1.out' or  name[:10]=='kps.1.skip') and bn_train2==True:
        #print(name + '.bn1')
        _x = BatchNormalization(epsilon=1e-5, name=name + '.bn1')(_x)
    else:
        _x = BatchNormalization(epsilon=1e-5, name=name + '.bn1')(_x, training=bn_train)
    _x = Activation('relu', name=name + '.relu1')(_x)

    _x = Conv2D(out_dim, 3, padding='same', use_bias=False, name=name + '.conv2')(_x)
    if (name[:9]=='kps.1.out' or  name[:10]=='kps.1.skip') and bn_train2==True:
        #print(name + '.bn2')
        _x = BatchNormalization(epsilon=1e-5, name=name + '.bn2')(_x)
    else:
        _x = BatchNormalization(epsilon=1e-5, name=name + '.bn2')(_x, training=bn_train)

    if num_channels != out_dim or stride != 1:
        shortcut = Conv2D(out_dim, 1, strides=stride, use_bias=False, name=name + '.shortcut.0')(
            shortcut)
        shortcut = BatchNormalization(epsilon=1e-5, name=name + '.shortcut.bn1')(shortcut, training=bn_train)

    _x = Add(name=name + '.add')([_x, shortcut])
    _x = Activation('relu', name=name + '.relu')(_x)
    return _x

def pre(_x, num_channels):
    _x = convolution(_x, 7, 128, name='pre.0', stride=2)
    _x = residual(_x, num_channels, name='pre.1', stride=2)
    return _x

def left_features(bottom, hgid, dims):
    features = [bottom]
    for kk, nh in enumerate(dims):
        pow_str = ''
        for _ in range(kk):
            pow_str += '.center'
        _x = residual(features[-1], nh, name='kps.%d%s.down.0' % (hgid, pow_str), stride=2)
        _x = residual(_x, nh, name='kps.%d%s.down.1' % (hgid, pow_str))
        features.append(_x)
    return features

def connect_left_right(left, right, num_channels, num_channels_next, name):
    left = residual(left, num_channels_next, name=name + 'skip.0')
    left = residual(left, num_channels_next, name=name + 'skip.1')
    out = residual(right, num_channels, name=name + 'out.0')
    out = residual(out, num_channels_next, name=name + 'out.1')
    out = UpSampling2D(name=name + 'out.upsampleNN')(out)
    out = Add(name=name + 'out.add')([left, out])
    return out

def bottleneck_layer(_x, num_channels, hgid):
    pow_str = 'center.' * 5
    _x = residual(_x, num_channels, name='kps.%d.%s0' % (hgid, pow_str))
    _x = residual(_x, num_channels, name='kps.%d.%s1' % (hgid, pow_str))
    _x = residual(_x, num_channels, name='kps.%d.%s2' % (hgid, pow_str))
    _x = residual(_x, num_channels, name='kps.%d.%s3' % (hgid, pow_str))
    return _x

def right_features(leftfeatures, hgid, dims):
    rf = bottleneck_layer(leftfeatures[-1], dims[-1], hgid)
    for kk in reversed(range(len(dims))):
        pow_str = ''
        for _ in range(kk):
            pow_str += 'center.'
        rf = connect_left_right(leftfeatures[kk], rf, dims[kk], dims[max(kk - 1, 0)], name='kps.%d.%s' % (hgid, pow_str))
    return rf

def hm_stop_gradient(x):
    return K.stop_gradient(K.sigmoid(x))

def stop_gradient(x):
    return K.stop_gradient(x)

def create_heads(heads, rf1, hgid, input_ref):

    _heads = []
    keys = list(heads.keys())  

    if hgid >= 0:
        head = keys[0]
        num_channels = heads[head]
        _x = Conv2D(256, 3, use_bias=True, padding='same', name=head + '.%d.0.conv' % hgid)(rf1)
        _x = Activation('relu', name=head + '.%d.0.relu' % hgid)(_x)
        _x = Conv2D(num_channels, 1, use_bias=True, name=head[0] + '%d' % hgid)(_x)
        #_heads.append(_x)

    hm = Lambda(hm_stop_gradient, name='hm_ref.'+ '%d' % hgid)(_x)
    hm = Concatenate()([rf1, hm])

    if hgid >= 0:
        head = keys[1]
        num_channels = heads[head]
        _x = Conv2D(256, 3, use_bias=True, padding='same', name=head + '.%d.0.conv' % hgid)(hm)
        _x = Activation('relu', name=head + '.%d.0.relu' % hgid)(_x)
        _x = Conv2D(num_channels, 1, use_bias=True, name=head[0] + '%d' % hgid)(_x)
        _heads.append(_x) 

    """if hgid >= 0:
        head = keys[2]
        num_channels = heads[head]
        #_x = Lambda(stop_gradient, name=head + '.%d.stop_g' % hgid)(hm)
        _x = Conv2D(256, 3, use_bias=True, padding='same', name=head + '.%d.0.conv' % hgid)(hm)
        _x = Activation('relu', name=head + '.%d.0.relu' % hgid)(_x)
        _x = Conv2D(num_channels, 1, use_bias=True, name=head[0] + '%d' % hgid)(_x)
        _heads.append(_x)"""

    if hgid >= 0:
        head = keys[-1]
        num_channels = heads[head]
        head = '%d'%hgid + head

        ref = input_ref#Lambda(get_ref, name='coor_ref')(_heads[-1])
        ref = Conv2D(16, 1, use_bias=True, padding='same', name=head + '.ref.%d.0.conv' % hgid)(ref)
        #ref = BatchNormalization(epsilon=1e-5, name=head + '.ref.%d.0.bn' % hgid)(ref)
        ref = Activation('relu', name=head + '.ref.%d.0.relu' % hgid)(ref)
        ref = Conv2D(128, 1, use_bias=True, padding='same', name=head + '.ref.%d.1.conv' % hgid)(ref)
        #ref = BatchNormalization(epsilon=1e-5, name=head + '.ref.%d.1.bn' % hgid)(ref)
        ref = Activation('relu', name=head + '.ref.%d.1.relu' % hgid)(ref)

        _x = Concatenate()([hm, ref])

        for i in range(3):
            _x = Conv2D(256*(1+hgid), 3, use_bias=True, padding='same', name=head + '.%d.%d.conv' %(hgid,i))(_x)
            _x = Activation('relu', name=head + '.%d.%d.relu' %(hgid,i))(_x)
        _x = Conv2D(num_channels, 1, use_bias=True, name=head[1] + '%d' % hgid)(_x)
        _heads.append(_x)

    return _heads

def HourglassNetwork(heads, num_stacks=2, cnv_dim=256, inres=(512, 512), weights='ctdet_coco',
                        dims=[256, 384, 384, 384, 512]):

    if not (weights in {'ctdet_coco', 'hpdet_coco', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                            '`None` (random initialization), `ctdet_coco` '
                            '(pre-trained on COCO), `hpdet_coco` (pre-trained on COCO) '
                            'or the path to the weights file to be loaded.')
    input_ref = Input(shape=(inres[2], inres[3], 2), name='HGRef')
    input_layer = Input(shape=(inres[0], inres[1], 3), name='HGInput')
    inter = pre(input_layer, cnv_dim)
    prev_inter = None
    outputs = []
    for i in range(num_stacks):
        prev_inter = inter
        _heads, inter = hourglass_module(heads, inter, cnv_dim, i, dims, input_ref)
        if i == num_stacks - 1:
            outputs.extend(_heads)
        if i < num_stacks - 1:
            inter_ = Conv2D(cnv_dim, 1, use_bias=False, name='inter_.%d.0' % i)(prev_inter)
            inter_ = BatchNormalization(epsilon=1e-5, name='inter_.%d.bn1' % i)(inter_, training=bn_train)

            cnv_ = Conv2D(cnv_dim, 1, use_bias=False, name='cnv_.%d.0' % i)(inter)
            cnv_ = BatchNormalization(epsilon=1e-5, name='cnv_.%d.bn1' % i)(cnv_, training=bn_train)

            inter = Add(name='inters.%d.inters.add' % i)([inter_, cnv_])
            inter = Activation('relu', name='inters.%d.inters.relu' % i)(inter)
            inter = residual(inter, cnv_dim, 'inters.%d' % i)

    model = Model(inputs=[input_layer, input_ref], outputs=outputs)
    
    # I use pretrain when training
    if weights == 'ctdet_coco':
        print('loading ctdet coco')
        weights_path = get_file(
            '%s_hg.hdf5' % weights,
            CTDET_COCO_WEIGHTS_PATH,
            cache_subdir='models',
            file_hash='ce01e92f75b533e3ff8e396c76d55d97ff3ec27e99b1bdac1d7b0d6dcf5d90eb')
        model.load_weights(weights_path, by_name=True)
    elif weights == 'hpdet_coco':
        weights_path = get_file(
            '%s_hg.hdf5' % weights,
            HPDET_COCO_WEIGHTS_PATH,
            cache_subdir='models',
            file_hash='5c562ee22dc383080629dae975f269d62de3a41da6fd0c821085fbee183d555d')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model

def get_model(bn_train = False,bn_train2 = True):
    heads = {'classes': 34, 'hm_car': 1, 'dof_car': 8}
    model = HourglassNetwork(heads, num_stacks=2, inres=(None, None, None, None))
    
    train_layers = ['kps.1.out', 'kps.1.skip', 'cnvs.1']
    
    if bn_train2 == False:
        for layer in model.layers:
            layer.trainable = False
            if layer.name[:4]=='1dof':
                layer.trainable = True
            if layer.name=='d1':
                layer.trainable = True
            if layer.name[:8]=='hm_car.1':
                layer.trainable = True
            if layer.name[:3]=='hm1':
                layer.trainable = True
    
    elif bn_train == False:
        print('training deeply')
        for layer in model.layers:
            layer.trainable = False
    
        for tl in train_layers:
            if layer.name[:len(tl)] == tl :
                layer.trainable = True
    
            if layer.name[:4]=='1dof':
                layer.trainable = True
                      
            elif layer.name[:8]=='hm_car.1':
                layer.trainable = True        
            elif layer.name[:9]=='classes.1':
                layer.trainable = True  
            if layer.name=='c1':
                layer.trainable = True
            if layer.name=='d1':
                layer.trainable = True
            if layer.name[:3]=='h1':
                layer.trainable = True
    
            #if layer.trainable == True:
                #print(layer.name)
    
    return model
