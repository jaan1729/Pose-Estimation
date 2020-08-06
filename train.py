import argparse
import numpy as np
import tensorflow as tf
from Hourglass import get_model
from DataGen import train_generator
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import os
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_path = F"latest_model/cp.ckpt" 
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


def train_model(model, epochs, batch_size, train_split, load_wts):
    
    def focal_loss(hm_true, hm_pred):
        #hm_pred = tf.squeeze(hm_pred)
        #hm_pred = tf.math.sigmoid(hm_pred)
        pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
        neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
        neg_weights = tf.pow(1 - hm_true, 4)

        pos_loss = -tf.math.log(tf.clip_by_value(hm_pred, 1e-10, 1)) * tf.pow(1 - hm_pred, 2) * pos_mask
        neg_loss = -tf.math.log(tf.clip_by_value(1 - hm_pred, 1e-10, 1)) * tf.pow(hm_pred, 2) * neg_weights * neg_mask

        num_pos = tf.reduce_sum(pos_mask)
        pos_loss = tf.reduce_sum(pos_loss)
        neg_loss = tf.reduce_sum(neg_loss)

        cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
        return cls_loss
    
    def l1_loss(y_true, y_pred):
        mask = tf.zeros_like(y_true, dtype=tf.float32)
        mask = tf.equal(y_true, mask)
        mask = tf.cast(mask, tf.float32)
        mask = tf.reduce_sum(mask, axis=-1)

        one = tf.ones_like(mask)
        zero = tf.zeros_like(mask)
        mask = tf.where(mask == 7, x=zero, y=one)
        mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 1, 8))

        total_loss = tf.reduce_sum(tf.abs(y_true - y_pred * mask))
        reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)
        
        return reg_loss

    df = np.load('image_names.npy')
    l = len(df)*train_split//100
    train = df[:l]
    val = df[l:]
    train_gen = train_generator(train, batch_size)
    val_gen = train_generator(val, batch_size)
    model.compile(optimizer=Adam(),
                   loss={'d1':l1_loss, 'h1':focal_loss})
    if load_wts == 1:
        model.load_wts(checkpoint_path)
    model.fit(train_gen, steps_per_epoch = 17000//4, epochs = epochs, callbacks = [cp_callback])
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ep', default=10, type=int)
    parser.add_argument('--tr_split', default=90, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--wts', default=0, type=int)

    model = get_model()
    
    train_model(model, ep, batch_size, tr_split, wts)
    