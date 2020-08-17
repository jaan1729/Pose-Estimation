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
    print('sizes:  ',len(df), l)
    train = df[:l]
    np.random.shuffle(train)
    val = df[l:]
    val_size = len(df)-l
    train_gen = train_generator(train, batch_size)
    val_gen = train_generator(val, 4)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=1e-2,
                    decay_steps=5000,
                    decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule)               
    model.compile(optimizer=optimizer,
                   loss={'d1':l1_loss, 'h1':focal_loss})
    if load_wts == 1:

        model.load_weights(checkpoint_path)
    
    epoch_steps = l if batch_size ==1 else l//(batch_size//2)
    model.fit(train_gen, steps_per_epoch = epoch_steps , epochs = epochs, 
    verbose = 1, validation_data = val_gen, validation_steps=val_size//(4//2), callbacks = [cp_callback]) 
    #callbacks = [cp_callback]) l//(batch_size//2)  val_size//(batch_size//
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ep', default=20, type=int)
    parser.add_argument('--tr_split', default=90, type=int)
    parser.add_argument('--batch_size', default= 4, type=int)
    parser.add_argument('--wts', default=1, type=int)
    args, _ = parser.parse_known_args()
    model = get_model()
    
    train_model(model,args.ep,args.batch_size, args.tr_split, args.wts)
main()    
