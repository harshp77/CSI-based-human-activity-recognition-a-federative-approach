import tensorflow as tf
import numpy as np 
    

def modeler(win_len):
    input = tf.keras.layers.Input((win_len,90,1))
    x = tf.keras.layers.Conv2D(3, activation='relu', kernel_size=3, kernel_regularizer=tf.keras.regularizers.l1_l2())(input)
    x = tf.keras.layers.Dropout(0.3)
    x = tf.keras.layers.Conv2D(6, activation='relu', kernel_size=3, kernel_regularizer=tf.keras.regularizers.l1_l2())(input)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(7,activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs=input, outputs=out)
    # model.summary()
    # # train
    return model