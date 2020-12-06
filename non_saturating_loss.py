import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import Loss

class Discriminator_loss(Loss):
  def __init__(self,names = 'Discriminator'):
    super(Discriminator_loss,self).__init__(name = names)
  
  def call(self,y_true,y_fake):
     y_true = tf.clip_by_value(y_true,1e-15,1. - 1e-15)
     y_fake = tf.clip_by_value(y_fake,1e-15,1. - 1e-15)
     loss = -tf.math.log(y_true) - tf.math.log( 1 - y_fake)
     loss = tf.reduce_mean(loss)
     return loss

class Generator_loss(Loss):
  def __init__(self,names = 'Generator'):
    super(Generator_loss,self).__init__(name = names)
  
  def call(self,y_fake):
    y_fake = tf.clip_by_value(y_fake,1e-15,1. - 1e-15)
    loss = -tf.math.log(y_fake)
    return loss