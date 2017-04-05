import tensorflow as tf
import numpy as np

def main(_):

    #Variables PlaceHolder para las variables que se cargan.
    #W = tf.Variable(tf.zeros([1,3],dtype=tf.float32,name='weights'))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,fileName) #! fileName is not defined.
        #ahora podemos acceder a los valores desde los placeholders. 
    

    
