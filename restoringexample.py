import tensorflow as tf

def main(_):
    w = tf.Variable(tf.zeros([5,5,1,32]), dtype=tf.float32,name = 'conv1_weights')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'cnn06b_gradient.ckpt.data-00000-of-00001')
        print('conv1_weights:',sess.run(w))
