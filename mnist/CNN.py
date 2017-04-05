import tensorflow as tf
import numpy as np
import csv

NUM_PIXELS = 784
NUM_DIGITS = 10
NUM_ITERATIONS = 20
TRAIN_BATCH_SIZE = 50
MAX_PIXEL = 255
SIZE_IM = 28


#Defino las nuevas funciones...
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


#Cargo el conjunto de train y de test y obtenego el número de imágenes del train. .
train_data = np.loadtxt('./train.csv', dtype=np.float32, delimiter=',', skiprows=1)
num_im_train = len(train_data)
test_data = np.loadtxt('./test.csv', dtype=np.float32, delimiter=',', skiprows=1)

#Creo las variables simbólicas x e y_.
x = tf.placeholder(tf.float32, [None, NUM_PIXELS])  #Creo una variable simbólica. Ahora mismo no tiene un valor específico, es un placeholder. 28x28=784 pixeles/imagen.
y_ = tf.placeholder(tf.float32, [None, NUM_DIGITS]) #Creo un nuevo placeholder para la entropía.

#Primera capa convolucional.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,SIZE_IM,SIZE_IM,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Segunda capa convolucional.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Capa densamente conectada.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Para recudir Overfitting...
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout Layer.
W_fc2 = weight_variable([1024, NUM_DIGITS])
b_fc2 = bias_variable([NUM_DIGITS])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Entrenamiento y evaluación del modelo.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
num_im = num_im_train + 1
for i in range(NUM_ITERATIONS):                             #HAGO EL ENTRENAMIENTO!!!!
  if num_im > num_im_train:
    permutation = np.arange(num_im_train)                   #Obtengo los índices de las imagenes de train.
    np.random.shuffle(permutation)                          #Obtengo una permutación de las imagenes de train.
    train_data = train_data[permutation]                    #Permuto las imagenes del train conforme a la permutación de índices obtenida anteriormente.
    num_im = 0
  batch = train_data[num_im:num_im+TRAIN_BATCH_SIZE]        #En batch me quedo con las TRAIN_BATCH_SIZE primeras imágenes del train_data.
  if i%4 == 0:
    train_accuracy = accuracy.eval(feed_dict={
       x: batch[:, 1:] / MAX_PIXEL,
       y_: np.eye(NUM_DIGITS)[batch[:, 0].astype(int)],
       keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={
      x: batch[:, 1:] / MAX_PIXEL,
      y_: np.eye(NUM_DIGITS)[batch[:, 0].astype(int)],
      keep_prob: 0.5})
  num_im += TRAIN_BATCH_SIZE


  #Obtengo las predicciones del test
  test_results = train_step.run(feed_dict={
    x: test_data / MAX_PIXEL,
    keep_prob: 1.0
  })

  #Guardo las predicciones en un fichero .csv
  predicciones = np.hstack((np.arange(1, len(test_results) + 1).reshape((-1,1)), test_results.reshape((-1,1))))  #Creo la matriz con el número de imagen de test y su predicción.
  with open('./Pred/CNN20.csv', 'w', newline='') as softmax_csv:
    csv_writer = csv.writer(softmax_csv)
    csv_writer.writerow(['ImageId', 'Label'])
    csv_writer.writerows(predicciones)
