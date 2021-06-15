
import numpy as np
import tensorflow as tf
!pip install tensorflow_gan #install tensorflow_gans
import tensorflow_gan as tfgan
#import fid eval form tensorflow gan
from tensorflow.python.ops import array_ops

#already ready code from source https://github.com/tsc2017/Frechet-Inception-Distance/blob/master/fid.py

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1' #source of our inception model
INCEPTION_FINAL_POOL = 'pool_3' #final pool_layer of inception of model
BATCH_SIZE = 64 #bacth size

#for tpu
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
strategy = tf.distribute.TPUStrategy(resolver)

@tf.function
def inception_activations(images,num_splits = 1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.compat.v1.image.resize_bilinear(images, [size, size])
    generated_images_list = tf.split(images, num_or_size_splits = num_splits)
    activations = tf.map_fn(
        fn = tfgan.eval.classifier_fn_from_tfhub(INCEPTION_TFHUB, INCEPTION_FINAL_POOL, True),
        elems = tf.stack(generated_images_list),
        parallel_iterations = 1,
        back_prop = False,
        swap_memory = True,
        name = 'RunClassifier')
    activations = tf.concat(tf.unstack(activations), 0)
    return activations

def get_inception_activations(inps):
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    act = np.zeros([inps.shape[0], 2048], dtype = np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] / 255. * 2 - 1
        act[i * BATCH_SIZE : i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])] = inception_activations(inp)
    return act


#download dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

#convert the data into float32 type
x_train = tf.cast(x_train,dtype=tf.float32)
x_test = tf.cast(x_test,dtype=tf.float32)

activations1 = get_inception_activations(tf.transpose(x_train[:1024],[0,3,1,2]))
activations2 = get_inception_activations(tf.transpose(x_test[:1024],[0,3,1,2]))

#caluclate the fina fid score 
fcd = tfgan.eval.frechet_classifier_distance_from_activations(activations1, activations2)
print(fcd)