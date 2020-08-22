import tensorflow as tf
from keras.models import load_model

def init():
    # load the pre-trained Keras model
    model = load_model('manual-lenet-avg-8230.h5')
    graph = tf.get_default_graph()

init()