
from tflearn.layers.core import input_data, fully_connected,flatten
from tflearn.layers.estimator import regression
import tflearn
from tflearn.layers import dropout,conv_2d,max_pool_2d


def conv(classes,input_shape):
    model = input_data(input_shape,name="input")
    model = conv_2d(model,32,(3,3),activation='relu')
    model = conv_2d(model,64,(3,3),activation='relu')
    model = max_pool_2d(model,(2,2))
    model = dropout(model,0.25)
    model = flatten(model)
    model = fully_connected(model,128,activation='relu')
    model = dropout(model,0.5)
    model = fully_connected(model,classes,activation='softmax')
    model = regression(model, optimizer='adam', learning_rate=0.001,
                       loss='categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(model, tensorboard_verbose=3)
    return model
