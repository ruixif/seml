import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, initializers, models
import pickle
from random import shuffle

def make_model(input_shape, output_shape):
    input_matrix = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, (6,6), strides=(2,2))(input_matrix)
    conv1 = layers.Activation('relu')(conv1)
    conv2 = layers.Conv2D(128,(3,3), strides=(1,1))(conv1)
    conv2 = layers.Activation('relu')(conv2)
    conv_flat = layers.Flatten()(conv2)
    feature = layers.Dense(output_shape)(conv_flat)
    feature = layers.Activation('linear')(feature)
    return models.Model(inputs=input_matrix, outputs=feature)
    

class Trainer:
    def __init__(self, input_shape, output_shape, batchsize, lr=0.0001, pretrained=None):
        model = make_model(input_shape=input_shape, output_shape=output_shape)
        if pretrained is not None:
            model.load_weights(pretrained)
            
        self.model = model
        self.weights = model.weights
        self.inputs, = model.inputs
        self.outputs, = model.outputs
        self.label = tf.placeholder(tf.float32, shape=(None,output_shape))
        self.loss = tf.losses.mean_squared_error(labels=self.label, predictions=self.outputs)
        self.batch_loss = tf.reduce_mean(self.loss)
        self.train = tf.train.AdamOptimizer(lr).minimize(self.batch_loss)
        self.batchsize = batchsize
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    def fitmodel(self, X, y, maxepoch=1):
        datalist = []
        batchsize = self.batchsize
        #build training dataset
        for i, each in enumerate(zip(X,y)):
            datalist.append(each)
        
        #train and valid
        for epoch in range(maxepoch):
            shuffle(datalist)
            trainlist = datalist[0:6000]
            validlist = datalist[6000:]
            
            inner_max_cycle = int(len(trainlist)/batchsize)
            for i in range(inner_max_cycle):
                batch_data = np.array(trainlist[(i*batchsize):(i+1)*batchsize])
                this_X = np.array([each[0] for each in batch_data])[:,:,:,np.newaxis]
                this_y = np.array([each[1] for each in batch_data])
                
                #import pdb; pdb.set_trace()
                _, current_loss = self.sess.run([self.train, self.batch_loss], 
                                            feed_dict={self.inputs: this_X,
                                                       self.label: this_y,
                                                      })
            
            valid_max_cycle = int(len(validlist)/batchsize)
            
            valid_loss = 0
            for j in range(valid_max_cycle):
                valid_data = np.array(validlist[(j*batchsize):(j+1)*batchsize])
                this_X = np.array([each[0] for each in valid_data])[:,:,:,np.newaxis]
                this_y = np.array([each[1] for each in valid_data])
                this_loss, = self.sess.run([self.batch_loss,], feed_dict={self.inputs: this_X,
                                                             self.label: this_y,
                                                            })
                valid_loss += this_loss
            print(valid_loss)
            
            if epoch % 100 == 0:
                self.model.save_weights('good_weights.h5')

if __name__ == "__main__":
    coulombs = pickle.load(open("coulombs.pickle",'rb'))
    properties = pickle.load(open("properties.pickle",'rb'))
    var_lst = np.sqrt(np.var(properties, axis=0))
    mean_lst = np.mean(properties, axis=0)
    normalized_properties = (properties-mean_lst)/var_lst

    trainer = Trainer(input_shape=(23,23,1), output_shape=len(properties[0]), batchsize=32, pretrained='good_weights.h5')
    trainer.fitmodel(coulombs, normalized_properties, maxepoch=1000)




