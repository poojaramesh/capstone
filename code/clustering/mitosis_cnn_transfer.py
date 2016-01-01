'''
    Created 3 clusters based on background stain variation. Training 3 neural nets for each of the clusters
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from data_prep import get_input, flow, create_image_list, get_counts, create_image_clusters, get_image_clusters
from sklearn.cross_validation import train_test_split
from keras.models import model_from_json
import cPickle as pickle
import test_cnn

def train_model(feature_layers, classification_layers, image_list, nb_epoch, nb_classes, img_rows, img_cols, weights=None): 
    # Create testset data for cross-val
    num_images = len(image_list)
    test_size = int(0.2 * num_images)
    print("Train size: ", num_images-test_size)
    print("Test size: ", test_size)

    model = Sequential()
    for l in feature_layers + classification_layers:
        model.add(l)

    if not(weights is None):
        model.set_weights(weights)

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    print('Using real time data augmentation')
    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print('Training...')
        # batch train with realtime data augmentation
        progbar = generic_utils.Progbar(num_images-test_size)
        for X_batch, Y_batch in flow(image_list[0:-test_size]):
            X_batch = X_batch.reshape(X_batch.shape[0], 3, img_rows, img_cols)
            Y_batch = np_utils.to_categorical(Y_batch, nb_classes)
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[('train loss', loss)])

        print('Testing...')
        # test time!
        progbar = generic_utils.Progbar(test_size)
        for X_batch, Y_batch in flow(image_list[-test_size:]):
            X_batch = X_batch.reshape(X_batch.shape[0], 3, img_rows, img_cols)
            Y_batch = np_utils.to_categorical(Y_batch, nb_classes)
            score = model.test_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[('test loss', score)])
    return model, model.get_weights()



batch_size = 32
nb_classes = 3 
nb_epoch = 10
data_augmentation = True

# input image dimensions
img_rows, img_cols = 100,100
# the mitosis images are RGB
img_channels = 3


# the data, shuffled and split between train and test sets
image_list = create_image_list('/data/50_50_100_40', sample_size=[-1,-1,-1,-1,-1], use_mr=True)
image_names, clusters = get_image_clusters('/home/ubuntu/capstone/code/clustering/image_names.pkl','/home/ubuntu/capstone/code/clustering/clusters.pkl')

with open('image_list.pkl', 'w+') as f:
    pickle.dump(image_list, f)

# define two groups of layers: feature (convolutions) and classification (dense) 
feature_layers = [
    Convolution2D(32, 3, 3, border_mode='same',
                    input_shape=(img_channels, img_rows, img_cols)),
    Activation('relu'),
    Convolution2D(32, 3, 3),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1),
    Convolution2D(64, 3, 3, border_mode='same'),
    Activation('relu'),
    Convolution2D(64, 3, 3),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1)
]
classification_layers = [
    Flatten(),
    Dense(512),
    Activation('relu'),
    Dropout(0.5),
    Dense(nb_classes),
    Activation('softmax'),
]


# Train on cluster0
image_list0 = create_image_clusters(image_list, image_names, clusters, 0)
print("Distribution of classes (0,1,2): ", get_counts(image_list0))
model0, weights0 = train_model(feature_layers, classification_layers, image_list0, nb_epoch, nb_classes, img_rows, img_cols) 

# Train on cluster1
image_list1 = create_image_clusters(image_list, image_names, clusters, 1)
print("Distribution of classes (0,1,2): ", get_counts(image_list1))
model1, weights1 = train_model(feature_layers, classification_layers, image_list1, nb_epoch, nb_classes, img_rows, img_cols, weights0) 

# Train on cluster2
image_list2 = create_image_clusters(image_list, image_names, clusters, 2)
print("Distribution of classes (0,1,2): ", get_counts(image_list2))
model_final, weights2 = train_model(feature_layers, classification_layers, image_list2, nb_epoch, nb_classes, img_rows, img_cols, weight1) 

json_string = model_final.to_json()
open('cluster_model_architecture.json', 'w').write(json_string)
model_final.save_weights('cluster_model_weights.h5')

image_list_test = create_image_list(path='/data/Test/ScannerA', use_mr=False)
print("Calculating distribution of train")
image_names, count_0, count_1, count_2 = test_cnn.get_names_count(image_list)
print("Creating X, y for test {}".format(len(image_list_test)))
X, y = get_input(image_list_test)
print("Predicting on test")
y_prob = test_cnn.predict_prob(model_final, X, y)
y_predicted = test_cnn.thresholding(y_prob, count_0*1./y.size, count_1*1./y.size, count_2*1./y.size)
test_cnn.get_metrics(y, y_predicted)
