from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from data_prep import get_input, flow, create_image_list, get_counts
from sklearn.cross_validation import train_test_split
from keras.models import model_from_json
import cPickle as pickle

batch_size = 32
nb_classes = 2 
nb_epoch = 10
data_augmentation = True

# input image dimensions
img_rows, img_cols = 100,100
# the mitosis images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
image_list = create_image_list('/data/50_50_100_40', sample_size=[-1,-1,0,-1,-1], use_mr=True)

with open('image_list.pkl', 'w+') as f:
    pickle.dump(image_list, f)

# Create testset data for cross-val
num_images = len(image_list)
test_size = int(0.1 * num_images)
print("Train size: ", num_images-test_size)
print("Test size: ", test_size)
print("Training Distribution: ", get_counts(image_list[0:-test_size], nb_classes))

model = Sequential()

model.add(Convolution2D(256, 6, 6, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(256, 6, 6))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.1))

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


if not data_augmentation:
    print('Not using data augmentation or normalization')
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
    score = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('Test score:', score)

else:
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

    json_string = model.to_json()
    open('cnn1_model_architecture.json', 'w').write(json_string)
    model.save_weights('cnn1_model_weights.h5')

