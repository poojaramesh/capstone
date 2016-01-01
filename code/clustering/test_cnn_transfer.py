import numpy as np
from keras.models import model_from_json
from data_prep import get_label, get_input, flow, create_image_list
from sklearn.metrics import precision_score, recall_score
from os.path import basename
import cPickle as pickle
from sklearn.metrics import confusion_matrix

def unpack_model(model_file, weights_file):
    model = model_from_json(open(model_file).read())
    model.load_weights(weights_file)
    return model

def get_names_count(image_list):
    image_names = [basename(path).split('.')[0] for path in image_list]
    labels = np.array([get_label(name) for name in image_list]) 
    count_0 = (labels == 0).sum() 
    count_1 = (labels == 1).sum() 
    count_2 = (labels == 2).sum() 
    return image_names, count_0, count_1, count_2

def get_test(path):
    image_list = create_image_list(path=path, use_mr=False)
    X, y = get_input(image_list)
    return X, y

def predict_prob(model, X, y):
    X = X.reshape(X.shape[0], 3, X.shape[1], X.shape[2])
    y_prob = model.predict_proba(X)
    return y_prob

def predict(model, X, y):
    X = X.reshape(X.shape[0], 3, X.shape[1], X.shape[2])
    y_predict = model.predict(X)
    return y_predict

def thresholding(y_prob, p_0, p_1, p_2):
    y_threshold = np.array([[1./p_0, 1./p_1, 1./p_2]])
    y_prob_thres = y_prob * (y_threshold)
    y_predicted = np.argmax(y_prob_thres, axis=1) 
    return y_predicted

def get_metrics(y_true, y_predicted):
    precision = precision_score(y_true, y_predicted, labels=[0,1,2], average="weighted")
    recall = recall_score(y_true, y_predicted, labels=[0,1,2], average="weighted")
    print "Precision Score {}".format(precision)
    print "Recall Score {}".format(recall)
    y_0 = y_predicted[y_true==0]
    y_1 = y_predicted[y_true==1]
    y_2 = y_predicted[y_true==2]
    print "Background: True = {}, Predicted = {}".format((y_true==0).sum(), (y_0==0).sum())
    print "Non_Mitosis: True = {}, Predicted = {}".format((y_true==1).sum(), (y_1==1).sum())
    print "Mitosis: True = {}, Predicted = {}".format((y_true==2).sum(), (y_2==2).sum())
    print confusion_matrix(y_true, y_predicted)

if __name__ == '__main__':
    print "Unpacking the model"
    model = unpack_model('../runs/gpu1/ccn1_model_architecture.json', '../runs/gpu1/ccn1_model_weights.h5')
    print "Unpickling the image list"
    with open ('../runs/gpu1/image_list.pkl') as f:
        image_list_train = pickle.load(f)
    print "Creating test image list"
    image_list_test = create_image_list(path='/data/Test/ScannerA', use_mr=False)
    print "Calculating distribution of train"
    image_names, count_0, count_1, count_2 = get_names_count(image_list_train)
    print "Creating X, y for test {}".format(len(image_list_test))
    X, y = get_input(image_list_test)
    print "Predicting on test"
    y_prob = predict_prob(model, X, y)
    y_predicted = thresholding(y_prob, count_0*1./y.size, count_1*1./y.size, count_2*1./y.size)
    get_metrics(y, y_predicted)
