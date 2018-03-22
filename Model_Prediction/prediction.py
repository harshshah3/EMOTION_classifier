# some_file.py
import sys
sys.path.append('C:/Users/Harsh/PycharmProjects/EMOTION_classifier_project/CNN_Model')
sys.path.append('C:/Users/Harsh/PycharmProjects/EMOTION_classifier_project/Model_Training')

import CNN_Model.CNN as vgg

import cv2
import numpy as np

def preprocessing(img, size=(48, 48)):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, size).astype(np.float32)
    return img

def main():
    model = vgg.CNN_16('C:/Users/Harsh/PycharmProjects/EMOTION_classifier_project/Model_Training/final_weights.h5')
    print ('Image Prediction Mode')
    img = preprocessing(cv2.imread('C:/Users/Harsh/PycharmProjects/EMOTION_classifier_project/girl2.jpg'))
    X = np.expand_dims(img, axis=0)
    X = np.expand_dims(X, axis=0)
    # predict_classes will enable us to select most probable class
    result = model.predict_classes(X)
    print(result)

    #emotion = [0->'Angry', 1->'Fear',2-> 'Happy',3-> 'Sad',4-> 'Surprise',5-> 'Neutral']

if __name__ == "__main__":
    main()
