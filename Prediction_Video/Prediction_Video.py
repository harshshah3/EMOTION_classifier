import sys,cv2
import numpy as np

import CNN_Model.CNN as vc

windowsName = 'Preview Screen'

CASCADE_PATH = "C:/Users/Harsh/PycharmProjects/EMOTION_classifier/Prediction_Video/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(CASCADE_PATH)
emotion = ['Angry', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']
model = vc.CNN_16('C:/Users/Harsh/PycharmProjects/EMOTION_classifier/Model_Training/final_weights.h5')

capture = cv2.VideoCapture("C:/Users/Harsh/PycharmProjects/EMOTION_classifier/Vid2.mp4")
#capture = cv2.VideoCapture(0)

def grayFace(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    return img_gray

def getFaceCoordinates(image):
    img_gray = grayFace(image)
    rects = cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.3,
        minNeighbors=2,
        #scaleFactor=1.1,
        #minNeighbors=3,
        minSize=(48, 48)
        )
    return rects

def predict_emotion(gray_face):
    resized_img = cv2.resize(gray_face, (48,48), interpolation = cv2.INTER_AREA)
    image = resized_img.reshape(1, 1, 48, 48)
    results = model.predict(image, batch_size=1, verbose=1)
    return results

while True:
    flag, frame = capture.read()
    img_gray = grayFace(frame)
    rects = getFaceCoordinates(frame)
    for (x, y, w, h) in rects:
        face_image = img_gray[y:y+h,x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        results = predict_emotion(face_image)
        cv2.putText(frame, str(emotion[np.argmax(results)]), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        print (emotion[np.argmax(results)])
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()