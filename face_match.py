from tensorflow.keras.models import load_model
from numpy import expand_dims
from numpy import asarray
import numpy as np
import imutils
import cv2
import time

model = load_model('facenet_keras.h5')


def load_img(image_path):
    image = cv2.imread(image_path)
    return image


def imshow(img):
    cv2.imshow("image", img)
    key = cv2.waitKey(0)


def detect_face(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #print("roi_color")
    return roi_color

def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

def extract_face(face_pixels):
    newTrainX = list()
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)
    return newTrainX


def result1(value):
    match_Result = False
    #value = process()
    if value < 110:
        match_Result = True
        #print(match_Result)
        return match_Result
    #print(value)
    #print(match_Result)
    return match_Result


def process(img1, img2):
    start = time.time()
    image1 = load_img(img1)
    image2 = load_img(img2)
    #bimshow(image1)

    roi_color1 = detect_face(image1)
    roi_color1 = cv2.resize(roi_color1,(160,160))
    roi_color2 = detect_face(image2)
    roi_color2 = cv2.resize(roi_color2,(160,160))


    #imshow(roi_color1)
    #imshow(roi_color2)

    features1 = extract_face(roi_color1)
    features2 = extract_face(roi_color2)

    result_value = np.sum(np.abs(features1 - features2))
    match_result = result1(result_value)
    return match_result, result_value




if __name__ == '__main__':
    start = time.time()
    print(process("ben1.jpg", "ben2.jpg"))
    end = time.time()
    #result()
    takes_time = end - start
    print(takes_time)