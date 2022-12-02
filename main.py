import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.filters import img_to_array
cap = cv2.VideoCapture(0)
model = load_model('explo_model.h5')
bg=None
aweight = 0.5
def run_avg(img,aweight):
    global bg
    if bg is None:
        bg = img.copy().astype('float')
        return
    cv2.accumulateWeighted(img,bg,aweight)
def get_prediction(img):
    for_pred = cv2.resize(img,(64,64))
    x = img_to_array(for_pred)
    x = x/255.0
    x = x.reshape((1,) + x.shape)
    pred = np.argmax(model.predict(x))+1
    return pred
def segment(img,thres=10):
    global bg
    if bg is None:
        bg = img.copy().astype('float')
    diff = cv2.absdiff(bg.astype('uint8'),img)
    _, thresholded = cv2.threshold(diff,thres,255,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return
    else:
        segmented = max(contours,key = cv2.contourArea)
    return (thresholded,segmented)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret ==True:
        
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[100:300, 300:500]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        # run_avg(gray,aweight)
        hand = segment(gray)
        if hand is not None:
            (thresholded, segmented) = hand
            # cv2.drawContours(clone, [segmented + (300, 100)], -1, (0, 0, 255))
            cv2.imshow("Thesholded", thresholded)
            cv2.imshow("clone",clone)
            contours, _= cv2.findContours(thresholded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 5000:
                    print("Hand detecting for prediction")
                    pred = get_prediction(thresholded)
                    cv2.putText(clone, str(pred), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 30, 255), 3)

                    print(pred)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.waitKey()
cv2.destroyAllWindows()