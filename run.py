import cv2
from process import Processor


def test_camera(index=0):
    processor = Processor()
    cap = cv2.VideoCapture(index)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        show = processor.RecogAndDraw(img)
        cv2.imshow('py_face_recognition', show)
        cv2.waitKey(1)


if __name__ == "__main__":
    test_camera()
