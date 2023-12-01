from __future__ import print_function
import argparse
import cv2 as cv
from emotion import classify_emotion
from hyperparameters import HAAR_EYE_FILE_PATH, HAAR_FACE_FILE_PATH, HAAR_NOSE_FILE_PATH, CAMERA_IDX
from emotion import train_or_load_model
from filter import Filter

def parse_input_and_load():
    """
    + Sử dụng thư viện argparse để đọc các tham số dòng lệnh, bao gồm đường dẫn đến các tệp cascade cho khuôn mặt, mắt, mũi,
    chế độ của mô hình (train hoặc display), và thiết bị camera.
    + Load các tệp cascade cho khuôn mặt, mắt và mũi bằng cách sử dụng cv.CascadeClassifier. 
    Nếu có lỗi trong quá trình tải, chương trình sẽ xuất thông báo lỗi và kết thúc.
    + Trả về một tuple chứa thông tin về mode, cascade cho khuôn mặt, cascade cho mắt, cascade cho mũi và thiết bị camera.
    """
    #-- argument setup
    parser = argparse.ArgumentParser(description='Code for Emotion Based Filtering.')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default=HAAR_FACE_FILE_PATH)
    parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default=HAAR_EYE_FILE_PATH)
    parser.add_argument('--nose_cascade', help='Path to eyes cascade.', default=HAAR_NOSE_FILE_PATH)
    parser.add_argument("--model_mode",help='Whether or not we want to [train] the model, or [display] results.', default='display')
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=CAMERA_IDX)

    #-- store each input to be returned
    args = parser.parse_args()
    mode = args.model_mode
    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()
    nose_cascade = cv.CascadeClassifier()
    camera_device = args.camera

    #-- load the cascades
    if not face_cascade.load(cv.samples.findFile(args.face_cascade)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv.samples.findFile(args.eyes_cascade)):
        print('--(!)Error loading eyes cascade')
        exit(0)
    if not nose_cascade.load(cv.samples.findFile(args.nose_cascade)):
        print('--(!)Error loading nose cascade')
        exit(0)
    
    return (mode, face_cascade, eyes_cascade, nose_cascade, camera_device)

# def detectFace(frame, face_cascade, eyes_cascade, nose_cascade, model):
#     face_radius = 100

#     faces = face_cascade.detectMultiScale(image=frame, flags=cv.CASCADE_FIND_BIGGEST_OBJECT, minSize=(face_radius, face_radius))
#     # Loop through detected faces
#     for (x, y, w, h) in faces:
#         # Draw a rectangle around the face
#         cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#         # Extract the face region
#         face_roi = frame[y:y+h, x:x+w]
#         print("Face ROI size:", face_roi.shape)

#         # Detect eyes in the face region
#         eyes = eyes_cascade.detectMultiScale(face_roi)
#         for (ex, ey, ew, eh) in eyes:
#             cv.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

#         # Detect nose in the face region
#         nose = nose_cascade.detectMultiScale(face_roi)
#         for (nx, ny, nw, nh) in nose:
#             cv.rectangle(frame, (x+nx, y+ny), (x+nx+nw, y+ny+nh), (0, 0, 255), 2)
#     return frame

def main():
    """ 
    + lấy thông tin từ tham số dòng lệnh và tải các cascade.
    + gọi hàm train_or_load_model để tải hoặc huấn luyện mô hình dựa vào mode được chọn.

    + bắt đầu vòng lặp chính để đọc video từ camera. Trong mỗi lần lặp, chương trình sẽ đọc một frame từ video
    và thực hiện các bước nhận diện khuôn mặt, xác định cảm xúc, và áp dụng bộ lọc tương ứng.
    + kết thúc chương trình bằng cách ấn phím "l"
    """
    (mode, face_cascade, eyes_cascade, nose_cascade, camera_device) = parse_input_and_load()
    model = train_or_load_model(mode)
    cap = cv.VideoCapture(camera_device)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    my_filter = Filter()
    image = cv.imread(r"D:/KhanhLyTa/Li/li.jpg")
    image = cv.resize(image,(600,800))
    cv.imshow("Image:",image)
    cv.waitKey(33)
    my_filter.detect_and_display(image, face_cascade, eyes_cascade, nose_cascade, model)
    cv.waitKey(0)
    # while True:
    #     ret, frame = cap.read()
    #     if frame is None:
    #         print('--(!) No captured frame -- Break!')
    #         break
    #     my_filter.detect_and_display(frame, face_cascade, eyes_cascade, nose_cascade, model)
    #     if cv.waitKey(1) == ord('l'):
    #         cv.VideoCapture(camera_device).release()
    #         break

if __name__ == "__main__":
    main()
