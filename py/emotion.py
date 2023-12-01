from __future__ import print_function
from hyperparameters import Emotion, TRAINING_DIRECTORY, TESTING_DIRECTORY, MODEL_DATA_FILE_PATH, EMOTIONS, INPUT_DIM, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
import cv2 as cv
import numpy as np
from hyperparameters import HAAR_EYE_FILE_PATH, HAAR_FACE_FILE_PATH, HAAR_NOSE_FILE_PATH, CAMERA_IDX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import argparse

def classify_emotion(frame_gray, face, model):
    """
    Phân loại một khuôn mặt vào một trong bảy danh mục cảm xúc (xem trong hyperparameters.py) dựa trên các đặc trưng của khuôn mặt.

    Input:
        frame_gray: Là một mảng biểu diễn khung hình camera sau khi được áp dụng bộ lọc xám.

        face: Là một tuple chứa tọa độ (x, y) của góc trên bên trái của hình chữ nhật giới hạn khuôn mặt và chiều rộng và chiều cao của nó (w, h).

        model: Là mô hình TensorFlow tuần tự đã được huấn luyện để phân loại cảm xúc.
    Process:
        Slicing: Chọn ra phần hình ảnh tương ứng với khuôn mặt từ khung hình camera sử dụng thông tin vị trí và kích thước của khuôn mặt.
        Resizing: Đưa hình ảnh khuôn mặt về kích thước chuẩn (48x48 pixels) và thêm một chiều để đảm bảo định dạng đúng cho dự đoán mô hình.
        Dự đoán: Sử dụng mô hình để dự đoán xác suất của từng lớp cảm xúc thông qua hàm model.predict.
    Output: 
        Trả về lớp cảm xúc có xác suất cao nhất dưới dạng Enum Emotion.
    """
    # slice out the face region
    (x, y, w, h) = face
    face_gray_from_frame = frame_gray[y:y+h,x:x+w]

    #paper called for standard 48x48 sizing of any input 
    resized = np.expand_dims(np.expand_dims(cv.resize(face_gray_from_frame, (INPUT_DIM, INPUT_DIM)), -1), 0)

    # Predict & return classification
    prediction = model.predict(resized)
    return Emotion(np.argmax(prediction))


def get_num_images_in_dir(directory):
    """
    hàm này được sử dụng để đếm số lượng tệp hình ảnh trong một thư mục, bao gồm cả trong các thư mục con của các loại cảm xúc. 
    nó giúp xác định số lượng ảnh sẽ được sử dụng trong quá trình huấn luyện hoặc kiểm thử mô hình.
    Input:
        directory :: đường dẫn tới thư mục chứa data
    Output:
        số tệp ảnh có trong thư mục
    """
    # Get list of all images in training directory
    categories = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
    num_files = 0
    for root, _, files in os.walk(directory, categories):
        for name in files:
            num_files += 1
    return num_files

def train_or_load_model(mode):
    """
    num_training và num_testing: Đếm số lượng hình ảnh trong thư mục train và thư mục test

    Tạo mô hình neural network sử dụng TensorFlow và Keras. Mô hình này có cấu trúc CNN (Convolutional Neural Network) 
    với các lớp Conv2D, MaxPooling2D, Dropout, Flatten và Dense.
    Kiểm tra mode để quyết định liệu có huấn luyện mô hình mới hay tải mô hình đã lưu trước đó.
    + mode == "train": Tiến hành huấn luyện mô hình bằng cách sử dụng dữ liệu từ thư mục huấn luyện
    và kiểm thử. Kết quả của mô hình sẽ được lưu vào tệp tin MODEL_DATA_FILE_PATH.
    + mode == "display" hoặc mode không hợp lệ: Tải mô hình từ tệp tin MODEL_DATA_FILE_PATH.
    Trả về mô hình đã huấn luyện hoặc mô hình đã tải.
    """
    # Tally the number of images in both directories 
    num_training = get_num_images_in_dir(TRAINING_DIRECTORY)
    num_testing = get_num_images_in_dir(TESTING_DIRECTORY)

    # Create the model
    #2 convolutional layers followed by max pooling & dropout, then 2x(C2D -> MaxPooling)
    #then another dropout, followed by flatten, dense (relu), dropout, dense (softmax)
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(INPUT_DIM,INPUT_DIM,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    # switch functionality based on passed in mode
    if mode == "train":
        # Training / testing data generation
        training = ImageDataGenerator(rescale=1./255)
        train_generator = training.flow_from_directory(
            TRAINING_DIRECTORY,
            target_size=(INPUT_DIM,INPUT_DIM),
            batch_size=BATCH_SIZE,
            color_mode="grayscale",
            class_mode='categorical')

        testing = ImageDataGenerator(rescale=1./255)
        validation_generator = testing.flow_from_directory(
            TESTING_DIRECTORY,
            target_size=(INPUT_DIM,INPUT_DIM),
            batch_size=BATCH_SIZE,
            color_mode="grayscale",
            class_mode='categorical')

        # Compile our model & save results in "../data/model.h5"
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=LEARNING_RATE),metrics=['accuracy'])
        model_info = model.fit_generator(
                train_generator,
                steps_per_epoch=num_training // BATCH_SIZE,
                epochs=NUM_EPOCHS,
                validation_data=validation_generator,
                validation_steps=num_testing // BATCH_SIZE)
        model.save_weights(MODEL_DATA_FILE_PATH)
    # display live, so simply load the weights and return
    elif mode == "display":
        model.load_weights(MODEL_DATA_FILE_PATH)
        #model = load_model(MODEL_DATA_FILE_PATH) # UNCOMMENT IF YOU WANT TO USE THE ALT MODEL
    else:
        print("ERROR: Invalid mode please enter one of [train, display]")
        return None

    return model


# def parse_input_and_load():
#     """
#     + Sử dụng thư viện argparse để đọc các tham số dòng lệnh, bao gồm đường dẫn đến các tệp cascade cho khuôn mặt, mắt, mũi,
#     chế độ của mô hình (train hoặc display), và thiết bị camera.
#     + Load các tệp cascade cho khuôn mặt, mắt và mũi bằng cách sử dụng cv.CascadeClassifier. 
#     Nếu có lỗi trong quá trình tải, chương trình sẽ xuất thông báo lỗi và kết thúc.
#     + Trả về một tuple chứa thông tin về mode, cascade cho khuôn mặt, cascade cho mắt, cascade cho mũi và thiết bị camera.
#     """
#     #-- argument setup
#     parser = argparse.ArgumentParser(description='Code for Emotion Based Filtering.')
#     parser.add_argument('--face_cascade', help='Path to face cascade.', default=HAAR_FACE_FILE_PATH)
#     parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default=HAAR_EYE_FILE_PATH)
#     parser.add_argument('--nose_cascade', help='Path to eyes cascade.', default=HAAR_NOSE_FILE_PATH)
#     parser.add_argument("--model_mode",help='Whether or not we want to [train] the model, or [display] results.', default='display')
#     parser.add_argument('--camera', help='Camera divide number.', type=int, default=CAMERA_IDX)

#     #-- store each input to be returned
#     args = parser.parse_args()
#     mode = args.model_mode
#     face_cascade = cv.CascadeClassifier()
#     eyes_cascade = cv.CascadeClassifier()
#     nose_cascade = cv.CascadeClassifier()
#     camera_device = args.camera

#     #-- load the cascades
#     if not face_cascade.load(cv.samples.findFile(args.face_cascade)):
#         print('--(!)Error loading face cascade')
#         exit(0)
#     if not eyes_cascade.load(cv.samples.findFile(args.eyes_cascade)):
#         print('--(!)Error loading eyes cascade')
#         exit(0)
#     if not nose_cascade.load(cv.samples.findFile(args.nose_cascade)):
#         print('--(!)Error loading nose cascade')
#         exit(0)
    
#     return (mode, face_cascade, eyes_cascade, nose_cascade, camera_device)

# def test(frame, face_cascade, eyes_cascade, nose_cascade, model):
#     cv.imshow("Image",frame)
#     cv.waitKey(33)
#     sequential_failures = 0
#     face_cache = []
#     nose_cache = []
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     frame_gray = cv.equalizeHist(frame_gray)

#     # xác định phạm vi khuôn mặt
#     face_radius = 100

#     eye_list = np.zeros((2,2), dtype=int)
#     has_face = True
#     faces = face_cascade.detectMultiScale(image=frame_gray, flags=cv.CASCADE_FIND_BIGGEST_OBJECT, minSize=(face_radius, face_radius))

#         # Tính toán và cập nhật thông tin về vị trí khuôn mặt, cache cho việc
#         # theo dõi sự thay đổi trong khuôn mặt. Việc lưu trữ tạm thời giúp tránh việc
#         # truy xuất dữ liệu lại từ đầu mỗi khi cần sử dụng
#     if len(faces) > 0:
#         sequential_failures = 0
#         (fx, fy, fw, fh) = faces[0]
#         if len(face_cache) > 0: # nếu tồn tại khuôn mặt trước đó
#             (old_x, old_y, old_w, old_h) = face_cache
#             # nếu khuôn mặt mới nằm ngoài bounding box, reset
#             lower_x = old_x - (old_w)
#             upper_x = old_x + (old_w)
#             lower_y = old_y - (old_h)
#             upper_y = old_y + (old_h)
#             if (fx <= lower_x or fx >= upper_x or
#                 fy <= lower_y or fy >= upper_y):
#                 nose_cache = []
#         # Update face cache
#         face_cache = faces[0]
#     else:
#         sequential_failures += 1
#     # FACE CACHING

#         # Nếu không nhận diện được khuôn mặt quá nhiều lần liên tiếp (MAX_DETECT_FAILURES), thực hiện reset cache
#         # để tránh việc lưu trữ thông tin không chính xác về mặt và mũi
#     if sequential_failures >= 10:
#         face_cache = []
#         nose_cache = []

#     if len(face_cache) > 0: # Nếu có ít nhất một khuôn mặt được nhận diện, thực hiện các bước tiếp theo, bao gồm việc xác định vị trí của mắt và mũi trong khuôn mặt đã nhận diện.  
#         (x, y, w, h) = face_cache
#             # Classify the emotion and vote in the current filter election
#             # Nhận dạng cảm xúc và vote để apply filter lên mặt
#         emotion = classify_emotion(frame_gray, face_cache, model)
#         text = f"Emotion: {emotion.name}"
#         cv.putText(image, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#     cv.imshow("Emotion:",frame)
# (mode, face_cascade, eyes_cascade, nose_cascade, camera_device) = parse_input_and_load()
# model = train_or_load_model(mode)
# cap = cv.VideoCapture(camera_device)
# image = cv.imread(r"D:/KhanhLyTa/Li/li.jpg")
# image = cv.resize(image,(600,800))
# test(image,face_cascade,eyes_cascade,nose_cascade,model)
# cv.waitKey(0)

