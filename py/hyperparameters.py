"""
File này để lưu các tham số vào các biến cho chương trình : 
+ 1 enum cảm xúc, gắn cho mỗi cảm xúc một số nguyên từ 0-6
+ Số frame xét trong 1 chu kỳ thay đổi bộ lọc
+ Các file data dùng để train và test model
+ Các hằng số liên quan đến Cascade và OpenCV, đường dẫn đến camera (index = 0)
+ Kích thước đầu vào cho mô hình huấn luyện: 
    INPUT_DIM = 48 - đầu vào được resize về kích thước 48x48 pixels
    # Training hyperparameters
    BATCH_SIZE = 64 - Đây là số lượng mẫu dữ liệu được sử dụng trong mỗi lần cập nhật trọng số của mô hình
    NUM_EPOCHS = 50 - mô hình sẽ được huấn luyện qua toàn bộ tập dữ liệu 50 lần.
    LEARNING_RATE = 1e-4 - tham số để điều chỉnh trọng số trong mô hình
    MAX_DETECT_FAILURES = 10 - số lỗi liên tiếp khi xác định khuôn mặt để hệ thống reset lại
"""
import enum
class Emotion(enum.Enum): 
    angry = 0
    disgusted = 1
    fearful = 2
    happy = 3
    neutral = 4
    sad = 5
    surprised = 6
EMOTIONS = {Emotion.angry: "Angry",
            Emotion.disgusted: "Disgusted",
            Emotion.fearful: "Fearful", 
            Emotion.happy: "Happy", 
            Emotion.neutral: "Neutral", 
            Emotion.sad: "Sad", 
            Emotion.surprised: "Surprised"}

# The required number of frames to be processed before a filter is elected and votes are tallied.
VOTES_REQUIRED_FOR_ELECTION = 5

# These files are run from the /py directory
TRAINING_DIRECTORY = "data/train"
TESTING_DIRECTORY = "data/test"
MODEL_DATA_FILE_PATH = "data/model.h5"

# Cascade related files -- run from the ocv (open cv) directory
HAAR_FACE_FILE_PATH = "ocv/haarcascade_frontalface_alt.xml"
HAAR_EYE_FILE_PATH = "ocv/haarcascade_eye_tree_eyeglasses.xml"
HAAR_NOSE_FILE_PATH = "ocv/haarcascade_mcs_nose.xml"

# Default camera index - change if using external webcam
CAMERA_IDX = 0

# Research paper (in root directory) calls for all input to be 48x48
INPUT_DIM = 48

# Training hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

# Maximum number of failures before the caches reset
MAX_DETECT_FAILURES = 10