from __future__ import print_function
import hyperparameters as hp
from hyperparameters import Emotion, EMOTIONS, INPUT_DIM, VOTES_REQUIRED_FOR_ELECTION, MAX_DETECT_FAILURES
from emotion import classify_emotion
import cv2 as cv
import math
import numpy as np
import operator

class Filter:
    """
    Tất cả các logic có liên quan để lọc và duy trì trạng thái liên quan đến việc xem xét thay đổi
    """
    def __init__(self):
        """
        Tạo các cuộc bầu cử sau mỗi chu kỳ, tức là một nhóm cảm xúc phải thể hiện đủ lâu thì bộ lọc mới được áp dụng
        """
        self.reset_election()
        """
        Load ảnh của tai và mũi từ file và lưu vào các biến như elected_left_ear, happy_left_img,...
        Các biến nose_cache, face_cache, sequential_failures được sử dụng để theo dõi trạng thái của việc nhận diện mũi và khuôn mặt.
        """
        self.elected_left_ear = cv.imread("img/ear_neutral_left.png", -1)
        self.elected_right_ear =cv.imread("img/ear_neutral_right.png", -1)
        self.happy_left_img = cv.imread("img/ear_happy_left.png", -1)
        self.happy_right_img = cv.imread("img/ear_happy_right.png", -1)
        self.sad_left_img = cv.imread("img/ear_sad_left.png", -1)
        self.sad_right_img = cv.imread("img/ear_sad_right.png", -1)
        self.neutral_left_img = cv.imread("img/ear_neutral_left.png", -1)
        self.neutral_right_img = cv.imread("img/ear_neutral_right.png", -1)
        self.nose = cv.imread("img/nose.png", -1)

        self.nose_cache = []
        self.face_cache = []
        # how many times we've failed in a row to detect a face
        self.sequential_failures = 0

    def detect_and_display(self, frame, face_cascade, eyes_cascade, nose_cascade, model):
        """
        Phương thức này thực hiện việc nhận diện khuôn mặt và hiển thị các hiệu ứng dựa trên cảm xúc
        được phân loại từ mô hình: 
        + trước tiên ta xác định phạm vi khuôn mặt, sau đó nhận diện khuôn mặt bằng
        face_cascade (chỉ quan tâm đến khuôn mặt lớn nhất trong khung hình, các khuôn mặt sẽ được xem xét nếu
        có kích thước >= minsize).
        +nếu có ít nhất một khuôn mặt được nhận diện, thực hiện các bước tiếp theo, bao gồm việc xác định vị trí của mắt và mũi trong khuôn mặt đã nhận diện.
        Sử dụng các hàm của OpenCV để nhận diện khuôn mặt, mắt, và mũi trong khung hình từ webcam.
        Dựa vào cảm xúc được phân loại từ mô hình, thực hiện việc thay đổi giữa các hiệu ứng hình ảnh.
        """
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)

        # xác định phạm vi khuôn mặt
        face_radius = 100

        eye_list = np.zeros((2,2), dtype=int)
        has_face = True
        faces = face_cascade.detectMultiScale(image=frame_gray, flags=cv.CASCADE_FIND_BIGGEST_OBJECT, minSize=(face_radius, face_radius))

        # Tính toán và cập nhật thông tin về vị trí khuôn mặt, cache cho việc
        # theo dõi sự thay đổi trong khuôn mặt. Việc lưu trữ tạm thời giúp tránh việc
        # truy xuất dữ liệu lại từ đầu mỗi khi cần sử dụng
        if len(faces) > 0:
            self.sequential_failures = 0
            (fx, fy, fw, fh) = faces[0]
            if len(self.face_cache) > 0: # nếu tồn tại khuôn mặt trước đó
                (old_x, old_y, old_w, old_h) = self.face_cache
                # nếu khuôn mặt mới nằm ngoài bounding box, reset
                lower_x = old_x - (old_w)
                upper_x = old_x + (old_w)
                lower_y = old_y - (old_h)
                upper_y = old_y + (old_h)
                if (fx <= lower_x or fx >= upper_x or
                    fy <= lower_y or fy >= upper_y):
                    self.nose_cache = []
            # Update face cache
            self.face_cache = faces[0]
        else:
            self.sequential_failures += 1
        # FACE CACHING

        # Nếu không nhận diện được khuôn mặt quá nhiều lần liên tiếp (MAX_DETECT_FAILURES), thực hiện reset cache
        # để tránh việc lưu trữ thông tin không chính xác về mặt và mũi
        if self.sequential_failures >= MAX_DETECT_FAILURES:
            self.face_cache = []
            self.nose_cache = []

        if len(self.face_cache) > 0: # Nếu có ít nhất một khuôn mặt được nhận diện, thực hiện các bước tiếp theo, bao gồm việc xác định vị trí của mắt và mũi trong khuôn mặt đã nhận diện.  
            (x, y, w, h) = self.face_cache
            # Classify the emotion and vote in the current filter election
            # Nhận dạng cảm xúc và vote để apply filter lên mặt
            emotion = classify_emotion(frame_gray, self.face_cache, model)
            self.process_filter_vote(emotion)
            center = (x + w//2, y + h//2)        
            x_offset2=center[0]
            y_offset2=center[1]
            radius = int(round((w + h)*0.25))
            eye_list[0] = (center[0] - radius/2, center[1]-radius/2)
            eye_list[1] = (center[0] + radius/2, center[1]-radius/2)
            if (w < face_radius) or(h < face_radius):
                has_face = False

            faceROI = frame_gray[y:y+h, x:x+w]
            
            nose_coords = nose_cascade.detectMultiScale(image=faceROI, flags=cv.CASCADE_FIND_BIGGEST_OBJECT, minSize=(50, 50))
            if len(nose_coords) > 0:
                (nx, ny, nw, nh) = nose_coords[0]
                if len(self.nose_cache) > 0:
                    (old_x, old_y, old_w, old_h) = self.nose_cache
                    lower_x = old_x - (old_w)
                    upper_x = old_x + (old_w)
                    lower_y = old_y - (old_h)
                    upper_y = old_y + (old_h)
                    if (nx >= lower_x and nx <= upper_x and
                        ny >= lower_y and ny <= upper_y):
                        self.nose_cache = nose_coords[0]
                else:
                    self.nose_cache = nose_coords[0]
            nx, ny, nw, nh = None, None, None, None

            is_nose = False
            if len(self.nose_cache) > 0:
                (nx, ny, nw, nh) = self.nose_cache
                is_nose = True
            

            # for (x2,y2,w2,h2) in eyes:
            #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            #     radius = int(round((w2 + h2)*0.25))
            #     frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

            if has_face and (x_offset2 < frame.shape[0]):
                self.try_update_filter()
                scale_factor = radius / 100

                scale_tuple = lambda t, s : (math.floor(t[1] * s), math.floor(t[0] * s))
                scaled_left_ear = cv.resize(self.elected_left_ear, scale_tuple(self.elected_left_ear.shape, scale_factor))
                scaled_right_ear = cv.resize(self.elected_right_ear, scale_tuple(self.elected_right_ear.shape, scale_factor))
                scaled_nose = cv.resize(self.nose, scale_tuple(self.nose.shape, scale_factor))


                # Position the ears / nose according to the face orientation
                y1, y2 = eye_list[0][1]-scaled_left_ear.shape[0], eye_list[0][1] # TODO: error here?
                x1, x2 = eye_list[0][0]-scaled_left_ear.shape[1], eye_list[0][0] # error here?
                y11, y21 = eye_list[1][1]-scaled_right_ear.shape[0], eye_list[1][1]
                x11, x21 = eye_list[1][0], eye_list[1][0]+scaled_right_ear.shape[1]
                if is_nose:
                    nose_offset_y = y+ny+math.ceil(nh/2)
                    nose_offset_x = x+nx+math.ceil(nw/2)
                    y12, y22 = nose_offset_y-math.ceil(scaled_nose.shape[0]/2), nose_offset_y+math.floor(scaled_nose.shape[0]/2)
                    x12, x22 =nose_offset_x-math.ceil(scaled_nose.shape[1]/2), nose_offset_x+math.floor(scaled_nose.shape[1]/2)

                alpha_s = scaled_left_ear[:, :, 3] / 255.0
                alpha_s1 = scaled_right_ear[:, :, 3] / 255.0
                alpha_s2 = scaled_nose[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                alpha_l1 = 1.0 - alpha_s1
                alpha_l2 = 1.0 - alpha_s2
                for c in range(0, 3):
                    # Need to make sure all shapes are in bounds
                    if (y1 > 0 and y1 < frame.shape[0] and x1 > 0 and x1 < frame.shape[1] and
                        y2 > 0 and y2 < frame.shape[0] and x2 > 0 and x2 < frame.shape[1]):
                        frame[y1:y2, x1:x2, c] = (alpha_s * scaled_left_ear[:, :, c]) + (alpha_l * frame[y1:y2, x1:x2, c])
                    if (y11 > 0 and y11 < frame.shape[0] and x11 > 0 and x11 < frame.shape[1] and
                        y21 > 0 and y21 < frame.shape[0] and x21 > 0 and x21 < frame.shape[1]):
                        frame[y11:y21, x11:x21, c] =  (alpha_s1 * scaled_right_ear[:, :, c] + alpha_l1 * frame[y11:y21, x11:x21, c])
                    if is_nose:
                        frame[y12:y22, x12:x22, c] =  (alpha_s2 * scaled_nose[:, :, c] + alpha_l2 * frame[y12:y22, x12:x22, c])

        cv.imshow('Capture - Face detection', frame)

    def process_filter_vote(self, emotion):
        """
        Hàm này thực hiện bỏ phiếu cho cảm xúc được phân loại từ mô hình. 
        Các biểu hiện vui mừng (happy và surprised) sẽ được tính là phiếu cho bộ lọc vui mừng, 
        các biểu hiện buồn bã (sad, angry, disgusted) sẽ được tính là phiếu cho bộ lọc buồn bã, 
        và các biểu hiện trung tính (neutral và fearful) sẽ được tính là phiếu cho bộ lọc trung tính. 
        Đồng thời, số lượng phiếu trong chu kỳ bầu cử sẽ được tăng lên.
        """
        # Update the correct vote count monotonically
        if emotion == Emotion.happy or emotion == Emotion.surprised:
            self.filter_votes[Emotion.happy] += 1
        elif emotion == Emotion.sad or emotion == Emotion.angry or emotion == Emotion.disgusted:
            self.filter_votes[Emotion.sad] += 1
        else: #Neutral or fearful
            self.filter_votes[Emotion.neutral] += 1

        self.num_election_votes += 1

    def try_update_filter(self):
        """
        Chọn bộ lọc đúng để đặt lên người dùng dựa trên cảm xúc được bầu chọn nhiều nhất trong các khung hình gần đây nhất. 
        Nếu đủ số phiếu đã được bầu (đạt mức VOTES_REQUIRED_FOR_ELECTION, ở đây là sau 5 frame), 
        hàm sẽ thực hiện quyết định cuối cùng về bộ lọc dựa trên cảm xúc đã được bầu nhiều nhất.
        """
        # Check to see if we need to complete an election cycle
        if self.num_election_votes >= VOTES_REQUIRED_FOR_ELECTION:
            # Use operator module to choose the Emotion with the highest number of votes
            elected_emotion = max(self.filter_votes.items(), key=operator.itemgetter(1))[0]

            # TODO: Change the scaling of each image (as each is slightly differently sized & oriented)
            # Choose the appropriate preloaded image
            if elected_emotion == Emotion.happy:
                self.elected_left_ear = self.happy_left_img
                self.elected_right_ear = self.happy_right_img
            elif elected_emotion == Emotion.sad:
                self.elected_left_ear = self.sad_left_img
                self.elected_right_ear = self.sad_right_img
            else: # Fearful or Neutral 
                self.elected_left_ear = self.neutral_left_img
                self.elected_right_ear = self.neutral_right_img
            print("VOTED FOR FILTER: ", EMOTIONS[elected_emotion])
            #reset lại trạng thái bầu cử cho chu kì tiếp theo
            self.reset_election()

    def reset_election(self):
        """
        Hàm reset_election trong class Filter có chức năng đặt lại trạng thái của chu kỳ bầu cử để bắt đầu một chu kỳ bầu cử mới. 
        + self.filter_votes = {Emotion.happy: 0, Emotion.neutral: 0, Emotion.sad: 0}: Reset số phiếu của mỗi bộ lọc về 0. 
        Dictionary filter_votes được sử dụng để theo dõi số phiếu cho mỗi loại cảm xúc (vui mừng, trung tính, buồn bã).
        + self.num_election_votes = 0: Đặt lại số lượng phiếu đã được xử lý trong chu kỳ bầu cử hiện tại về 0. 
        Biến num_election_votes là biến đếm số lượng phiếu đã được xử lý, và nó tăng lên mỗi lần gọi hàm process_filter_vote khi có một cảm xúc được phát hiện.
        """
        self.filter_votes = {Emotion.happy: 0, Emotion.neutral: 0, Emotion.sad: 0}

        # The current number of votes currently processed in this filter election cycle;
        # 0 <= num_election_votes <= VOTES_REQUIRED_FOR_ELECTION
        self.num_election_votes = 0
        
