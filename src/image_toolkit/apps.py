from django.apps import AppConfig

import urllib.request
import cv2
import numpy as np
from dlib import get_frontal_face_detector, shape_predictor

from scipy.spatial import distance as dist
from scipy.spatial import ConvexHull

from onedsoni.settings.base import SAVE_IMAGES_DIR, PRE_TRAINED_MODELS_DIR

class ImageToolkitConfig(AppConfig):
    name = 'image_toolkit'

class IMAGE_HELPER:

    def __init__(self):

        pass
    def __init__(self, image, process_image = True, resize = None):
        pass

    def save_image_in_static(image, file_name):
        location_to_save = SAVE_IMAGES_DIR + '/' +file_name
        #TODO put some checks maybe
        if cv2.imwrite(location_to_save , image):
            return location_to_save
        else :
            #add some not done image path
            return -1

    def get_faces_marked(image, faces,
                        color = (0,255,0),
                        line_width = 3,
                        inplace = False,
                        is_opencv = True):
        temp_image = np.copy( image )
        if len(faces) > 0:
            if is_opencv:
                for index, (x,y,w,h) in enumerate(faces):
                    cv2.rectangle( temp_image, (x, y), (x+w, y+h),
                                    color, line_width)
                    cv2.putText(temp_image, str(index + 1), (x, y),
			                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else :
                for index,face in enumerate(faces):
                    x = face.left()
                    y = face.top()
                    x2 = face.right()
                    y2 = face.bottom()
                    cv2.rectangle( temp_image, (x, y), (x2, y2),
                                        color, line_width)
                    cv2.putText(temp_image, str(index + 1), (x, y),
			                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return temp_image
        else:
            return None

    def fetch_image_from_file( image_path):
            #check if image exists
            return cv2.imread( image_path, 0)
    def fetch_image_from_url( url):
            response = urllib.request.urlopen( url)
            data = response.read()
            image_data = np.asarray( bytearray( data), dtype = 'uint8')
            try:
                image = cv2.imdecode( image_data, cv2.IMREAD_COLOR)
            except Exception as e:
                image = None
            finally:
                return image

    def resize_image(image,
                    factor = None, new_width = None,
                    new_height = None, inplace = False ):
        if image is None:
            return None
        if new_width is None and new_height is None and factor is None:
            return None
        height, width, channels = image.shape
        if factor is not None:
            new_width = int(width*factor)
            new_height = int( height*factor)
        elif new_width is not None:
            new_height = height // (width//new_width)
        else :
            new_width = width// (height//new_height)
        return cv2.resize( temp_image, (new_width, new_height))

class OPENCV_HELPER:
    haarcascade_file = "haarcascade_frontalface_default.xml"
    def __init__( self):
        pass
    def __init__(self,url = None, image = None, process_image = True, resize = None):
        if image is None:
            self.image = IMAGE_HELPER.fetch_image_from_url(url)
        else :
            self.image = image
        if self.image is None:
            return None
        if resize is not None:
            factor = resize.get('factor', None)
            new_width = resize.get('new_width', None)
            new_height = resize.get('new_height', None)
            self.image = IMAGE_HELPER.resize_image(image=self.image, factor = factor,
                                new_width=new_width,
                                new_height=new_height)
        if process_image == True:
            faces_data = self.get_faces()
            self.facial_points = faces_data['facial_points']
            self.num_faces = faces_data['num_faces']
            self.marked_image = IMAGE_HELPER.get_faces_marked(image=self.image,
                                                    faces=self.facial_points,
                                                    is_opencv = True)
            self.marked_image_loc = IMAGE_HELPER.save_image_in_static(
                                        self.marked_image, 'opencv_faces.jpg')

    def get_faces(self,scaleFactor = 1.1,minNeighbors = 5,
                    minSize = (30,30),flags = cv2.CASCADE_SCALE_IMAGE  ):

        image_to_check = self.image
        if ( len(image_to_check.shape) > 2):
            image_to_check = cv2.cvtColor( image_to_check,
                                        cv2.COLOR_BGR2GRAY)
        image_to_check = cv2.equalizeHist( image_to_check)
        face_dector = cv2.CascadeClassifier(
                        cv2.data.haarcascades + OPENCV_HELPER.haarcascade_file)
        faces = face_dector.detectMultiScale(
                                    image_to_check,
                                    scaleFactor = scaleFactor,
                                    minNeighbors = minNeighbors,
                                    minSize = minSize,
                                    flags = flags,
                                    )
        send_data = {"num_faces" : len(faces),
                    "facial_points" : faces}
        return send_data

class DLIB_HELPER:
    PREDICTOR_PATH = PRE_TRAINED_MODELS_DIR+"/shape_predictor_68_face_landmarks.dat"
    FULL_POINTS = list(range(0, 68))
    FACE_POINTS = list(range(17, 68))
    JAWLINE_POINTS = list(range(0, 17))
    RIGHT_EYEBROW_POINTS = list(range(17, 22))
    LEFT_EYEBROW_POINTS = list(range(22, 27))
    NOSE_POINTS = list(range(27, 36))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    MOUTH_OUTLINE_POINTS = list(range(48, 61))
    MOUTH_INNER_POINTS = list(range(61, 68))

    detector = get_frontal_face_detector()
    predictor = shape_predictor(PREDICTOR_PATH)

    def __init__( self):
        pass
    def __init__(self,url = None, image = None, process_image = True, resize = None):
        if image is None:
            self.image = IMAGE_HELPER.fetch_image_from_url(url)
        else :
            self.image = image
        if self.image is None:
            return None

        self.feature_marked_image = np.copy( self.image)

        if resize is not None:
            factor = resize.get('factor', None)
            new_width = resize.get('new_width', None)
            new_height = resize.get('new_height', None)
            self.image = IMAGE_HELPER.resize_image(image=self.image, factor = factor,
                                new_width=new_width,
                                new_height=new_height)

        if process_image == True:
            faces_data = self.get_faces()
            self.facial_points = faces_data['facial_points']
            self.num_faces = faces_data['num_faces']
            self.marked_image = IMAGE_HELPER.get_faces_marked(image=self.image,
                                                    faces=self.facial_points,
                                                    is_opencv = False)
            self.facial_features_list = self.get_facial_features()

            self.get_facial_features_marked() #self.feature_marked_image

            self.marked_image_loc = IMAGE_HELPER.save_image_in_static(
                                        self.marked_image, 'dlib_faces.jpg')
            self.feature_marked_image_loc = IMAGE_HELPER.save_image_in_static(
                                        self.feature_marked_image,
                                        'dlib_faces_with_features.jpg')
    def get_shape_to_np(shape):
        xy = np.zeros((68,2), dtype = 'int')
        for i in range(0,68):
            xy[i] = ( shape.part(i).x, shape.part(i).y )
        return xy

    def get_facial_features_marked(self, color=(0,255,0), radius = 2):
        self.feature_marked_image = np.copy( self.image)
        full_features = []
        for index, rect in enumerate( self.facial_points):
            shape = DLIB_HELPER.predictor( self.gray_scale_image, rect)
            shape = DLIB_HELPER.get_shape_to_np(shape)
            for x,y in shape:
                cv2.circle( self.feature_marked_image, (x,y), radius, color,-1)

    def get_faces(self):
        image_to_check = self.image
        if ( len(image_to_check.shape) > 2):
            image_to_check = cv2.cvtColor( image_to_check,
                                        cv2.COLOR_BGR2GRAY)
        image_to_check = cv2.equalizeHist( image_to_check)
        self.gray_scale_image = image_to_check
        faces = DLIB_HELPER.detector( image_to_check, 0)

        send_data = {"num_faces" : len(faces),
                    "facial_points" : faces}
        return send_data

    def get_facial_features(self):
        self.facial_features_list = list()
        for index, face in enumerate(self.facial_points):
            x = face.left()
            y = face.top()
            x1 = face.right()
            y1 = face.bottom()
            landmarks = np.matrix([[p.x, p.y] for p in DLIB_HELPER.predictor(self.gray_scale_image, face).parts()])
            full_features = landmarks[DLIB_HELPER.FULL_POINTS]
            face = landmarks[DLIB_HELPER.FACE_POINTS]
            left_eyebrow = landmarks[DLIB_HELPER.LEFT_EYEBROW_POINTS]
            right_eyebrow = landmarks[DLIB_HELPER.RIGHT_EYEBROW_POINTS]
            left_eye = landmarks[DLIB_HELPER.LEFT_EYE_POINTS]
            right_eye = landmarks[DLIB_HELPER.RIGHT_EYE_POINTS]
            nose = landmarks[DLIB_HELPER.NOSE_POINTS]
            mouth_outer = landmarks[DLIB_HELPER.MOUTH_OUTLINE_POINTS]
            mouth_inner = landmarks[DLIB_HELPER.MOUTH_INNER_POINTS]
            jawline = landmarks[DLIB_HELPER.JAWLINE_POINTS]
            smiling = self.is_smiling(mouth_outer)
            self.facial_features_list.append({'full_features':full_features, 'face':face,
                    'left_eyebrow':left_eyebrow, 'right_eyebrow':right_eyebrow,
                    'left_eye':left_eye, 'right_eye': right_eye, 'nose':nose,
                    'mouth_outer':mouth_outer, 'mouth_inner':mouth_inner,
                    'jawline':jawline, 'face_number':(index + 1), 'is_smiling': smiling})
        return self.facial_features_list

    def is_smiling(self, mouth):
        p1 = dist.euclidean(mouth[3], mouth[9])
        p2 = dist.euclidean(mouth[2], mouth[10])
        p3 = dist.euclidean(mouth[4], mouth[8])
        avg_vertical_ = (p1+p2+p3)/3
        horizontal_ = dist.euclidean(mouth[0], mouth[6])
        ratio=avg_vertical_/horizontal_
        if ratio < .37 and ratio > .3:
            return 1
        else:
            return 0
        # return ratio

    def mouth_size(self, mouth):
        mouthWidth = dist.euclidean(mouth[0], mouth[-1])
        hull = ConvexHull(mouth)
        mouthCenter = np.mean(mouth[hull.vertices, :], axis=0)
        mouthCenter = mouthCenter.astype(int)
        return int(mouthWidth), mouthCenter
    def place_mouth(self, frame, mouthCenter, mouthSize):
        mouthSize = int(mouthSize * 1.5)
        x1 = int(mouthCenter[0,0] - (mouthSize*4))
        x2 = int(mouthCenter[0,0] + (mouthSize*4))
        y1 = int(mouthCenter[0,1] - (mouthSize*4))
        y2 = int(mouthCenter[0,1] + (mouthSize*4))
        h, w = frame.shape[:2]
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > w:
            x2 = w
        if y2 > h:
            y2 = h
        mouthOverlayWidth = x2 - x1
        mouthOverlayHeight = y2 - y1
        imgMouth = self.imgMouth
        orig_mouth_mask = self.orig_mouth_mask
        orig_mouth_mask_inv = self.orig_mouth_mask_inv
        mouthOverlay = cv2.resize(imgMouth, (mouthOverlayWidth,mouthOverlayHeight), interpolation = cv2.INTER_AREA)
        maskMouth = cv2.resize(orig_mouth_mask, (mouthOverlayWidth,mouthOverlayHeight), interpolation = cv2.INTER_AREA)
        mask_invMouth = cv2.resize(orig_mouth_mask_inv, (mouthOverlayWidth,mouthOverlayHeight), interpolation = cv2.INTER_AREA)
        roi2 = frame[y1-self.cig_y_shift:y2-self.cig_y_shift, x1-self.cig_x_shift:x2-self.cig_x_shift]
        roi2_bg = cv2.bitwise_and(roi2,roi2,mask = mask_invMouth)
        roi2_fg = cv2.bitwise_and(mouthOverlay,mouthOverlay,mask = maskMouth)
        dst = cv2.add(roi2_bg,roi2_fg)
        frame[y1-self.cig_y_shift:y2-self.cig_y_shift, x1-self.cig_x_shift:x2-self.cig_x_shift] = dst
    def eye_size(self, eye):
        eyeWidth = dist.euclidean(eye[0], eye[3])
        hull = ConvexHull(eye)
        eyeCenter = np.mean(eye[hull.vertices, :], axis=0)
        eyeCenter = eyeCenter.astype(int)
        return int(eyeWidth), eyeCenter
    def place_eye(self, frame, eyeCenter, eyeSize):
        eyeSize = int(eyeSize * 1.5)
        x1 = int(eyeCenter[0,0] - (eyeSize//2))
        x2 = int(eyeCenter[0,0] + (eyeSize//2))
        y1 = int(eyeCenter[0,1] - (eyeSize//2))
        y2 = int(eyeCenter[0,1] + (eyeSize//2))
        h, w = frame.shape[:2]
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > w:
            x2 = w
        if y2 > h:
            y2 = h
        eyeOverlayWidth = x2 - x1
        eyeOverlayHeight = y2 - y1
        imgEye = self.imgEye
        orig_mask = self.orig_mask
        orig_mask_inv = self.orig_mask_inv
        eyeOverlay = cv2.resize(imgEye, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(orig_mask, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(orig_mask_inv, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)
        roi = frame[y1:y2, x1:x2]
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        roi_fg = cv2.bitwise_and(eyeOverlay,eyeOverlay,mask = mask)
        dst = cv2.add(roi_bg,roi_fg)
        frame[y1:y2, x1:x2] = dst

    def mask_image(self, mouth, eyes):
        self.cig_x_shift = 35
        self.cig_y_shift = -4
        self.imgEye = cv2.imread('eye2.png',-1)
        self.imgMouth = cv2.imread('mouth.png', -1)

         # Create the mask from the overlay image
        self.orig_mask = self.imgEye[:,:,3]
        self.orig_mouth_mask = self.imgMouth[:,:,3]

         # Create the inverted mask for the overlay image
        self.orig_mask_inv = cv2.bitwise_not(self.orig_mask)
        self.orig_mouth_mask_inv = cv2.bitwise_not( self.orig_mouth_mask)

         # Convert the overlay image image to BGR
         # and save the original image size
        self.imgEye = self.imgEye[:,:,0:3]
        self.imgMouth = self.imgMouth[:,:,0:3]
        origEyeHeight, origEyeWidth = self.imgEye.shape[:2]
        origMouthHeight, origMouthWidth = self.imgMouth.shape[:2]

        left_eye = eyes[0]
        right_eye = eyes[1]

        mouthSize, mouthCentre = self.mouth_size(mouth)
        leftEyeSize, leftEyeCenter = self.eye_size(left_eye)
        rightEyeSize, rightEyeCenter = self.eye_size(right_eye)
        self.place_eye(self.image, leftEyeCenter, leftEyeSize)
        self.place_eye(self.image, rightEyeCenter, rightEyeSize)
        self.place_mouth( self.image, mouthCentre, mouthSize)
        cv2.imwrite(self.marked_image_loc, self.image)
