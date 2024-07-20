from PIL import Image
import random
import numpy as np
import argparse
import cv2 as cv
import os

from numpy._core.multiarray import WRAP, array

class GlitchImage:
    def __init__(self, img: Image.Image):
        img_arr = np.asarray(img)
        width, height = img.size
        self.img_arr = img_arr
        self.img_width = width
        self.img_height = height
        self.out_arr = np.array(img)

        self.faces = []

    def apply_glitch(self, y_start, y_stop, x_start, x_stop, use_wrap = False):
        block_height = random.randint(int(self.img_height / 80), int(self.img_height / 50))
        # y_start = random.randint(0, self.img_height - block_height)
        # y_stop = y_start + block_height

        x_offset = random.randint(10, int(self.img_width / 3))

        if random.random() < .5:
            # glitch left
            x_start = self.img_width - x_offset
            x_stop = x_offset

            left = self.img_arr[y_start:y_stop, x_start:]
            wrap = self.img_arr[y_start:y_stop, :x_start]

            self.out_arr[y_start:y_stop, :x_stop] = left
            if use_wrap:
                self.out_arr[y_start:y_stop, x_stop:] = wrap
        else:
            # glitch left
            x_start = x_offset
            x_stop = self.img_width - x_offset

            right = self.img_arr[y_start:y_stop, :x_stop]
            wrap = self.img_arr[y_start:y_stop, x_stop:]

            self.out_arr[y_start:y_stop, x_start:] = right
            if use_wrap:
                self.out_arr[y_start:y_stop, :x_start] = wrap

    def glitch(self, nr: int, face=False):
        nr_of_glitches = 1
        if face == False and len(self.faces) == 0:
            y_offset = 400
            y_start = random.randint(0, self.img_height - y_offset)
            y_stop = y_start + y_offset
            y_size = int(y_offset / nr)
        else:
            nr_of_glitches = len(self.faces)

        for glitch_nr in range(0, nr_of_glitches):
            # print("nr of glitches", glitch_nr)
            if face == True:
                x_start, y_start, x_stop, y_stop, confidence = self.faces[glitch_nr]
                print("face nr", glitch_nr)
                print("y_start", "y_stop")
                print(y_start, y_stop)
                y_size = int((y_stop - y_start) / nr)

            for i in range(nr):
                # y_part_size = random.randint(y_size - 10, y_size + 10)
                y_part_size = y_size

                y_start_part = y_start + (y_part_size * i)
                y_stop_part = y_start + (y_part_size * (i+1))

                print(y_start_part, y_stop_part, y_size)
                self.apply_glitch(y_start_part, y_stop_part, 0, 0)

    def find_face(self):
        # https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
        prototxt_path = os.path.abspath("face_dnn/deploy.prototxt")
        # https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
        model_path = os.path.abspath("face_dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel")

        model = cv.dnn.readNetFromCaffe(prototxt_path, model_path)

        # convert PLI Image (RGB) to cv2 image (BGR)
        image = cv.cvtColor(self.img_arr, cv.COLOR_RGB2BGR)
        # image = cv.imread(path)

        # get width and height of the image
        h, w = image.shape[:2]
        # blob = cv.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0, (300,300), (104.0, 177.0, 123.0))
        # set the image into the input of the neural network
        model.setInput(blob)
        # perform inference and get the result
        output = np.squeeze(model.forward())

        faces = []
        for i in range(0, output.shape[0]):
            # get the confidence
            confidence = output[i, 2]
            # if confidence is above 50%, then draw the surrounding box

            if confidence > 0.5:
                # get the surrounding box cordinates and upscale them to original image
                box = output[i, 3:7] * np.array([w, h, w, h])
                # convert to integers
                start_x, start_y, end_x, end_y = box.astype(int)
                faces.append([start_x, start_y, end_x, end_y, confidence])
        self.faces = faces

    def to_image(self, path=None, debug=False):

        font_scale = 1.0
        cv_img=cv.cvtColor(self.out_arr, cv.COLOR_RGB2BGR)
        if debug:
            for face in self.faces:
                start_x, start_y, end_x, end_y, confidence = face

                # # draw the rectangle surrounding the face
                cv.rectangle(cv_img, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
                # # draw text as well
                cv.putText(cv_img, f"{confidence*100:.2f}%", (start_x, start_y-5), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)

        cv.imshow("image", cv_img)
        cv.waitKey(0)

        if path != None:
            full_path = os.path.abspath(path)
            cv.imwrite(full_path, cv_img)

def open_image(path: str) -> Image.Image:
    img = Image.open(path)
    return img

def img_to_glitch(img: Image.Image) -> GlitchImage:
    return GlitchImage(img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path")
    parser.add_argument("-o", "--out", required=False, default=None)
    parser.add_argument("-d", "--debug", required=False, default=False, action="store_true")
    parser.add_argument("-f", "--face", required=False, default=True)
    args = parser.parse_args()
    img = open_image(args.img_path)
    glitch_img = img_to_glitch(img)
    glitch_img.find_face()
    glitch_img.glitch(10, args.face)
    glitch_img.to_image(args.out, args.debug)
