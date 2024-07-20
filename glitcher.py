from PIL import Image
import random
import numpy as np
import argparse
import cv2 as cv
import os

from numpy._core.multiarray import zeros

__DEBUG__ = False

class GlitchImage:
    def __init__(self, img: Image.Image):
        img_arr = np.asarray(img)
        width, height = img.size
        self.img_arr = img_arr
        self.img_width = width
        self.img_height = height
        self.out_arr = np.array(img)

        self.faces = []

    def glitch_rgb(self, img_arr):
        b,g,r = cv.split(img_arr)
        zeros = np.zeros(img_arr.shape[:2], dtype="uint8")

        # randomize RGB shuffel include zeroes in the mix
        arr = [b, g, r, zeros]
        random.shuffle(arr)
        return cv.merge(arr[:3])

    def apply_glitch(self, y_start, y_stop, x_start, x_stop, use_wrap = True):
        block_height = random.randint(int(self.img_height / 80), int(self.img_height / 50))

        x_offset = random.randint(-int(self.img_width / 10), int(self.img_width / 10))

        whole_chunk = self.img_arr[y_start:y_stop, :]
        print("whole chunk shape", whole_chunk.shape)

        # grab a chunk a bit bigger then x,y start to cover the whole target
        x_size_offset = random.randint(int(self.img_width / 10), int(self.img_width / 5))
        print("x_offset", x_offset, "size offset x ", x_size_offset)
        start = x_start-x_size_offset
        end = x_stop+x_size_offset

        # if random.random() < .5:
        #     end = x_stop+x_size_offset
        # else:
        #     end = x_stop-x_size_offset

        size = end - start

        print("get chunk from x", start, "end", end, "size", size)

        move_to_x_start = x_start + x_offset
        move_to_x_stop = x_start + size + x_offset
        if move_to_x_stop > self.img_width:
            print("in arr to big, stop", move_to_x_stop - self.img_width)
            end = end - (move_to_x_stop - self.img_width )
            move_to_x_stop = self.img_width
        if move_to_x_start > self.img_width:
            print("in arr to big, start",  self.img_width - move_to_x_start)
            start = start + (self.img_width - move_to_x_start)
            move_to_x_start = self.img_width

        print("paste chunk to x", move_to_x_start, "end", move_to_x_stop, "size", size)

        change = self.img_arr[y_start:y_stop, start:end]

        print("change arr shape", change.shape)
        print("out arr shape", self.out_arr[y_start:y_stop, move_to_x_start:move_to_x_stop].shape)

        if random.random() < .5:
            change = self.glitch_rgb(change)

        self.out_arr[y_start:y_stop, move_to_x_start:move_to_x_stop] = change

        debug_color = (0, 255, 0)
        debug_color2 = (255, 0, 255)
        if __DEBUG__:
            cv_img=cv.cvtColor(self.out_arr, cv.COLOR_RGB2BGR)
            # cv.rectangle(cv_img, (x_start, y_start), (x_stop, y_stop), color=debug_color, thickness=2)
            cv.rectangle(cv_img, (move_to_x_start, y_start), (move_to_x_stop, y_stop), color=debug_color, thickness=2)
            cv.rectangle(cv_img, (start, y_start), (end, y_stop), color=debug_color2, thickness=2)
            # cv.arrowedLine(cv_img,(x_start,y_start),(x_stop, y_start), color=debug_color, thickness=2)
            cv.imshow("image", cv_img)
            cv.waitKey(0)

    def glitch(self, nr: int):
        if len(self.faces) <= 0:
            return

        for glitch_nr in range(0, len(self.faces)):
            x_start, y_start, x_stop, y_stop, confidence = self.faces[glitch_nr]
            y_size = int((y_stop - y_start) / nr)

            for i in range(nr):
                # y_part_size = random.randint(y_size - 10, y_size + 10)
                y_part_size = y_size

                y_start_part = y_start + (y_part_size * i)
                y_stop_part = y_start + (y_part_size * (i+1))

                # print(y_start_part, y_stop_part, y_size)
                self.apply_glitch(y_start_part, y_stop_part, x_start, x_stop)

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
    # parser.add_argument("-f", "--face", required=False, default=True)
    args = parser.parse_args()
    __DEBUG__ = args.debug
    img = open_image(args.img_path)
    glitch_img = img_to_glitch(img)
    glitch_img.find_face()
    glitch_img.glitch(12)
    glitch_img.to_image(args.out, args.debug)
