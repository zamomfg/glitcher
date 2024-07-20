from PIL import Image
import random
import numpy as np
import argparse

from numpy._core.multiarray import WRAP

class GlitchImage:
    def __init__(self, img: Image.Image):
        img_arr = np.asarray(img)
        width, height = img.size
        self.img_arr = img_arr
        self.img_width = width
        self.img_height = height
        self.out_arr = np.array(img)

    def glitch(self):
        block_height = 100
        # y_start = random.randint(0, self.img_height - block_height)
        y_start = 1000
        y_stop = y_start + block_height

        block_width = 400
        # x_start = random.randint(0, self.img_width)
        x_start = 1500
        x_stop = x_start + block_width

        x_offset = random.randint(0, block_width)
        # copy = self.img_arr[y_start:y_stop, x_start:x_stop]
        print(self.img_width, self.img_height)
        print("start_x", x_start, "stop_x", x_stop)
        print("block_width", block_width)
        print("start_y", y_start, "stop_y", y_stop)
        print("block_height", block_height)

        # test = self.out_arr[y_start:y_stop, x_start:, :]
        # self.out_arr[y_start:y_stop, x_start:, :] += [np.uint8(50), np.uint8(50), np.uint8(0)]
        # self.out_arr[y_start:y_stop, :x_stop, :] += copy[:, :, 1:]#[np.uint8(0), np.uint8(20), np.uint8(70)]
        # self.out_arr = copy
        pixel_square = self.img_arr[y_start:y_stop, :x_stop]
        wraped_square = self.img_arr[y_start:y_stop, x_stop:]
        Image.fromarray(wraped_square).show()
        Image.fromarray(pixel_square).show()

        # print(pixel_square.shape, wraped_square.shape)
        # print(self.out_arr[y_start:y_stop, x_start:].shape, self.out_arr[y_start:y_stop, :x_start].shape)
        # self.out_arr[y_start:y_stop, x_start:] = pixel_square
        # self.out_arr[y_start:y_stop, :x_start] = wraped_square

    def to_image(self) -> Image.Image:
        return Image.fromarray(self.out_arr)


def open_image(path: str) -> Image.Image:
    img = Image.open(path)
    return img

def img_to_glitch(img: Image.Image) -> GlitchImage:
    return GlitchImage(img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path")
    args = parser.parse_args()
    img = open_image(args.img_path)
    glitch_img = img_to_glitch(img)
    glitch_img.glitch()
    # glitch_img.to_image().show()
