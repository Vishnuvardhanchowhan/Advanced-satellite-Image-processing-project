import tkinter
import tkinter as ttk
from tkinter import *
from tkinter import filedialog
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import locale
import math

root = tkinter.Tk()
locale.setlocale(locale.LC_NUMERIC, 'pl_PL.UTF8')

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])
sobel_x = sobel_x / 8
sobel_y = sobel_y / 8

rect_mask = np.ones([3, 3], dtype='float64')
rect_mask /= 9


def harris_response_value(IxIx, IyIy, IxIy, k):
    return IxIx * IyIy - IxIy ** 2 - k * (IxIx + IyIy) ** 2


def conv_image(img, kernel):
    n = img.shape[0]
    m = img.shape[1]
    convoluted_img = np.zeros(shape=(n - 2, m - 2))
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            window = img[i - 1:i + 2, j - 1:j + 2]
            convoluted_img[i - 1, j - 1] = np.sum(np.multiply(window, kernel))

    return convoluted_img


def corner_detection_with_gauss(k, sigma, threshold, img):
    image = cv2.imread(img)
    image_gray = cv2.imread(img, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    n = image_gray.shape[0]
    m = image_gray.shape[1]

    output_image = np.copy(image)

    Ix = conv_image(image_gray, sobel_x)
    Iy = conv_image(image_gray, sobel_y)

    IxIx = gaussian_filter(Ix ** 2, sigma)
    IxIy = gaussian_filter(Iy * Ix, sigma)
    IyIy = gaussian_filter(Iy ** 2, sigma)
    for i in range(n - 2):
        for j in range(m - 2):
            if harris_response_value(IxIx[i, j], IyIy[i, j], IxIy[i, j], k) > threshold:
                output_image[i, j] = [255, 0, 0]
    return output_image


def corner_detection_without_gauss(k, threshold, img):
    image = cv2.imread(img)
    image_gray = cv2.imread(img, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    n = image_gray.shape[0]
    m = image_gray.shape[1]

    output_image = np.copy(image)

    Ix = conv_image(image_gray, sobel_x)
    Iy = conv_image(image_gray, sobel_y)
    Ixx = Ix ** 2
    Ixy = Iy * Ix
    Iyy = Iy ** 2
    IxIx = np.zeros(shape=(n - 2, m - 2))
    IxIy = np.zeros(shape=(n - 2, m - 2))
    IyIy = np.zeros(shape=(n - 2, m - 2))
    for i in range(1, n - 3):
        for j in range(1, m - 3):
            w1 = Ixx[i - 1:i + 2, j - 1:j + 2]
            IxIx[i, j] = np.sum(np.multiply(w1, rect_mask))
            w2 = Ixy[i - 1:i + 2, j - 1:j + 2]
            IxIy[i, j] = np.sum(np.multiply(w2, rect_mask))
            w3 = Iyy[i - 1:i + 2, j - 1:j + 2]
            IyIy[i, j] = np.sum(np.multiply(w3, rect_mask))

    for i in range(n - 2):
        for j in range(m - 2):
            if harris_response_value(IxIx[i, j], IyIy[i, j], IxIy[i, j], k) > threshold:
                output_image[i, j] = [255, 0, 0]
    return output_image


root.title("Basic GUI Layout with Grid")
root.maxsize(900, 600)  # width x height
root.config(bg="skyblue")
root.title("Corner detection GUI")
# Create left and right frames
left_frame = Frame(root, width=200, height=400, bg='grey')
left_frame.pack(side='left', fill='both', padx=10, pady=5, expand=True)
right_frame = Frame(root, width=650, height=400, bg='grey')
right_frame.pack(side='right', fill='both', padx=10, pady=5, expand=True)
# Create frames and labels in left_frame
img_load = Frame(left_frame, width=90, height=185, bg='lightgrey')
img_load.pack(side='left', fill='both', padx=5, pady=5, expand=True)
slider_bar = Frame(right_frame, width=90, height=185, bg='lightgrey')
slider_bar.pack(side='right', fill='both', padx=5, pady=5, expand=True)


def openFile():
    filepath = filedialog.askopenfilename(
        initialdir="D:\Work_Files\Books_and_study_files\Engineering\Electrical\GNR602_Satellite_Image_Processing",
        title="Open Image file",
        filetypes=(("JPG files", "*.jpg"),
                   ("all files", "*.*")))
    return filepath


def show_corner(sigma, threshold, filepath):
    plt.imshow(corner_detection_with_gauss(0.06, sigma, threshold, filepath))
    plt.title('Corner detection with gaussian')
    plt.show()
    plt.imshow(corner_detection_without_gauss(0.06, threshold, filepath))
    plt.title('Corner detection without gaussian')
    plt.show()


def tuner():
    filepath = openFile()
    sigma = s1.get()
    threshold = thresold.get()
    show_corner(sigma, threshold, filepath)


locale.setlocale(locale.LC_NUMERIC, 'pl_PL.UTF8')


class NewScale(tkinter.Frame):
    def __init__(self, master=None, **options):
        tkinter.Frame.__init__(self, master)

        # Disable normal value display...
        options['showvalue'] = False
        # ... and use custom display instead
        options['command'] = self._on_scale

        # Set resolution to 1 and adjust to & from value
        self.res = options.get('resolution', 1)
        from_ = int(options.get('from_', 0) / self.res)
        to = int(options.get('to', 100) / self.res)
        options.update({'resolution': 1, 'to': to, 'from_': from_})

        # This could be improved...
        if 'digits' in options:
            self.digits = ['digits']
            del options['digits']
        else:
            self.digits = 2

        self.scale = tkinter.Scale(self, **options)
        self.scale_label = tkinter.Label(self)
        orient = options.get('orient', tkinter.HORIZONTAL)
        if orient == tkinter.HORIZONTAL:
            side, fill = 'right', 'y'
        else:
            side, fill = 'top', 'x'
        self.scale.pack(side=side, fill=fill)
        self.scale_label.pack(side=side)

    def _on_scale(self, value):
        value = locale.atof(value) * self.res
        value = locale.format_string('%.*f', (self.digits, value))
        value = value.replace(",", ".")
        self.scale_label.configure(text=value)

    def get(self):
        return self.scale.get() * self.res

    def set(self, value):
        self.scale.set(int(0.5 + value / self.res))


sig = DoubleVar()
thresold = DoubleVar()
s1 = NewScale(slider_bar, variable=sig, from_=0.00, to=10.00, orient=HORIZONTAL, resolution=0.01, length=500)
s1.pack(padx=5, pady=5)
Label(slider_bar, text="Sigma value").pack(side='top', padx=5, pady=5)
s2 = tkinter.Scale(slider_bar, variable=thresold, from_=0, to=5000, orient=HORIZONTAL, bigincrement=10, length=1000)
s2.pack(padx=5, pady=5)
Label(slider_bar, text="Threshold value").pack(side='top', padx=5, pady=5)

button = Button(img_load, text="Select Image", command=tuner)
button.pack()
root.mainloop()
