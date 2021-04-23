from tkinter import * 

from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
from resizeimage import resizeimage

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image, ImageChops, ImageFont, ImageDraw, ImageEnhance
import numpy as np
import os
import tensorflow
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import decode_predictions

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 1.0, -90)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png")))
    img = Image.open(image_data)
    cropped_img = trim(img)
    res_image = resizeimage.resize_cover(cropped_img, [80, 200])
    res_image.save('crop.png',res_image.format)

    basewidth = 500
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = Label(frame, image=img).pack()


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(80, 200))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()
    return img_tensor

def classify():
    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png")))
    img = load_image(image_data)
    model_1 = load_model('Modelo_1136-sdrop.h5')

    predictions = model_1.predict(img)
    #label = decode_predictions(predictions)
    #table = Label(frame, text="Top image class predictions and confidences").pack()
    #for i in range(0, len(label[0])):
    result = Label(frame,
                    text= 'La probabilidad de anomalia: ' + str(predictions) ).pack()



root = tk.Tk()
root.title('Clasificador portable')
#root.iconbitmap('class.ico')
#root.resizable(False, False)

tit = tk.Label(root, text="Clasificador de Mamografías", padx=25, pady=6, font=("", 12)).pack()

canvas = tk.Canvas(root, height=800, width=800, bg='grey')
canvas.pack()

frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

chose_image = tk.Button(root, text='Recortar Imagen',
                        padx=35, pady=10,
                        fg="black", bg="grey", command=load_img)
chose_image.pack(side=tk.LEFT)

class_image = tk.Button(root, text='Clasificar Imagen',
                        padx=35, pady=10,
                        fg="black", bg="grey", command=classify)
class_image.pack(side=tk.RIGHT)

#vgg_model = 

root.mainloop()

    
