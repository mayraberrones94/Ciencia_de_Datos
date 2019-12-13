from PIL import Image, ImageChops, ImageFont, ImageDraw, ImageEnhance
import numpy as np
import os

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 1.0, -90)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

if __name__ == "__main__":
	for f in os.listdir('.'):
		if f.endswith('.png'):
			img = Image.open(f)
			fn, fext = os.path.splitext(f)
			draw = ImageDraw.Draw(img)
			#draw.rectangle(((0, 0), (145, 101)), fill="black")
			#draw.rectangle(((0, 0), (230, 48)), fill="black")
			#draw.rectangle(((500, 0), (890, 101)), fill="black")
			_box = img.convert('RGB').getbbox()
			crop = img.crop(img)
			cropped_img = trim(crop)
			#resize = cropped_img.resize((160, 400),Image.ANTIALIAS)
			cropped_img.save('crop/{}.png'.format(fn))