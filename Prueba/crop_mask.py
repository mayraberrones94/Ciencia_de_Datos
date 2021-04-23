from PIL import Image, ImageChops
import os

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -95)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

if __name__ == "__main__":
	for f in os.listdir('.'):
		if f.endswith('.png'):
			img = Image.open(f)
			fn, fext = os.path.splitext(f)
			cropped_img = trim(img)
			resize = cropped_img.resize((80, 200),/
			 Image.ANTIALIAS)
			resize.save('crop/{}.png'.format(fn))