
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os


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


if __name__ == "__main__":
	from PIL import Image
	model = load_model('/Users/mayraberrones/Downloads/Incan-modelos/m2-adam-128-64.h5' )
	for f in os.listdir('.'):
		if f.endswith('.png'):
			new_image = load_image(f)
			fn, fext = os.path.splitext(f)
			pred = model.predict(new_image)
			#print('{}'.format(fn), pred)
			if pred == 0:
				print('{} 0'.format(fn))
			elif pred >.5:
				print('{} 1'.format(fn), pred)
			else:
				print('{} 0'.format(fn), 1 - pred)



