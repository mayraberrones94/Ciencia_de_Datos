import os
source1 = "/home/mayra/Documentos/INCAN/No-80"
dest11 = "/home/mayra/Documentos/INCAN/Train-80/No-80"
files = os.listdir(source1)
import shutil
import numpy as np
for f in files:
	if np.random.rand(1) < 0.75:
		shutil.move(source1 + '/'+ f, dest11 + '/'+ f)