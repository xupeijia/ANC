import numpy as np
from scipy import signal
import utils
import os
import matplotlib.pyplot as plt


path_from = 'D:/Study/Projects/MRI/Data/Body/GE/'
filenames = [filename for filename in os.listdir(path_from) if '.txt' in filename]
filename = path_from + filenames[0]
t, x = utils.read_vibration(file_name=filename, start_line=5, sep='\t')


