import paddlehub as hub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pylab import plt
import numpy as np

def segment_human(img_path):
    image=mpimg.imread(img_path)
    module = hub.Module(name="ace2p")
    return module.segmentation(images = [image],
                    output_dir = 'temp',
                    visualization = True)
    # with open("test.txt", 'w') as file:
    #     file.write(np.array2string(results[0]['data'][0][0]))
    
if __name__ == "__main__":
    segment_human("test.jpg")
