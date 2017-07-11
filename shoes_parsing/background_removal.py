from PIL import ImageOps, Image
from scipy.ndimage.filters import sobel, gaussian_filter
import numpy as np
import cv2
import matplotlib.pyplot as plt

THRESHOLD = 50


def trans_mask_sobel(img):
    """ Generate a transparency mask for a given image """

    # Find object
    img = ImageOps.invert(img)
    arr = np.array(img)
    arr = sobel(arr)
    arr = gaussian_filter(arr,5)
    mask = arr[:,:,:] > THRESHOLD
    arr[mask] = 255
    mask = arr < THRESHOLD
    arr[mask] = 0
    plt.imshow(arr)
    plt.show()
    mask = np.zeros((arr.shape[0]+2, arr.shape[1] + 2,3)).astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cv2.floodFill(arr, mask, (0,0), (255))
    
    
    return arr


def alpha_composite(image, mask):
    """ Composite two images together by overriding one opacity channel """

    compos = pg.Image(mask)
    compos.composite(
        image,
        image.size(),
        pg.CompositeOperator.CopyOpacityCompositeOp
    )
    return compos

def remove_background(img):
    """ Remove the background of the image in 'filename' """
    transmask = trans_mask_sobel(img)
    #img = alphacomposite(transmask, img)
    #img.trim()
    #img.write('out.png')
    #return os.path.abspath('out.png')


img = Image.open('test.jpg')
remove_background(img)






