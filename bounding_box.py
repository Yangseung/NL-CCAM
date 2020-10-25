"""
Find the bounding box of an object
===================================

This example shows how to extract the bounding box of the largest object

"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import pdb

def connected_component(im):
    # l=224
    # n=10
    # im = ndimage.gaussian_filter(im, sigma=l/(4.*n))

    mask = im > im.mean()
    # mask = im > 0

    label_im, nb_labels = ndimage.label(mask)

    # Find the largest connected component

    # sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

    # mask_size = sizes < 1000
    # remove_pixel = mask_size[label_im]
    # label_im[remove_pixel] = 0

    labels = np.unique(label_im)
    label_im = np.searchsorted(labels, label_im)
    # Now that we have only one connected component, extract it's bounding box
    object_slices = ndimage.find_objects(label_im)
    # Find the object with the largest area
    areas = [np.product([x.stop - x.start for x in slc]) for slc in object_slices]
    largest = object_slices[np.argmax(areas)]

    # for i in labels[1:]:
    #     slice_x, slice_y = ndimage.find_objects(label_im == i)[0]
    #     pdb.set_trace()
    #     boxAArea = (int(slice_x[1]) - int(slice_x[0]) + 1) * (int(slice_y[1]) - int(slice_y[0]) + 1)
    #     if boxAArea >= box_max:
    #         box_max = boxAArea
    #         j = i
    # slice_x, slice_y = ndimage.find_objects(label_im == j)[0]
    return [largest[0].start, largest[0].stop], [largest[1].start, largest[1].stop]
