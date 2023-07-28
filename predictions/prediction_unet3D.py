import cv2
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
def get_labeled_image(image, label, is_categorical=False):

    if not is_categorical:
        label = to_categorical(label, num_classes=4).astype(np.uint8)

    image = cv2.normalize(image[:,:,:,0], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                         dtype=cv2.CV_32F).astype(
        np.uint8
    )

    labeled_image = np.zeros_like(label[:,:,:,1:])

    labeled_image[:,:,:,0] = image * (label[:,:,:,0])
    labeled_image[:,:,:,1] = image * (label[:,:,:,0])
    labeled_image[:, :, :, 2] = image * (label[:, :, :, 0])

    labeled_image += label[:, :, :, 1:] * 255
    return labeled_image


def predict_and_viz(image, label, model, threshold, loc=(100,100,50)):

    image_labeled = get_labeled_image(image.copy(), label.copy())

    model_label = np.zeros([3,320, 320, 160])

    for x in range(0, image.shape[0], 160):
        for y in range(0, image.shape[1], 160):
            for z in range(0, image.shape[2], 160):
                patch = np.zeros([4,160,160,16])
                p = np.moveaxis(image[x: x+160, y: y+160, z: z+16], 3, 0)
                patch[:, 0:patch.shape[1], 0:patch.shape[2], 0:patch.shape[3]] = p
                pred = model.predict(np.expand_dims(patch,0))
                model_label[:, x:x + p.shape[1],
                y: y + p.shape[2],
                z: z + p.shape[3]] += pred[0][:, :p.shape[1],:p.shape[2],
                                      :p.shape[3]]


    model_label = np.moveaxis(model_label[:, 0:240, 0:240, 0:155], 0, 3)
    model_label_reformatted = np.zeros((240,240,150,4))

    model_label_reformatted = to_categorical(label, num_classes=4).astype(np.uint8)

    model_label_reformatted[:,:,:,1:4] = model_label

    model_labeled_image = get_labeled_image(image, model_label_reformatted,
                                            is_categorical=True)

    fig, ax = plt.subplots(2,3, figsize=[10, 7])

    x,y,z = loc

    ax[0][0].imshow(np.rot90(image_labeled[x,:,:,:]))
    ax[0][0].set_ylabel('Grouth Truth', fontsize=15)
    ax[0][0].set_xlabel('Sagital', fontsize=15)

    ax[0][1].imshow(np.rot90(image_labeled[:,y,:,:]))
    ax[0][1].set_xlabel('Coronal', fontsize=15)

    ax[0][2].imshow(np.rot90(image_labeled[:,:,z,:]))
    ax[0][2].set_xlabel('Tranversal', fontsize=15)

    ax[1][0].imshow(np.rot90(model_labeled_image[x,:,:,:]))
    ax[1][0].set_ylabel('Prediction', fontsize=15)

    ax[1][1].imshow(np.rot90(model_labeled_image[:,y, :, :]))
    ax[1][2].imshow(model_labeled_image[:, :, z, :])

    fig.subplots_adjust(wspace=0, hspace=.12)

    for i in range(2):
        for j in range(3):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])

    return model_label_reformatted