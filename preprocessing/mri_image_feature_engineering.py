import numpy as np
import keras
def get_sub_volume(image, label,
                   orig_x=240, orig_y = 240, orig_z=155,
                   output_x=160, output_y=160, output_z=16,
                   num_classes= 4, max_tries=1000,
                   background_threshold=0.95,
                   first_channel = True):
    X, y = None, None

    tries = 0

    while tries < max_tries:

        start_x = np.random.randint(0, orig_x-output_x+1)
        start_y = np.random.randint(0, orig_y-output_y+1)
        start_z = np.random.randint(0, orig_z-output_z+1)

        y = label[start_x: start_x+output_x,
                  start_y: start_y+output_y,
                  start_z: start_z+output_z]

        y = keras.utils.to_categorical(y, num_classes= num_classes)

        bgrd_ratio = np.sum(y[:,:,:, 0]) / (output_x*output_y*output_z)

        tries += 1

        if bgrd_ratio < background_threshold:
            X = np.copy(image[start_x: start_x + output_x,
                              start_y: start_y + output_y,
                              start_z: start_z + output_z,:])

            if first_channel:
                X = np.moveaxis(X, -1, 0)

                y = np.moveaxis(y, -1, 0)




            return X, y

    print(f"Tried {tries} times to find a sub-volume. Giving up...")
    return None, None

def standardize(image):

    standardized_image = np.zeros(image.shape)

    for c in range(image.shape[0]):
        for z in range(image.shape[3]):

            image_slice = image[c,:,:,z]
            centered = image_slice - image_slice.mean()

            if np.std(centered) != 0:
                centered_scaled = centered / centered.std()

            standardized_image[c, :, :, z] = centered_scaled

    return standardized_image
