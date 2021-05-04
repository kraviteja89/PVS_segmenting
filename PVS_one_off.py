import numpy as np
import matplotlib.pyplot as plt
import h5py
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, Add
import os
from keras_vggface import VGGFace


def learn_pvs_model(model, pvs_samples, roi_samples, validation_split=0.2, batch_size=32, n_epochs=5, learning_rate=0.0005):
    """
    Fit the keras model for identifying masks for Paravascular spaces
    :param model: The keras model to be learned
    :param pvs_samples: the sample rgb images with the parvascular space
    :param roi_samples: the outputs for the PVS ROI (mask)
    :param validation_split, batch_size, n_epochs, learning_rate: keras model.fit parameters,
    see https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit

    :return: model: keras model with learned parameters
    """
    n_frames, ht, wd = roi_samples.shape
    roi_samples = roi_samples.reshape([n_frames, ht, wd, 1]).astype(float)

    opt = Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    model.fit(pvs_samples, roi_samples, epochs=n_epochs, batch_size=batch_size, validation_split=validation_split)
    return model


def model_from_vgg(last_layer='pool4'):
    """
    returns a neural network with layers upto <last_layer> from vgg16
    with the weights for the vggface layers preloaded
    """
    vgg_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3))
    X = vgg_model.get_layer(last_layer).output
    layer_shape = vgg_model.get_layer(last_layer).output_shape

    n_encoder_layers = int(np.log2(224/layer_shape[2]))

    for n in range(n_encoder_layers):
        X = Conv2DTranspose(int(layer_shape[3]/(2**(n+1))), (3, 3), activation='relu', padding='same', name='deconv'+str(n+1))(X)
        X = UpSampling2D(size=(2, 2), interpolation='bilinear', name='unpool'+str(n+1))(X)

    mask = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='mask')(X)

    custom_model = Model(vgg_model.input, mask)
    return custom_model


def make_color_images(green_data, red_data):
    """
    Make color images of shape (None, 224, 224, 3) from the red channel and green channel two-photon data,
    each of shape (None, 224, 224)
    The red channel image is repeated in the red and blue channels
    """

    n_frames, ht, wd = green_data.shape
    green_data = green_data.reshape([n_frames, ht, wd, 1])
    red_data = red_data.reshape([n_frames, ht, wd, 1])

    colored_data = np.concatenate((red_data, green_data, red_data), axis=-1)
    return colored_data


if __name__ == "__main__":
    # data file location
    animal_ID = 'RK050'
    day_ID = '210425'
    file_num = '002'
    frame_size = 224
    last_layer = 'pool4'  # change this value to alter the last layer of vgg16
    pvs_fname = "datasets//" + animal_ID + '_' + day_ID + '_' + file_num + '_python.mat'
    pvs_file = h5py.File(pvs_fname, 'r')

    #  read the data from matlab file
    green_samples = np.array(pvs_file["image_samples_g"])
    red_samples = np.array(pvs_file["image_samples_r"])

    green_data = np.array(pvs_file["the_data_g"])
    red_data = np.array(pvs_file["the_data_r"])

    roi_samples = np.array(pvs_file["ROI_samples"])
    pvs_file.close()

    roi_samples = roi_samples.reshape([-1, frame_size, frame_size])

    colored_samples = make_color_images(1-green_samples.reshape([-1, frame_size, frame_size]),
                                        red_samples.reshape([-1, frame_size, frame_size]))

    # if learned model exists load the data
    # otherwise create the neural network and learn the weights
    file_name2 = pvs_fname[:-11] + "from_vgg" + last_layer
    if os.path.exists(file_name2):
        print("Model exists....loading from file: " + file_name2)
        model2 = load_model(file_name2)
    else:
        model2 = model_from_vgg(last_layer)
        model2 = learn_pvs_model(model2, colored_samples, roi_samples, n_epochs=75, validation_split=0.0, learning_rate=0.0002)
        model2.save(file_name2)

    #  calculate the ROIs and areas for the all slices and frames
    n_frames, n_slices, ht, wd = green_data.shape
    colored_data = make_color_images(1-green_data.reshape([-1, ht, wd]), red_data.reshape([-1, ht, wd]))
    PVS_ROIs = model2.predict(colored_data) > 0.5
    PVS_ROIs = PVS_ROIs.reshape([n_frames, n_slices, ht, wd]).astype(int)
    PVS_areas = np.sum(PVS_ROIs, axis=(2, 3))

    # plot PVS areas
    fig1 = plt.figure()
    plt.plot(PVS_areas)

    #  plot sample segmented images in each slice
    for n in range(n_slices):
        colored_data1 = make_color_images(green_data[:, n, :, :], red_data[:, n, :, :])
        n1 = np.random.randint(n_frames)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(colored_data1[n1])
        axes[1].imshow(PVS_ROIs[n1, n, :, :])

    plt.show()

    #  save segmented data
    roi_fname = "datasets//" + animal_ID + '_' + day_ID + '_' + file_num + '_vgg16_' + last_layer + '.h5'
    roi_file = h5py.File(roi_fname, 'w')
    roi_file['PVS_areas'] = PVS_areas
    roi_file['PVS_ROIs'] = PVS_ROIs
    roi_file.close()


