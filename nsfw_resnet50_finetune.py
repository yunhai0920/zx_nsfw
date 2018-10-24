# -*- coding:utf-8 -*-
"""
@author:HuangJie
@time:18-9-29 下午4:31

"""
import os
import glob
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
BAT_SIZE = 32
NB_EPOCH = 10
IM_WIDTH, IM_HEIGHT, CHANNEL = 224, 224, 3


def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


def resnet50_train(train_dir, val_dir, output_dir, batch_size, plot):
    nb_train_samples = get_nb_files(train_dir)
    nb_classes = len(glob.glob(train_dir + "/*"))
    nb_val_samples = get_nb_files(val_dir)
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BAT_SIZE)

    validation_datagen = ImageDataGenerator()
    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BAT_SIZE)

    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    predictions = Dense(nb_classes, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=SGD(lr=0.0001, decay=1e-5, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history_fit = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples//batch_size,
        epochs=NB_EPOCH,
        validation_data=validation_generator,
        validation_steps=nb_val_samples//batch_size)
    model.save(output_dir)
    if plot:
        plot_training(history_fit)


if __name__ == '__main__':
    train_file = '/home/hj/ZX_DL/DL_NSFW/zxsoft_train/zxnsfw-3/train'
    val_file = '/home/hj/ZX_DL/DL_NSFW/zxsoft_train/zxnsfw-3/val'
    output_model_file = '/home/hj/PycharmProjects/zx_nsfw/result/resnet50-ft.h5'
    batchsize = BAT_SIZE
    resnet50_train(train_file, val_file, output_model_file, batchsize, plot=True)
