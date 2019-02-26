import os
import numpy as np
import numpy.random as rd
from tqdm import tqdm
import cv2


ONE_HOT = np.array(['American Shorthair', 'Angora', 'Ashera', 'British Shorthair',
                    'Exotic', 'Himalayan', 'Maine Coon', 'Persian', 'Ragdoll', 'Siamese', 'Sphynx'])


def one_hot(arg_list, classes):
    if set(arg_list).issubset(set(classes)):
        one_hot_list = []

        for arg in arg_list:
            one_hot_df = []

            for feature in classes:

                if arg == feature:
                    one_hot_df.append(1.)
                elif arg != feature:
                    one_hot_df.append(0.)

            one_hot_list.append(one_hot_df)

        one_hot_list = np.array(one_hot_list)

        return one_hot_list


def open_image_jpeg(image_path):
    image = cv2.imread(image_path, 1)
    image = np.array(image, dtype=float)
    image = image/225

    return image


def create_image_label(image_path, one_hot_classes):
    label_name = image_path.split('\\')[-2]

    label_list = one_hot([label_name], one_hot_classes)[0]

    return label_list


def create_df(image_path, one_hot_classes):
    data_frame = [open_image_jpeg(image_path), create_image_label(
        image_path, one_hot_classes)]

    return data_frame


def vectorize_data(folder_dir, one_hot_classes=ONE_HOT, train_to_test_ratio=0.87):
    data_holder = []
    train_data = []
    test_data = []

    walk = [i for i in os.walk(folder_dir)][1:]

    for root, _, files in tqdm(walk, desc='folders'):

        for image in tqdm(files, desc='images', leave=False):

            image_path = os.path.join(root, image)
            image_df = create_df(image_path, one_hot_classes)
            data_holder.append(image_df)

    rd.shuffle(data_holder)

    training_number = round(len(data_holder)*train_to_test_ratio)

    train_data = np.array(data_holder[:training_number])
    test_data = np.array(data_holder[training_number:])

    save_folder_path = os.path.split(folder_dir)[0]
    save_folder_path = os.path.join(save_folder_path, 'vectorized data/')

    np.save(save_folder_path+'train_data', train_data)
    np.save(save_folder_path+'test_data', test_data)

    return train_data, test_data
