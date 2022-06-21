import os

import csv
import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = './train_data'
TEST_DIR = './test_data'
IMAGES = './images/images'
IMG_SIZE = 128
LR = 1e-3

print(int(len(os.listdir((IMAGES))) * 0.8))

pokemon_dict = {}
with open('pokemon.csv', newline='') as csvfile:
    pokereader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(pokereader, None)
    for row in pokereader:
        if len(row) == 2:
            pokemon_dict[row[0]] = [row[1]]
        else:
            pokemon_dict[row[0]] = [row[1], row[2]]

pokemon_types = ['Normal', 'Fire', 'Water', 'Grass', 'Flying', 'Fighting', 'Poison', 'Electric', 'Ground', 'Rock',
                 'Psychic', 'Ice', 'Bug', 'Ghost', 'Steel', 'Dragon', 'Dark', 'Fairy']


def label_img(img):
    word_label = img.split('.')[0].lower()
    label_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in pokemon_dict[word_label]:
        for index in range(0, len(pokemon_types)):
            if pokemon_types[index] == i:
                label_arr[index] = 1
    return label_arr


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(IMAGES)[:int(len(os.listdir((IMAGES))) * 0.8)]):
        label = label_img(img)
        path = os.path.join(IMAGES, img)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    print("create_train_data done")
    return training_data


train_data = create_train_data()


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(IMAGES)[int(len(os.listdir((IMAGES))) * 0.8):]):
        path = os.path.join(IMAGES, img)
        label = label_img(img)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), np.array(label)])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    print("process_test_data done")
    return testing_data


test_data = process_test_data()

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf

tf.compat.v1.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 4, activation='relu')
convnet = max_pool_2d(convnet, 4)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = fully_connected(convnet, 8192, activation='relu')

convnet = fully_connected(convnet, 18, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format("pokemon")):
    model.load("pokemon")
    print('model loaded!')

train = train_data
test = test_data

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=15, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id="pokemon")

import matplotlib.pyplot as plt

test_data = process_test_data()

fig = plt.figure()

sum = [0,0]
for num, data in enumerate(test_data[:81]):
    actual_type = data[1]
    img_data = data[0]

    y = fig.add_subplot(9, 9, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
    model_out = model.predict([data])[0]
    # 'Normal', 'Fire', 'Water', 'Grass', 'Flying', 'Fighting', 'Poison', 'Electric', 'Ground', 'Rock',
    # 'Psychic', 'Ice', 'Bug', 'Ghost', 'Steel', 'Dragon', 'Dark', 'Fairy']

    if np.max(model_out) == model_out[0]:
        if actual_type[0] == 1:
            sum[0]+=1
        str_label = 'Normal'
    elif np.max(model_out) == model_out[1]:
        if actual_type[1] == 1:
            sum[0]+=1
        str_label = 'Fire'
    elif np.max(model_out) == model_out[2]:
        if actual_type[2] == 1:
            sum[0]+=1
        str_label = 'Water'
    elif np.max(model_out) == model_out[3]:
        if actual_type[3] == 1:
            sum[0]+=1
        str_label = 'Grass'
    elif np.max(model_out) == model_out[4]:
        if actual_type[4] == 1:
            sum[0]+=1
        str_label = 'Flying'
    elif np.max(model_out) == model_out[5]:
        if actual_type[5] == 1:
            sum[0]+=1
        str_label = 'Fighting'
    elif np.max(model_out) == model_out[6]:
        if actual_type[6] == 1:
            sum[0]+=1
        str_label = 'Poison'
    elif np.max(model_out) == model_out[7]:
        if actual_type[7] == 1:
            sum[0]+=1
        str_label = 'Electric'
    elif np.max(model_out) == model_out[8]:
        if actual_type[8] == 1:
            sum[0]+=1
        str_label = 'Ground'
    elif np.max(model_out) == model_out[9]:
        if actual_type[9] == 1:
            sum[0]+=1
        str_label = 'Rock'
    elif np.max(model_out) == model_out[10]:
        if actual_type[10] == 1:
            sum[0]+=1
        str_label = 'Psychic'
    elif np.max(model_out) == model_out[11]:
        if actual_type[11] == 1:
            sum[0]+=1
        str_label = 'Ice'
    elif np.max(model_out) == model_out[12]:
        if actual_type[12] == 1:
            sum[0]+=1
        str_label = 'Bug'
    elif np.max(model_out) == model_out[13]:
        if actual_type[13] == 1:
            sum[0]+=1
        str_label = 'Ghost'
    elif np.max(model_out) == model_out[14]:
        if actual_type[14] == 1:
            sum[0]+=1
        str_label = 'Steel'
    elif np.max(model_out) == model_out[15]:
        if actual_type[15] == 1:
            sum[0]+=1
        str_label = 'Dragon'
    elif np.max(model_out) == model_out[16]:
        if actual_type[16] == 1:
            sum[0]+=1
        str_label = 'Dark'
    elif np.max(model_out) == model_out[17]:
        if actual_type[17] == 1:
            sum[0]+=1
        str_label = 'Fairy'
    sum[1]+=1

    y.imshow(orig)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

print("Guessed % = " + str(sum[0]/sum[1]) + " Random guessing % = " + str(1/18))
plt.show()
