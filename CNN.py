from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
import time
import numpy as np


def classifying_filepath():
    file_path = glob.glob('./Data_set/*/*.jpg')
    #print(file_path)

    # establish two empty dict to save path+label
    image_path = []
    image_label = []
    # split the original file_path to a tuple which contains file label and file path
    image_path_with_label = list(map(lambda filename: (filename.split("\\")[1], filename), file_path))

    for num in range(len(image_path_with_label)):
        image_label.append(image_path_with_label[num][0])
        image_path.append(image_path_with_label[num][1])
        #print(image_path_with_label[1][0])
    return image_label, image_path


def test_filepath():
    file_path = glob.glob('./Data_set/Test_Face/*/*.jpg')
    # print(file_path)

    # establish two empty dict to save path+label
    image_path = []
    image_label = []
    # split the original file_path to a tuple which contains file label and file path
    image_path_with_label = list(map(lambda filename: (filename.split("\\")[1], filename), file_path))

    for num in range(len(image_path_with_label)):
        image_label.append(image_path_with_label[num][0])
        image_path.append(image_path_with_label[num][1])
        # print(image_path_with_label[1][0])
    return image_label, image_path


# CNN model defined here
def convolution_model(float_image_batch):
    batch_size = 10
    print('first')
    image_sequence = tf.image.convert_image_dtype(float_image_batch, tf.float32)
    float_image_batch = tf.reshape(image_sequence, [-1, 800, 800, 1])
    conv2d_layer_one = tf.contrib.layers.convolution2d(
        float_image_batch,
        num_outputs=12,  # The number of filters to generate
        kernel_size=(7, 7),  # It's only the filter height and width.
        activation_fn=tf.nn.relu,
        stride=(2, 2))
    print('second')
    pool_layer_one = tf.nn.max_pool(conv2d_layer_one,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')

    # input size of the image : 400*400*24
    conv2d_layer_two = tf.contrib.layers.convolution2d(
        pool_layer_one,
        num_outputs=24,  # More output channels means an increase in the number of filters
        kernel_size=(5, 5),
        activation_fn=tf.nn.relu,
        stride=(1, 1))
    pool_layer_two = tf.nn.max_pool(conv2d_layer_two,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 3, 3, 1],
                                    padding='SAME')

    # input size of the image : 200*200*48
    conv2d_layer_three = tf.contrib.layers.convolution2d(
        pool_layer_two,
        num_outputs=36,  # More output channels means an increase in the number of filters
        kernel_size=(5, 5),
        activation_fn=tf.nn.relu,
        stride=(1, 1))
    pool_layer_three = tf.nn.max_pool(conv2d_layer_three,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')

    # input size of the image : 100*100*64
    print('here')
    flattened_layer_two = tf.reshape(
        pool_layer_three,
        [
            batch_size,  # Each image in the image_batch
            -1  # Every other dimension of the input
        ])

    hidden_layer_three = tf.contrib.layers.fully_connected(
        flattened_layer_two,
        128,
        activation_fn=tf.nn.relu
    )
    # Dropout some of the neurons, reducing their importance in the model
    hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.5)

    # The output of this are all the connections between the previous layers and the 120 different dog breeds
    # available to train on.
    final_fully_connected = tf.contrib.layers.fully_connected(
        hidden_layer_three,
        5,  # Number of people in the training set
    )
    out_put = final_fully_connected
    return out_put


# load and display the image according to the given path
def image_load_display_preprocessing(path="./Her_Face/IMG_4302.jpg"):

    img_empty = Image.new('RGBA', (800, 800), color='white')
    img1 = Image.open(path)
    # generate a random number to rotate the image
    rotate_num = np.random.randint(0, 360, dtype='int32')
    #print(rotate_num)
    img1 = img1.rotate(rotate_num)
    # filter the image
    # generate a random number to blur the image
    blur_num = np.random.randint(0, 2, dtype='int32')
    #print(blur_num)
    img1 = img1.filter(ImageFilter.GaussianBlur(radius=blur_num))
    #
    width, height = img1.size
    width_mul = 800. / width
    height_mul = 800. / height

    if width < height:
        img1 = img1.resize((round(height_mul * width)-5, 800), Image.BILINEAR)
    else:
        img1 = img1.resize((800, (round(width_mul * height)-5)), Image.BILINEAR)

    img1.save("temp.jpg")
    img1 = Image.open('./temp.jpg')
    img_empty.paste(img1)
    # paste the loaded image to empty image to get a 800*800 image
    plt.figure("New Face")

    img_empty = img_empty.convert('L')

    plt.imshow(img_empty)
    plt.show()
    # add sleep time before each time of training to show the process
    #time.sleep(0.1)
    img_empty = np.asarray(img_empty)
    return img_empty


def generate_test_data(test_label, test_path):
    # dataset has 143 images, batch_size = 10
    image_sequence = []
    label_sequence = []
    print(test_path)

    for i in range(len(test_path)):
        image_temp = test_path[i]
        test_temp = test_label[i]

        test = image_load_display_preprocessing(image_temp)
        label = test_temp
        if label == 'My_Face':
            one_hot_label = [1, 0, 0, 0, 0]
        elif label == 'Her_Face':
            one_hot_label = [0, 1, 0, 0, 0]
        elif label == 'Don_Face':
            one_hot_label = [0, 0, 1, 0, 0]
        elif label == 'Mom_Face':
            one_hot_label = [0, 0, 0, 1, 0]
        else:
            one_hot_label = [0, 0, 0, 0, 1]

        image_sequence.append(test)
        label_sequence.append(one_hot_label)

    np_image_sequence = np.asarray(image_sequence)
    np_label_sequence = np.asarray(label_sequence)
    return np_image_sequence, np_label_sequence


# pick num = batch_size images and load for training
def generate_data(image_label, image_path):
    # dataset has 143 images, batch_size = 10
    random_begining = np.random.randint(0, 121)
    image_sequence = []
    label_sequence = []
    print(image_path)

    for i in range(10):
        image_temp = image_path[random_begining+i]
        label_temp = image_label[random_begining+i]

        image = image_load_display_preprocessing(image_temp)
        label = label_temp
        if label == 'My_Face':
            one_hot_label = [1, 0, 0, 0, 0]
        elif label == 'Her_Face':
            one_hot_label = [0, 1, 0, 0, 0]
        elif label == 'Don_Face':
            one_hot_label = [0, 0, 1, 0, 0]
        elif label == 'Mom_Face':
            one_hot_label = [0, 0, 0, 1, 0]
        else:
            one_hot_label = [0, 0, 0, 0, 1]

        image_sequence.append(image)
        label_sequence.append(one_hot_label)

    np_image_sequence = np.asarray(image_sequence)
    np_label_sequence = np.asarray(label_sequence)
    return np_image_sequence, np_label_sequence


# parameters define
batch_size = 10
image_label, image_path = classifying_filepath()
test_label, test_path = test_filepath()
test_image_sequence, test_label_sequence = generate_test_data(test_label, test_path)
# define placeholder
X = tf.placeholder(tf.float32, [batch_size, 800, 800, 1])
Y = tf.placeholder(tf.float32, [batch_size, 5])
# tensor flow
predict = convolution_model(X)
cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
optimizer = tf.train.AdamOptimizer(1e-2).minimize(cost_func)

saver = tf.train.Saver(max_to_keep=4)
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    # forbid the alteration to the graph
    sess.graph.finalize()
    for i in range(500):
        time_start = time.time()
        print(i)
        # in each loop, generate a set of image data
        if i % 50 == 0:
            saver.save(sess, "./Model_saved/My_model", global_step=i)
            print('model saved')
        image_sequence, label_sequence = generate_data(image_label, image_path)
        image_sequence = np.array(image_sequence)
        image_sequence = image_sequence.reshape([-1, 800, 800, 1])
        _, cost = sess.run([optimizer, cost_func], feed_dict={X: image_sequence, Y: label_sequence})
        if i % 50 == 0:
            writer = tf.summary.FileWriter("E://Python//Facedetection//test", sess.graph)

        # evaluate the model during training

        print('evaluate')
        test_image_sequence = np.array(test_image_sequence)
        test_image_sequence = test_image_sequence.reshape([-1, 800, 800, 1])
        _, test_cost = sess.run([optimizer, cost_func], feed_dict={X: test_image_sequence, Y: test_label_sequence})
        print('cost function', test_cost)
        print(' ')
        time_end = time.time()
        print('time cost', time_end - time_start)

sess.close()
del sess

