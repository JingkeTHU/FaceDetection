import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

# input: filename used to match file path
# default


def f1(): return tf.one_hot(0, 5)


def f2(): return tf.one_hot(1, 5)


def f3(): return tf.one_hot(2, 5)


def f4(): return tf.one_hot(3, 5)


def f5(): return tf.one_hot(4, 5)


def read_and_decode(file="./test*.tfrecords"):
    filenames = tf.train.match_filenames_once(file)
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'path': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string)
        }
    )
    record_image = tf.decode_raw(features['image'], tf.uint8)
    record_image = tf.reshape(record_image, [400, 400, 1])
    float_image = tf.image.convert_image_dtype(record_image, tf.float32)
    record_label_str = tf.cast(features['label'], tf.string)

    one_hot_label = tf.case({tf.equal(record_label_str, 'My_Face'): f1, tf.equal(record_label_str, 'Her_Face'): f2,
                            tf.equal(record_label_str, 'Dan_Face'): f3, tf.equal(record_label_str, 'Qin_Face'): f4,
                            tf.equal(record_label_str, 'Mom_Face'): f5})
    return float_image, one_hot_label, record_label_str


def convolution_model(float_image_batch):
    conv2d_layer_one = tf.contrib.layers.convolution2d(
        float_image_batch,
        num_outputs=32,  # The number of filters to generate
        kernel_size=(5, 5),  # It's only the filter height and width.
        activation_fn=tf.nn.relu,
        stride=(2, 2),
        trainable=True)
    pool_layer_one = tf.nn.max_pool(conv2d_layer_one,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')

    print()
    conv2d_layer_two = tf.contrib.layers.convolution2d(
        pool_layer_one,
        num_outputs=64,  # More output channels means an increase in the number of filters
        kernel_size=(5, 5),
        activation_fn=tf.nn.relu,
        stride=(1, 1),
        trainable=True)
    pool_layer_two = tf.nn.max_pool(conv2d_layer_two,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')

    flattened_layer_two = tf.reshape(
        pool_layer_two,
        [
            batch_size,  # Each image in the image_batch
            -1  # Every other dimension of the input
        ])

    hidden_layer_three = tf.contrib.layers.fully_connected(
        flattened_layer_two,
        512,
        activation_fn=tf.nn.relu
    )
    # Dropout some of the neurons, reducing their importance in the model
    hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.3)

    # The output of this are all the connections between the previous layers and the 120 different dog breeds
    # available to train on.
    final_fully_connected = tf.contrib.layers.fully_connected(
        hidden_layer_three,
        5,  # Number of people in the training set
    )
    out_put = final_fully_connected
    return out_put


# training
img, label, record_label_str = read_and_decode("./test*.tfrecords")

min_after_dequeue = 10
batch_size = 1
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [img, label], batch_size=batch_size, num_threads=3, capacity=capacity, min_after_dequeue=min_after_dequeue)
X = tf.placeholder(tf.float32, [batch_size, 400, 400, 1])
Y = tf.placeholder(tf.float32, [batch_size, 5])
with tf.name_scope('predict'):
    predict = convolution_model(image_batch)
with tf.name_scope('cost_func'):
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
# optimizer: learning rate - 1e-2
optimizer = tf.train.AdamOptimizer(1e-2).minimize(cost_func)

sess = tf.Session()
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

writer = tf.summary.FileWriter("E://Python//Facedetection//test", sess.graph)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
saver = tf.train.Saver([cost_func, predict])
threads = tf.train.start_queue_runners(sess=sess)
for i in range(500):
    print(i)
    if i % 100 == 0:
        saver.save(sess, './my_model', global_step=i)
    rs = sess.run(merged)
    writer.add_summary(rs, i)
    image_got, label_got = sess.run([image_batch, label_batch])
    x = image_got
    y = label_got
    print(sess.run(record_label_str))
    print(sess.run(predict))
    print(label_got)
    _, cost = sess.run([optimizer, cost_func], feed_dict={X: x, Y: y})
    print(cost)





'''
for i in range(2000):
    # display an image after each 0.2 s + processing time

    time.sleep(0.2)
    image = sess.run(record_image)
    label = sess.run(record_label)

    a = np.shape(image)
    c = np.zeros((a[0], a[1]), dtype=np.int32)
    for row in range(a[0]):
        for line in range(a[1]):
            c[row][line] = int(image[row][line])

    print(label)
    plt.imshow(c)
    plt.show()

    # create a saver to save the model with best performance


    # start to establish the convolution model



    # start to establish training model





#print(image)


coord.request_stop()
coord.join(threads)
'''
