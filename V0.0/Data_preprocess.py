import tensorflow as tf
import glob
from collections import defaultdict
from itertools import groupby



def write_records_file(dataset, record_location, sess):
    # parameters initialization
    writer = None
    current_index = 0

    for labels, images in dataset.items():
        print(labels)
        for image in images:
            print(image)
            if current_index % 10 == 0:
                if writer:
                    writer.close()
                record_filename = "{record_location}-{current_index}.tfrecords".format(
                    record_location=record_location,
                    current_index=current_index)
                writer = tf.python_io.TFRecordWriter(record_filename)
            current_index = current_index + 1

            image_data = tf.read_file(image)

            # try to decode the JPEG images
            try:
                image_decoded = tf.image.decode_jpeg(image_data)
            except:
                print(image)
                continue

            # generate a random number to decide how many times image preprocessing will be imposed on this image
            # resize all the images to (200,200)
            croped = tf.image.resize_image_with_crop_or_pad(image_decoded, 400, 400)
            random = tf.floor(3 * tf.abs(tf.random_normal([1, 1])))
            times = sess.run(random)
            print(times)
            # start to to image preprocessing
            for prepro_time in range(int(times)+1):
                # generate images by adjust color
                color_adjust = tf.image.adjust_brightness(croped, sess.run(tf.abs(tf.random_normal([1, 1])/10)))
                grayscale_image = tf.image.rgb_to_grayscale(color_adjust)

                # generate a random number between 1-3 to decide how or whether flip the image
                flip = sess.run(tf.random_uniform([1, 1], 0, 3, dtype=tf.int32))
                if flip == 0:
                    grayscale_image = tf.image.flip_left_right(grayscale_image)
                if flip == 2:
                    grayscale_image = tf.image.flip_up_down(grayscale_image)

                # convert the image to uint8
                image_bytes = sess.run(tf.cast(grayscale_image, tf.uint8)).tobytes()

                image_label = labels.encode("utf-8")
                image_path = image.encode("utf-8")

                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                    'path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
                 }))

                writer.write(example.SerializeToString())
    writer.close()


image_filenames = glob.glob("./*/*.jpg")

training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

image_filenames_with_label = list(map(lambda filename: (filename.split("\\")[1], filename), image_filenames))
a = groupby(image_filenames_with_label, lambda x: x[0])
sess = tf.Session()

for image_label, image_path in groupby(image_filenames_with_label, lambda x: x[0]):
    print(image_label)
    print(image_path)
    for i, image in enumerate(image_filenames):
        print(i)
        print(image)
        if i % 5 == 0:
            testing_dataset[image_label].append(image[1])
        else:
            training_dataset[image_label].append(image[1])

print(training_dataset)

write_records_file(testing_dataset, "./test_image", sess)
write_records_file(training_dataset, "./train_image", sess)


