#!/Users/arthur/Documents/Coding/vgg/venv/bin/python
import numpy as np
import tensorflow as tf
import class_names as classes
from scipy.misc import imread, imresize

# We will not exactly ROI pool, but crop the Conv Feature Map using the proposals
# and resizing to 14x14, then it will be 2x2 pooled in order to get the required 7x7 map
# Proposals are x_min,y_min,x_max,y_max
def roi_pool(feature_map, proposals):
    # Boxes are in shape [y1, x1, y2, x2], as required for crop_and_resize.
    # All x,y are normalized to [0,1]
    crops = tf.image.crop_and_resize(
            feature_map, bboxes, batch_ids,
            [14, 14], name="crops"
        )

    # Returns resized 14x14x512 Tensor. 

# Pyython Ref: https://github.com/tryolabs/luminoth/blob/master/luminoth/models/fasterrcnn/roi_pool.py
# VGG-D model from https://arxiv.org/pdf/1409.1556.pdf.
# Weights from https://www.cs.toronto.edu/~frossard/post/vgg16/#weights
# https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/
def model(features, labels, mode):
    K = 9
    weights = np.load("vgg16_weights.npz")
    # Make sure the mean RGB value is subtracted from the pixels.
    input_layer = tf.reshape(features["img"], [1, 224, 224, 3])
    # Also input features['boxes'] which will go to the RPN for training and recognition.

    # VGG - D 
    # Naming scheme: conv<receptive field size>-<num channels>-<layer num>
    conv_1 = tf.layers.conv2d(name="conv3-64-1", inputs=input_layer, filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv1_1_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv1_1_b"]))
    conv_2 = tf.layers.conv2d(name="conv3-64-2", inputs=conv_1, filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv1_2_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv1_2_b"]))
    pool1 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=[2, 2], strides=2, padding="same")

    conv_3 = tf.layers.conv2d(name="conv3-128-1", inputs=pool1, filters=128, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv2_1_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv2_1_b"]))
    conv_4 = tf.layers.conv2d(name="conv3-128-2", inputs=conv_3, filters=128, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv2_2_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv2_2_b"]))
    pool2 = tf.layers.max_pooling2d(inputs=conv_4, pool_size=[2, 2], strides=2, padding="same")

    # These layers (3_1) and up are trainable. (Optional to make it all untrainable if we testing on same dataset)
    conv_5 = tf.layers.conv2d(name="conv3-256-1", inputs=pool2, filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv3_1_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv3_1_b"]))
    conv_6 = tf.layers.conv2d(name="conv3-256-2", inputs=conv_5, filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv3_2_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv3_2_b"]))
    conv_7 = tf.layers.conv2d(name="conv3-256-3", inputs=conv_6, filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv3_3_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv3_3_b"]))
    pool3 = tf.layers.max_pooling2d(inputs=conv_7, pool_size=[2, 2], strides=2, padding="same")

    conv_8 = tf.layers.conv2d(name="conv3-512-1", inputs=pool3, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv4_1_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv4_1_b"]))
    conv_9 = tf.layers.conv2d(name="conv3-512-2", inputs=conv_8, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv4_2_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv4_2_b"]))
    conv_10 = tf.layers.conv2d(name="conv3-512-3", inputs=conv_9, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv4_3_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv4_3_b"]))
    pool4 = tf.layers.max_pooling2d(inputs=conv_10, pool_size=[2, 2], strides=2, padding="same")

    conv_11 = tf.layers.conv2d(name="conv3-512-4", inputs=pool4, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv5_1_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv5_1_b"]))
    conv_12 = tf.layers.conv2d(name="conv3-512-5", inputs=conv_11, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv5_2_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv5_2_b"]))
    conv_13 = tf.layers.conv2d(name="conv3-512-6", inputs=conv_12, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu,
                              kernel_initializer=tf.constant_initializer(weights["conv5_3_W"]),
                              bias_initializer=tf.constant_initializer(weights["conv5_3_b"]))

    # The output of Conv5_1 is where we branch for the Region Proposal Network.
    conv_rpn_1 = tf.layers.conv2d(name="conv_rpn_1", inputs=conv_11, filters=512, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
    conv_rpn_objectness = tf.layers.conv2d(name="conv_rpn_objectness", inputs=conv_11, filters=2*K, kernel_size=[1,1], padding="same", activation=tf.nn.relu)
    # X_center, Y_Center, Width, Height
    conv_rpn_regression = tf.layers.conv2d(name="conv_rpn_regression", inputs=conv_11, filters=4*K, kernel_size=[1,1], padding="same", activation=tf.nn.relu)
    # Have these do cross enthropy (objectness) and Smooth L1 (regression) losses then feed into an optimizer...
    # https://github.com/tensorflow/tensorflow/issues/15773
    # https://stackoverflow.com/questions/46291253/tensorflow-sigmoid-and-cross-entropy-vs-sigmoid-cross-entropy-with-logits


    # Branch for Img classification
    pool5 = tf.layers.max_pooling2d(inputs=conv_13, pool_size=[2, 2], strides=2, padding="same") # Replaced with ROI?, that is compatiable with
    # The first FC layer. so H=W=7
    flatten = tf.layers.flatten(inputs=pool5)
    fc1 = tf.layers.dense(name="fc1", inputs=flatten, units=4096, kernel_initializer=tf.constant_initializer(weights["fc6_W"]),
                              bias_initializer=tf.constant_initializer(weights["fc6_b"]))
    fc2 = tf.layers.dense(name="fc2", inputs=fc1, units=4096, kernel_initializer=tf.constant_initializer(weights["fc7_W"]),
                              bias_initializer=tf.constant_initializer(weights["fc7_b"]))
    logits = tf.layers.dense(name="logits", inputs=fc2, units=1000,kernel_initializer=tf.constant_initializer(weights["fc8_W"]),
                              bias_initializer=tf.constant_initializer(weights["fc8_b"])) # remove this.

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor") # Remove this?
    }
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


# I am passing a numpy input, i need to pass a TF input.
# def predict_fn():
#     filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("laska.png"))
#     image_reader = tf.WholeFileReader()
#     _, image_file = image_reader.read(filename_queue)
#     image = tf.image.decode_png(image_file, channels=3)
#     resized_image = tf.image.resize_images(image, [244, 244])
#     print "pred"
#     tf.Print(resized_image, [resized_image])
#     return tf.estimator.inputs.numpy_input_fn(
#       x={"img": resized_image},
#       shuffle=False)

def predict_fn():
    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))
    # Broad cast remove the mean:
    img = img1 - [123.68, 116.779, 103.939]
    # Zero mean the image
    #mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    return tf.estimator.inputs.numpy_input_fn(
      x={"img": img},
      shuffle=False)

def main():
    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224)) # I think it might need to be 1,224,224(3?)
    # Broad cast remove the mean:
    mimg = img1 - [123.68, 116.779, 103.939]
    img = mimg.reshape(1, 224, 224, 3)
    # Zero mean the image
    #mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"img": img }, #And also ROI in images.
      shuffle=False)

    classifier = tf.estimator.Estimator(model_fn=vgg_model, model_dir="model_tmp")
    vals = classifier.predict(input_fn=input_fn)

    print next(vals)
    print "Done"
    #sess = tf.InteractiveSession()

# Try creating a TF Dataset from the Pascal:
# https://www.tensorflow.org/guide/datasets_for_estimators
if __name__ == '__main__':
    main()
    # weights = np.load("vgg16_weights.npz")
    # keys = sorted(weights.keys())
    # for k,v in enumerate(keys):
    #     print 86016