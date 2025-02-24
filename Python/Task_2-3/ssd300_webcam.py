from pathlib import Path
import time
from tensorflow.keras import backend as K
import numpy as np
from models.keras_ssd300 import ssd_300
import cv2
from skimage.transform import resize as imresize

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


############################## MODEL ##########################################
# Copyright 2018 Pierluigi Ferrari (SSD keras port)

# Set the network image size.
img_height = 300
img_width = 300

# Build the Keras model
K.clear_session() # Clear previous models from memory.
model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], 
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=10,
                nms_max_output_size=400)

# Load the trained weights into the model.
weights_path = Path('weights/VGG_VOC0712_SSD_300x300_iter_120000.h5')
model.load_weights(str(weights_path), by_name=True)


classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

###############################################################################



##################################### TASK ####################################
#
# NOTE: The video player can be stopped by selecting the video player window
#       and pressing 'q'
#
# Your task:
#   1. Read a frame from camera using cam.read
#      If no webcam is availabe, use run.gif similarly as in Task 1 or any other suitable video.
#
#   2. Mirror the frame using cv2.flip (this can be omitted in case your using a video clip)
#
#   3. Make a resized copy of the frame for the network.
#      Use the imported function "imresize" and
#      size (img_height, img_width) (defined above).
#      We also want to preserve the original range (0-255),
#      therefore set the additional parameter "preserve_range"
#      to True.
#
#   4. You need an additional first dimension (batch_size, now 1)
#      to feed the image to the network. This can be achieved with:
#      image = np.array([image])
#
#   5. Use the resized input image to get predictions from the network with model.predict
#      Predictions are in the following form:
#      [class_id, confidence, xmin, ymin, xmax, ymax]
#      These predictions should also be thresholded according to
#      the prediction confidence. Anything above 0
#      (i.e omit 0 confidence values) should be used as a valid prediction.
#
#   6. The bounding box coordinates are for the resized image
#      (size: img_height, img_width). Transform these coordinates to the original
#      frame by multiplying x coordinates with the ratio of original and reshaped widths,
#      and y coordinates with the ratio of heights.
#      Remember to also convert these to integer-type.
#
#   7. Use cv2.rectangle to insert a bounding box to the original frame.
#
#   8. Use cv2.putText to insert the predicted class label to the bounding box.
#      Class labels are defined above, and can be indexed with the class_id
#      in the predictions.
#
###############################################################################


# Create a camera instance, options for prerecorded videos also
#cam = cv2.VideoCapture('run.gif')
#cam = cv2.VideoCapture('walking.mp4')
cam = cv2.VideoCapture(0)

# Check if instantiation was successful
if not cam.isOpened():
    raise Exception("Could not open camera")
    
confidence_threshold = 0.0

while True:
    start = time.time() 
    #img = np.zeros((480, 640, 3))

    ##-your-code-starts-here-##
    # Task 1: Read a frame
    x, frame = cam.read()  
    if not x:
        break

    # Task 2: Mirror the frame
    frame = cv2.flip(frame, 1)  

    # Task 3: Resize copy of the previous frame
    resize = imresize(frame, (img_height, img_width), preserve_range=True)

    # Task 4: added first dimension
    input = np.array([resize])

    # Task 5: get predictions from the network
    prediction = model.predict(input)

    # Task 6: process predictions and bounding box coordinates
    for box_cord in prediction[0]:
        
        confidence = box_cord[1]
        if confidence > confidence_threshold:
            x_min = int(box_cord[2] * frame.shape[1] / img_width)
            y_min = int(box_cord[3] * frame.shape[0] / img_height)
            x_max = int(box_cord[4] * frame.shape[1] / img_width)
            y_max = int(box_cord[5] * frame.shape[0] / img_height)

            # Task 7: draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Task 8: class label
            label = f"{classes[int(box_cord[0])]}: {confidence:.2f}"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    ##-your-code-ends-here-##

    # Insert FPS/quit text and show image
    fps = "{:.0f} FPS".format(1 / (time.time() - start))
    frame = cv2.putText(frame, fps, (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                      fontScale=0.8, color=(0, 255, 255))
    frame = cv2.putText(frame, 'Press q to quit', (440, 20), cv2.FONT_HERSHEY_SIMPLEX,
                      fontScale=0.8, color=(0, 0, 255))
    cv2.imshow('Video feed', frame)

    # q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()   
cv2.destroyAllWindows()

