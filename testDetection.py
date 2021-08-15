import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import os
from object_detection.utils import label_map_util

# Allow us to transform detection to a class label
from object_detection.utils import visualization_utils as viz_utils

# Create model from checkpoint and config file
from object_detection.builders import model_builder
import cv2
import numpy as np

# Folder containing tensorflow files (Installed via github)
TENSORFLOW_PATH = "../RealTimeObjectDetection/Tensorflow"

# Folder containing all own files (Images, Annotations, Models, etc..)
WORKSPACE_PATH = "Workspace"

# Tensorflow script to generate tf Record
SCRIPTS_PATH = TENSORFLOW_PATH + "/scripts"
APIMODEL_PATH = TENSORFLOW_PATH + "/models"

ANNOTATION_PATH = WORKSPACE_PATH + "/annotations"
IMAGE_PATH = WORKSPACE_PATH + '/images'
MODEL_PATH = WORKSPACE_PATH + '/my_models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + "/pre-trained-models"

# Change this
CUSTOM_MODEL_NAME = 'over_mobilenet'
CONFIG_PATH = MODEL_PATH + f'/{CUSTOM_MODEL_NAME}/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + f'/{CUSTOM_MODEL_NAME}/'


# Load configurations from pipeline.config file
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
# Build model, passing in your configs file
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Create the labels index
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

# Restore checkpoint by passing in the detection_model
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
# Change name accordingly. Take the latest checkpoint in the MODEL_DIR (My latest is ckpt-2.index so i use ckpt-2)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-1')).expect_partial()

# Create the detection function
@tf.function
def detect_fn(image):
    # Preprocess the image (Resize)
    image, shapes = detection_model.preprocess(image)
    print(image.shape)
    # Make prediction
    prediction_dict = detection_model.predict(image, shapes)
    # Bounding boxes are drawn
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


SAMPLE_IMAGE = IMAGE_PATH + f"/train/1.png"
image = cv2.imread(SAMPLE_IMAGE)
image_np = np.array(image)

# Convert np array to tf tensor
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
# Make the detections. This returns the bounding boxes
detections = detect_fn(input_tensor)
#Get the number of detections
num_detections = int(detections.pop('num_detections'))

detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}

detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1

# Make a copy just in case something goes wrong
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            # Raw image np array
            image_np_with_detections,
            # bounding boxes cooridnates
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.0,
            agnostic_mode=False)

cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
cv2.waitKey()