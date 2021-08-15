'''
- Files are already split into /train /test folders
- Single class called "head" with id set =1
- Training epochs set to 50
- Model name set to "over_mobilenet"
python train.py -L "{'name': 'head', 'id': 1 }" -E 50 -N "over_mobilenet"

- Files are still in /screenshots. Split with 70% in the training set
python train.py -L "{'name': 'head', 'id': 1 }" -E 50 -N "over_mobilenet" -T yes -S 0.7
'''

import os
import shutil
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import argparse
import ast
import glob
from shutil import move

parser = argparse.ArgumentParser(description='Train object detection model. Place Images + Annotations in workspace/images/train (or test) \n '
                                             'Usage: python tmp.py -L "{"name": "head", "id": 1 }" -E 50 -N "over_mobilenet"')
parser.add_argument('-L', metavar='--labels', type=str, help='str: labels list E.g: "{"name": "head", "id": 1}" separate using ***')
parser.add_argument('-E', metavar='--epochs', type=int, help='int: Number of training epochs')
parser.add_argument('-N', metavar='--name', type=str, help='str: Name of custom model')
# if "yes" means images + annots are in screenshots, will use "trainSize" to split into /train and /test
parser.add_argument('-T', metavar='--split', type=str, nargs='?', default="no", help="'yes' if images are in /screenshots. Will use --trainSize to split to /train /test")
parser.add_argument('-S', metavar='--trainSize', type=float, nargs='?', default=1, help="Used if --split is 'yes'")
parser.add_argument('-M', metavar='--model', type=str, nargs='?', default='ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', help="Specify which pretrained model in Workspace/pre-trained-models to use")
args = parser.parse_args()

labels = []
labelArg = args.L.split('***')
for a in labelArg:
    ll = ast.literal_eval(a)
    labels.append(ll)

CUSTOM_MODEL_NAME = args.N
TRAIN_STEPS = args.E
PRETRAINED_MODEL_NAME = args.M

if args.T == "yes":
    screenshot_folder = "Workspace/images/screenshots"
    train_folder = "Workspace/images/train"
    test_folder = "Workspace/images/test"

    train_portion = args.S
    img_list = glob.glob(screenshot_folder + "/*.png")
    img_list_size = len(img_list)
    train_size = int(img_list_size * train_portion)
    test_size = img_list_size - train_size

    for idx in range(train_size):
        img = img_list[idx]
        img_file = img.split('\\')[1]
        img_number = img_file.split(".")[0]
        annot_file = img_number + ".xml"
        img_filepath = screenshot_folder + f'/{img_file}'
        annot_filepath = screenshot_folder + f'/{annot_file}'

        new_img_filepath = train_folder + f'/{img_file}'
        new_annot_filepath = train_folder + f'/{annot_file}'
        move(img_filepath, new_img_filepath)
        move(annot_filepath, new_annot_filepath)

    for idx in range(test_size):
        img = img_list[train_size + idx]
        img_file = img.split('\\')[1]
        img_number = img_file.split(".")[0]
        annot_file = img_number + ".xml"
        img_filepath = screenshot_folder + f'/{img_file}'
        annot_filepath = screenshot_folder + f'/{annot_file}'

        new_img_filepath = test_folder + f'/{img_file}'
        new_annot_filepath = test_folder + f'/{annot_file}'
        move(img_filepath, new_img_filepath)
        move(annot_filepath, new_annot_filepath)

    print(f"Split with {train_size * 100}% in /train and {test_size * 100}% in /test")

print(labels)
print(f'Creating Model: {CUSTOM_MODEL_NAME}')
print(f'Training for{TRAIN_STEPS}')
print(f'Using {PRETRAINED_MODEL_NAME}')

# Folder containing tensorflow files (Installed via github)
TENSORFLOW_PATH = "../RealTimeObjectDetection/Tensorflow"

# Folder containing all own files (Images, Annotations, Models, etc..)
WORKSPACE_PATH = "Workspace"

# Tensorflow script to generate tf Record
SCRIPTS_PATH = TENSORFLOW_PATH + "/scripts"
APIMODEL_PATH = TENSORFLOW_PATH + "/models"
# May need to customize generate_tfrecord.py file
GEN_TFRECORD_FILE = "mask_generate_tfrecord.py"

ANNOTATION_PATH = WORKSPACE_PATH + "/annotations"
IMAGE_PATH = WORKSPACE_PATH + '/images'
MODEL_PATH = WORKSPACE_PATH + '/my_models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + "/pre-trained-models"

# Change this
# CUSTOM_MODEL_NAME = 'over_ssdnet'
CONFIG_PATH = MODEL_PATH + f'/{CUSTOM_MODEL_NAME}/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + f'/{CUSTOM_MODEL_NAME}/'


# labels = [{'name': 'head', 'id': 1}]
with open(ANNOTATION_PATH + '/label_map.pbtxt', 'w+') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

os.system(f"python {SCRIPTS_PATH + f'/{GEN_TFRECORD_FILE}'} -x {IMAGE_PATH + '/train'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/train.record'}")
os.system(f"python {SCRIPTS_PATH + f'/{GEN_TFRECORD_FILE}'} -x{IMAGE_PATH + '/test'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/test.record'}")

OVERWATCH_MODEL_PATH = MODEL_PATH + '/' + CUSTOM_MODEL_NAME
if not os.path.exists(OVERWATCH_MODEL_PATH):
    os.mkdir(OVERWATCH_MODEL_PATH)

pretrained_config_path = PRETRAINED_MODEL_PATH + f'/{PRETRAINED_MODEL_NAME}/pipeline.config'
shutil.copy(pretrained_config_path, OVERWATCH_MODEL_PATH)


CONFIG_PATH = OVERWATCH_MODEL_PATH + '/pipeline.config'
config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
    proto_str = f.read()
    # Pass in the config string as first param, and the config variable as second param
    text_format.Merge(proto_str, pipeline_config)


# 2 classes
pipeline_config.model.ssd.num_classes = len(labels)
# Set batch size to 4
pipeline_config.train_config.batch_size = 4
# Where you want your model to start training from. Here, you want to utilize the checkpoint-0 in the pre-trained model (Transfer Learning)
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH + f'/{PRETRAINED_MODEL_NAME}/checkpoint/ckpt-0'
# Specify the type of task
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
# Specify your label_map_path and the tfrecords for train & test
pipeline_config.train_input_reader.label_map_path= ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']


config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:
    f.write(config_text)


# TRAIN_STEPS=50
OBJECT_DETECTION_MAIN = f"{APIMODEL_PATH}/research/object_detection/model_main_tf2.py"
PIPELINE_CONFIG_PATH = OVERWATCH_MODEL_PATH + "/pipeline.config"


# Specify: classesnames, train_steps, model_name

TRAIN_COMMAND = f"python {OBJECT_DETECTION_MAIN} --model_dir={OVERWATCH_MODEL_PATH} --pipeline_config_path={PIPELINE_CONFIG_PATH} --num_train_steps={TRAIN_STEPS}"
print(TRAIN_COMMAND)
os.system(TRAIN_COMMAND)