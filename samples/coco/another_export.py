from __future__ import print_function

from keras import backend as K

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def

#from keras.models import Model

import shutil
import os
import numpy as np
import tensorflow as tf



import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import re
import time
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
#os.environ['CUDA_VISIBLE_DEVICES']='0,1'
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")



class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES= 5
    BATCH_SIZE = 1

config = InferenceConfig()
config.display()



# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
max =0
dirs = os.listdir(os.path.join(os.getcwd(),'../../logs/coco20180518T1056'))
freg = re.compile(r'_\d+')
for dir in dirs:
    number = freg.search(dir)
    if number is not None:
        if int(number.group()[1:])>int(max):
            max =number.group()[1:]
max = '0125'
COCO_MODEL_PATH='/home/dev02/mask_rcnn/logs/coco20180518T1056/mask_rcnn_coco_'+max+'.h5'
model.load_weights(COCO_MODEL_PATH, by_name=True)

image = skimage.io.imread('/home/dev02/mask_rcnn/test_images/chosen/test8.jpg')

results = model.detect([image], verbose=1)

#old method
# K._LEARNING_PHASE = tf.constant(0)
# K.set_learning_phase(0)

# with K.get_session() as sess:

export_path_base = './serving_graph/'
export_path = os.path.join(
    tf.compat.as_bytes(export_path_base),
    tf.compat.as_bytes('2'))

print('Exporting trained model to', export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

input_image = tf.saved_model.utils.build_tensor_info(model.input_image)
input_image_meta = tf.saved_model.utils.build_tensor_info(model.input_image_meta)
input_anchors = tf.saved_model.utils.build_tensor_info(model.input_anchors)

mrcnn_mask_output = tf.saved_model.utils.build_tensor_info(model.mrcnn_mask_output)
detection_output = tf.saved_model.utils.build_tensor_info(model.detection_output)

prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'input_image': input_image,
                'image_metas': input_image_meta,
                'input_anchors': input_anchors,
                },
        outputs={
            'mrcnn_mask_output': mrcnn_mask_output,
            'detection_output':detection_output
        }
    ))

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')



with K.get_session() as sess:

    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={'predict': prediction_signature})
    builder.save()
