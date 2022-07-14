import os
import cv2
import wget
import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from google.protobuf import text_format
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils

# Setup paths
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow', 'scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow', 'models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace', 'annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace', 'images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace', 'models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace', 'models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace', 'models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace', 'models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow', 'protoc')
}

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

labels = [{'name':'licence', 'id':1}]

# Setup function
def setup():
    # Futher installations given from VERIFICATION_SCRIPT output
    os.system('pip install tensorflow tensorflow-gpu --upgrade')
    os.system('pip install tensorflow_io')
    os.system('pip install tf-models-official')
    os.system('pip install pytz')
    return

# Tensorflow setup
def setup_tf():
    # Create folders
    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)

    # Tensorflow setup
    if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
        os.system('git clone https://github.com/tensorflow/models {}'.format(paths['APIMODEL_PATH']))

    # Install Tensorflow Object Detection 
    if not os.path.exists(os.path.join(paths['PROTOC_PATH'], 'bin')):
        url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
        wget.download(url)
        os.system('move protoc-3.15.6-win64.zip {}'.format(paths['PROTOC_PATH']))
        os.system('cd {} && tar -xf protoc-3.15.6-win64.zip'.format(paths['PROTOC_PATH']))
        os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))   
        os.system('cd Tensorflow/models/research\
            && protoc object_detection/protos/*.proto --python_out=.\
            && copy object_detection\\packages\\tf2\\setup.py setup.py\
            && python setup.py build && python setup.py install')
        os.system('cd Tensorflow/models/research/slim && pip install -e .')

    if not os.path.exists(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz')):
        wget.download(PRETRAINED_MODEL_URL)
        os.system('move {} {}'.format(PRETRAINED_MODEL_NAME + '.tar.gz', paths['PRETRAINED_MODEL_PATH']))
        os.system('cd {} && tar -zxvf {}'.format(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz'))

def verify_installation():
    # Verify Installation
    VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
    os.system('python {}'.format(VERIFICATION_SCRIPT))
    return

def create_label_map():
    with open(files['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

def create_tf_records():
    if not os.path.exists(files['TF_RECORD_SCRIPT']):
        os.system('git clone https://github.com/nicknochnack/GenerateTFRecord {}'.format(paths['SCRIPTS_PATH']))
        
        # Change the file parameters (member[4] into member[5])
        with open(files['TF_RECORD_SCRIPT'], 'r') as f:
            data = f.read()
            data = data.replace('member[4]', 'member[5]')

        with open(files['TF_RECORD_SCRIPT'], 'w') as f:
            f.write(data)
        
        # Execute the script
        os.system('python {} -x {} -l {} -o {}'.format(
            files['TF_RECORD_SCRIPT'],
            os.path.join(paths['IMAGE_PATH'], 'train'),
            files['LABELMAP'],
            os.path.join(paths['ANNOTATION_PATH'], 'train.record')
        ))

        os.system('python {} -x {} -l {} -o {}'.format(
            files['TF_RECORD_SCRIPT'],
            os.path.join(paths['IMAGE_PATH'], 'test'),
            files['LABELMAP'],
            os.path.join(paths['ANNOTATION_PATH'], 'test.record')
        ))
    return

# Model configuration
def create_pipeline_config():
    os.system('copy {} {}'.format(
        os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config'),
        os.path.join(paths['CHECKPOINT_PATH'])
    ))
    
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = len(labels)
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = files['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

    config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
        f.write(config_text) 

    return

def train():
    TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
    command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=10000".format(
        TRAINING_SCRIPT, 
        paths['CHECKPOINT_PATH'],
        files['PIPELINE_CONFIG']
    )

    print(command)
    os.system(command)
    return

def test():
    TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
    command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(
        TRAINING_SCRIPT, 
        paths['CHECKPOINT_PATH'],
        files['PIPELINE_CONFIG'],
        paths['CHECKPOINT_PATH']
    )

    print(command)
    os.system(command)
    return

def load_from_checkpoint():
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-13')).expect_partial()
    return detection_model

@tf.function
def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def load_and_detect():
    detection_model = load_from_checkpoint()
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

    # While loop to read and process images
    IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'Cars428.png')

    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor, detection_model)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates = True,
        max_boxes_to_draw = 5,
        min_score_thresh = .8,
        agnostic_mode = False
    )

    matplotlib.use('TkAgg')
    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.show()

    return

# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)

if __name__ == '__main__':
    # verify_installation()
    # setup_tf()
    # create_label_map()
    # create_tf_records()
    # create_pipeline_config()

    # train()
    # test()

    load_and_detect()
