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

# Define colors
TEXT_RESET = '\033[0m'
TEXT_RED = '\033[91m'
TEXT_GREEN = '\033[92m'

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
last_checkpoint = 'ckpt-13'

# Class for the detector NN
class PlateDetect():
    # Constructor
    def __init__(self, initial_path:str='') -> None:
        self.detection_model = None
        self.category_index = None
        self.initial_path = initial_path

        # gpus = tf.config.list_physical_devices('GPU')
        # if gpus: print(gpus)
        # else: print("No GPU available!")

        return

    # Function to setup Tensorflow
    def setup_tf(self) -> None:
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

        # Download pretrained model
        if not os.path.exists(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz')):
            wget.download(PRETRAINED_MODEL_URL)
            os.system('move {} {}'.format(PRETRAINED_MODEL_NAME + '.tar.gz', paths['PRETRAINED_MODEL_PATH']))
            os.system('cd {} && tar -zxvf {}'.format(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz'))
        return

    # Function to verify requirements installation
    def verify_installation(self) -> None:
        VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
        os.system('python {}'.format(VERIFICATION_SCRIPT))
        return

    # Function to create the label map file
    def create_label_map(self) -> None:
        with open(files['LABELMAP'], 'w') as f:
            for label in labels:
                f.write('item { \n')
                f.write('\tname:\'{}\'\n'.format(label['name']))
                f.write('\tid:{}\n'.format(label['id']))
                f.write('}\n')
        return

    # Function to create the pipeline config file
    def create_tf_records(self) -> None:
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

    # Function to configure the pretrained model
    def create_pipeline_config(self) -> None:
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

    # Function to train the model
    def train(self) -> None:
        TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
        command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=10000".format(
            TRAINING_SCRIPT, 
            paths['CHECKPOINT_PATH'],
            files['PIPELINE_CONFIG']
        )

        print(command)
        os.system(command)
        return

    # Function to test the model
    def test(self) -> None:
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

    # Function to load the model from a checkpoint
    def load_from_checkpoint(self) -> None:
        print(TEXT_GREEN + '>> Loading Detector model from {} ...'.format(last_checkpoint) + TEXT_RESET)

        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(self.initial_path + files['PIPELINE_CONFIG'])
        self.detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(os.path.join(self.initial_path, paths['CHECKPOINT_PATH'], last_checkpoint)).expect_partial()

        self.category_index = label_map_util.create_category_index_from_labelmap(self.initial_path + files['LABELMAP'])
        
        print(TEXT_GREEN + '>> Model loaded successfully from {}.'.format(last_checkpoint) + TEXT_RESET)
        return 

    # Function to detect a plate in an image
    @tf.function
    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    # Function to test the NN on workspace images
    def test_workspace_image(self, image:str):
        IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], image)

        img = cv2.imread(IMAGE_PATH)
        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)

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
            self.category_index,
            use_normalized_coordinates = True,
            max_boxes_to_draw = 5,
            min_score_thresh = .4,
            agnostic_mode = False
        )

        matplotlib.use('TkAgg')
        plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        plt.show()

        return

    # Function to process an image and return the cropped plate image and its coordinates
    def detect_and_crop(self, image:np.ndarray) -> tuple[cv2.Mat, list[int]]:
        # Create the tensor to feed into the NN
        input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates = True,
            max_boxes_to_draw = 5,
            min_score_thresh = .4,
            agnostic_mode = False
        )

        # matplotlib.use('TkAgg')
        # plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        # plt.show()

        print(detections['detection_boxes'])
        print(detections['detection_classes'])
        print(detections['detection_scores'])

        # Get the coordinates of the plate
        x1 = int(detections['detection_boxes'][0][0] * image.shape[0])
        y1 = int(detections['detection_boxes'][0][1] * image.shape[1])
        x2 = int(detections['detection_boxes'][0][2] * image.shape[0])
        y2 = int(detections['detection_boxes'][0][3] * image.shape[1])

        # Crop the plate
        plate = image[x1:x2, y1:y2]
        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        return plate, (x1, y1, x2, y2)
