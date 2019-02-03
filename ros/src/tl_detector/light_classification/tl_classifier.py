import numpy as np
import os
import tensorflow as tf
import cv2

#from utils import label_map_util
#from utils import visualization_utils as vis_util
#from distutils.version import StrictVersion

from styx_msgs.msg import TrafficLight

FROZEN_INFERENCE_GRAPH_LOC = os.getcwd() + "/graph_traffic_light.pb"
#LABELS_LOC = os.getcwd() + "/label_map.pbtxt"
NUM_CLASSES = 1

class TLClassifier(object):
    def __init__(self):
        #load classifier
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(FROZEN_INFERENCE_GRAPH_LOC, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            # end with
        # end with

    # Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    #label_map = label_map_util.load_labelmap(LABELS_LOC)
    #categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
    #                                                            use_display_name=True)
    #category_index = label_map_util.create_category_index(categories)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #implement light color prediction
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:

                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_expanded = np.expand_dims(image, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded})
            # end for
        # end with

        #return TrafficLight.UNKNOWN
        return np.squeeze(scores)
