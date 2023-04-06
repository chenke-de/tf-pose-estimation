#!/usr/bin/env python3
import time
import os
import sys
import ast

from threading import Lock
import rospy
import rospkg
import tensorflow as tf
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image
from tfpose_ros.msg import Persons, Person, BodyPartElm

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import model_wh, get_graph_path


def humans_to_msg(humans):
    persons = Persons()

    for human in humans:
        person = Person()

        for k in human.body_parts:
            body_part = human.body_parts[k]

            body_part_msg = BodyPartElm()
            body_part_msg.part_id = body_part.part_idx
            body_part_msg.x = body_part.x
            body_part_msg.y = body_part.y
            body_part_msg.confidence = body_part.score
            person.body_part.append(body_part_msg)
        persons.persons.append(person)

    return persons


def callback_image(data):
    # et = time.time()
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr('[tf-pose-estimation] Converting Image Error. ' + str(e))
        return

    acquired = tf_lock.acquire(False)
    if not acquired:
        return

    try:
        humans = pose_estimator.inference(cv_image, resize_to_default=True, upsample_size=resize_out_ratio)
    finally:
        tf_lock.release()

    msg = humans_to_msg(humans)
    msg.image_w = data.width
    msg.image_h = data.height
    msg.header = data.header

    if data.header.frame_id == "/csi_cam_0_link":
        pub_pose.publish(msg)
    elif data.header.frame_id == "/csi_cam_1_link":
        pub_pose_right.publish(msg)
    elif data.header.frame_id == "/csi_cam_5_link":
        pub_pose_left.publish(msg)

if __name__ == '__main__':
    rospy.loginfo('initialization+')
    rospy.init_node('TfPoseEstimatorROS', anonymous=True, log_level=rospy.INFO)

    # parameters
    image_topic = rospy.get_param('~camera', '')
    image_front_topic = rospy.get_param('~camera_front', '')
    image_right_topic = rospy.get_param('~camera_right', '')
    image_left_topic = rospy.get_param('~camera_left', '')
    model = rospy.get_param('~model', 'cmu')

    resolution = rospy.get_param('~resolution', '432x368')
    resize_out_ratio = float(rospy.get_param('~resize_out_ratio', '4.0'))
    tf_lock = Lock()

    if not image_topic:
        rospy.logerr('Parameter \'camera\' is not provided.')
        sys.exit(-1)

    try:
        w, h = model_wh(resolution)
        graph_path = get_graph_path(model)

        rospack = rospkg.RosPack()
        graph_path = os.path.join(rospack.get_path('tfpose_ros'), graph_path)
    except Exception as e:
        rospy.logerr('invalid model: %s, e=%s' % (model, e))
        sys.exit(-1)

    if rospy.get_param('~allow_growth'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        pose_estimator = TfPoseEstimator(graph_path, target_size=(w, h), tf_config=config)
    else:
        pose_estimator = TfPoseEstimator(graph_path, target_size=(w, h))
    cv_bridge = CvBridge()

    rospy.Subscriber(image_front_topic, Image, callback_image, queue_size=1, buff_size=2**24)
    rospy.Subscriber(image_right_topic, Image, callback_image, queue_size=1, buff_size=2**24)
    rospy.Subscriber(image_left_topic, Image, callback_image, queue_size=1, buff_size=2**24)
    pub_pose = rospy.Publisher('~pose', Persons, queue_size=1)
    pub_pose_right = rospy.Publisher('~pose_right', Persons, queue_size=1)
    pub_pose_left = rospy.Publisher('~pose_left', Persons, queue_size=1)

    rospy.loginfo('start+')
    rospy.spin()
    rospy.loginfo('finished')
