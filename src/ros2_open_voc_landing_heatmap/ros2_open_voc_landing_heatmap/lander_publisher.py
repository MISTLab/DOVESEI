import math
from threading import Event

import numpy as np
import cv2


from ros2_open_voc_landing_heatmap_srv.srv import GetLandingHeatmap
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Twist

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time, Duration

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


from message_filters import ApproximateTimeSynchronizer, Subscriber


from cv_bridge import CvBridge


EPS = 0.001
class TwistPublisher(Node):

    def __init__(self):
        super().__init__('lander_publisher')
        self.declare_parameter('img_topic', '/carla/flying_sensor/rgb_down/image')
        self.declare_parameter('depth_topic', '/carla/flying_sensor/depth_down/image')
        self.declare_parameter('heatmap_topic', '/heatmap')
        self.declare_parameter('twist_topic', '/quadctrl/flying_sensor/ctrl_twist_sp')
        self.declare_parameter('mov_avg_size', 10)
        self.declare_parameter('resize', 15)
        self.declare_parameter('gain', 20)
        self.declare_parameter('z_speed', 1.0)
        self.declare_parameter('depth_new_size', 100)
        self.declare_parameter('mean_depth_side', 20)
        self.declare_parameter('altitude_landed', 1)
        self.declare_parameter('min_altitude_semantics', 10)
        img_topic = self.get_parameter('img_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        self.heatmap_topic = self.get_parameter('heatmap_topic').value
        self.twist_topic = self.get_parameter('twist_topic').value
        self.mov_avg_size = self.get_parameter('mov_avg_size').value
        self.resize = self.get_parameter('resize').value
        self.gain = self.get_parameter('gain').value
        self.z_speed = self.get_parameter('z_speed').value
        self.depth_new_size = self.get_parameter('depth_new_size').value
        self.altitude_landed = self.get_parameter('altitude_landed').value
        self.min_altitude_semantics = self.get_parameter('min_altitude_semantics').value
        self.mean_depth_side = self.get_parameter('mean_depth_side').value

        assert (self.resize % 2) == 1, self.get_logger().error('resize parameter MUST be odd!')

        
        self.mov_avg_counter = 0

        self.heatmap_mov_avg = None

        self.img_msg = None

        self.landing_done = False

        self.cli = self.create_client(GetLandingHeatmap, 'generate_landing_heatmap',
                                      callback_group=ReentrantCallbackGroup())
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('generate_landing_heatmap service not available, waiting again...')
        self.req = GetLandingHeatmap.Request()
        self.cv_bridge = CvBridge()

        # # QoS profile that will only keep the last message
        # # qos_prof = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        # qos_prof = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1)
                
        self.twist_pub = self.create_publisher(Twist, self.twist_topic,1)
        self.heatmap_pub = self.create_publisher(ImageMsg, self.heatmap_topic,1)

        self.tf_trials = 5
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        queue_size = 2
        delay_btw_msgs = 0.02
        tss = ApproximateTimeSynchronizer(
            [Subscriber(self, ImageMsg, img_topic),
             Subscriber(self, ImageMsg, depth_topic)],
             queue_size,
             delay_btw_msgs
             )
        
        tss.registerCallback(self.sense_and_act)
        self.get_logger().info('Ready to publish some twist messages!')


    def get_tf(self, t=0.0, timeout=1.0, map_frame="map", target_frame="flying_sensor"):
        try:
            now = Time(nanoseconds=t)
            trans = self.tf_buffer.lookup_transform(map_frame, target_frame,
                now,
                timeout=Duration(seconds=timeout))

            self.get_logger().debug(f'TF received {trans}')
            curr_pos = [trans.transform.translation.x, 
                        trans.transform.translation.y, 
                        trans.transform.translation.z]

            curr_quat = [trans.transform.rotation.x,
                        trans.transform.rotation.y,
                        trans.transform.rotation.z,
                        trans.transform.rotation.w]

            s = trans.header.stamp.sec
            ns = trans.header.stamp.nanosec
            return (s + ns/1E9), curr_pos, curr_quat

        except TransformException as ex:
            self.get_logger().error(f'Could not transform {map_frame} to {target_frame}: {ex}')


    def error_from_depth_clusters(self, depthmsg, dist_weight = 2, n_clusters = 2, max_dist=20):
        depth = self.cv_bridge.imgmsg_to_cv2(depthmsg, desired_encoding='passthrough')
        depth = np.asarray(cv2.resize(depth, (int(self.depth_new_size*depth.shape[1]/depth.shape[0]),self.depth_new_size)))
        # In CARLA the depth goes up to 1000m, but we want up to 20m
        depth[depth>max_dist] = np.nan
        depth = depth/max_dist
        depth[np.isnan(depth)] = 1.0
        depth += np.random.rand(*depth.shape)*0.001

        depth_center = depth.shape[0]/2, depth.shape[1]/2

        ij = np.argwhere(depth>=0).astype(float)
        ij[:,0] /= depth.shape[0] # values (coordinates) become [0,1]
        ij[:,1] /= depth.shape[1] # values (coordinates) become [0,1]
        z = depth.reshape(np.prod(depth.shape),1)

        X = np.hstack((ij,dist_weight*z))
        # At this point it should have a filter to exclude infs or nans
        # X = X[np.isfinite(X)] # checks for both inf and nan

        # Define criteria = ( type, max_iter, epsilon)
        max_iter = 300
        epsilon = 0.001
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
        # Set flags (Just to avoid line break in the code)
        flags = cv2.KMEANS_PP_CENTERS # same as sklearn default

        attempts = 10

        # Apply KMeans
        compactness,labels,centers = cv2.kmeans(np.float32(X),n_clusters,None,criteria,attempts,flags)

        labels = labels.ravel() # to match scikit-learn kmeans
        unique_labels, counts_labels = np.unique(labels[labels>=0], return_counts=True)

        objectives = []
        for l in unique_labels:
            tmp = X[labels==l][:,:2]
            area_ratio = ((tmp[:,0].max()-tmp[:,0].min())*depth.shape[0]*(tmp[:,1].max()-tmp[:,1].min())*depth.shape[1])/tmp.shape[0]
            z_label = z.ravel()[labels==l]
            z_std = z_label.std()
            objective = 1/(z_std*area_ratio)
            objectives.append(objective) # unique_labels are sorted...

        objectives = np.asarray(objectives)

        if objectives.std() < objectives.mean()*0.1:
            # don't move...
            x = 0.0
            y = 0.0
        else:
            l = np.argsort(objectives)[-1]
            x = -(centers[l][0] - int(depth_center[0])) / depth_center[0]
            y = (centers[l][1] - int(depth_center[1])) / depth_center[1]
        
        xc = int(depth_center[0])
        yc = int(depth_center[1])
        halfside = int(self.mean_depth_side/2)
        mean_depth_under_drone = (depth[xc-halfside:xc+halfside,yc-halfside:yc+halfside]*max_dist).mean()
        return x,y,mean_depth_under_drone



    def error_from_semantics(self, rgbmsg):
        self.get_logger().warn(f'Sending heatmap service request...')
        response = self.get_heatmap(rgbmsg, 
                                    ["building", "tree", "road", "water", "transmission lines", "lamp post", "vehicle", "people"],
                                    7)
        if response is None:
            self.get_logger().warn(f'Empty response?!?!')
            return
        
        self.get_logger().warn(f'Heatmap received!')
        heatmap_msg = response.heatmap
        heatmap = self.cv_bridge.imgmsg_to_cv2(heatmap_msg, desired_encoding='mono8')
        
        # Debug
        # heatmap = np.zeros(heatmap.shape, dtype=heatmap.dtype)
        # heatmap[:50,:50] = 255
        # heatmap[-50:,-50:] = 255

        if self.heatmap_mov_avg is None:
            resize_w = int(self.resize*(heatmap.shape[1]/heatmap.shape[0]))
            resize_w = resize_w + 1-(resize_w % 2) # always odd
            self.heatmap_mov_avg = np.zeros((self.mov_avg_size, self.resize, resize_w), dtype='uint8')
        
        heatmap_resized = cv2.resize(heatmap,(self.heatmap_mov_avg.shape[2],self.heatmap_mov_avg.shape[1]),cv2.INTER_AREA)

        self.heatmap_mov_avg[self.mov_avg_counter] = heatmap_resized
        heatmap_resized = self.heatmap_mov_avg.mean(axis=0).astype('uint8')
        if self.mov_avg_counter < (self.mov_avg_size-1):
            self.mov_avg_counter += 1
        else:
            self.mov_avg_counter = 0

        heatmap_center = heatmap_resized.shape[0]/2, heatmap_resized.shape[1]/2
        # descending order, best landing candidates
        x_idx, y_idx = np.dstack(np.unravel_index(np.argsort(heatmap_resized.ravel()), heatmap_resized.shape))[0][::-1][0]

        x = (-(x_idx - int(heatmap_center[0]))) / heatmap_center[0]
        y = (y_idx - int(heatmap_center[1])) / heatmap_center[1]

        self.get_logger().warn(f'Publishing resized heatmap image at {self.heatmap_topic}')
        img_msg = self.cv_bridge.cv2_to_imgmsg(heatmap_resized, encoding='mono8')
        self.heatmap_pub.publish(img_msg)

        return x,y


    def get_rangefinder(self):
        self.get_tf()
        res = self.get_tf()
        if res is None:
            return None
        t, init_pos, init_quat = res
        return init_pos[2]


    def sense_and_act(self, rgbmsg, depthmsg):
        self.get_logger().warn(f'New data received!')

        altitude = self.get_rangefinder()
        if altitude is None:
            return
        
        x = y = 0.0
        if not self.landing_done:
            xd_err,yd_err,mean_depth_under_drone = self.error_from_depth_clusters(depthmsg)

            x_err = y_err = 0.0
            if altitude > self.min_altitude_semantics:
                x_err,y_err = self.error_from_semantics(rgbmsg)

            self.get_logger().info(f"Segmentation X,Y err: {x_err:.2f},{y_err:.2f}, Depth Cluster X,Y err: {xd_err:.2f},{yd_err:.2f}")
            self.get_logger().info(f"Mean depth under drone: {mean_depth_under_drone:.2f}, Altitude: {altitude:.2f}")
            
            
            # TODO: FUSE THE DEPTH AND THE HEATMAP
            x = x_err
            y = y_err


            x = x if abs(x) > EPS else 0.0
            y = y if abs(y) > EPS else 0.0

            z = -self.z_speed if (x+y)==0.0 else 0.0


        if altitude<self.altitude_landed and (x+y)==0.0:
            x = y = z = 0.0
            self.get_logger().warn("Landed!")
            self.landing_done = True

        twist = Twist()
        twist.linear.x = x * self.gain
        twist.linear.y = -y * self.gain
        twist.linear.z = z
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        self.get_logger().warn(f'Publishing velocities ({(twist.linear.x, twist.linear.y, twist.linear.z)}) at {self.twist_topic}')
        self.twist_pub.publish(twist)


        

    def get_heatmap(self, image_msg, prompts, erosion_size):
        #request.image, request.prompts, request.erosion_size
        self.req.image = image_msg
        # the service expects a string of prompts separated by ';'
        self.req.prompts = ";".join(prompts)
        self.req.erosion_size = int(erosion_size)       

        event = Event()
        def future_done_callback(future):
            event.set()
        
        future = self.cli.call_async(self.req)
        future.add_done_callback(future_done_callback)
        
        event.wait(timeout=5)

        return future.result()


    def on_shutdown_cb(self):
        self.get_logger().warn('Shutting down... sending zero velocities!')
        self.twist_pub.publish(Twist())


def main():
    rclpy.init()
    lander_publisher = TwistPublisher()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(lander_publisher)
    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        lander_publisher.on_shutdown_cb()
        executor.shutdown()
        lander_publisher.destroy_node()


if __name__ == '__main__':
    main()