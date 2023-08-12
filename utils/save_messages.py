import sys

import cv2

from sensor_msgs.msg import Image as ImageMsg
from message_filters import ApproximateTimeSynchronizer, Subscriber
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge



class ImageSaving(Node):

    def __init__(self, folder):
        super().__init__('image_saving_module')
        self.folder = folder
        self.declare_parameter('img_topic', '/carla/flying_sensor/rgb_down/image')
        self.declare_parameter('depth_topic', '/carla/flying_sensor/depth_down/image')
        self.declare_parameter('heatmap_topic', '/heatmap')
        self.declare_parameter('raw_heatmap_topic', '/final_heatmap')
        img_topic = self.get_parameter('img_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        heatmap_topic = self.get_parameter('heatmap_topic').value
        raw_heatmap_topic = self.get_parameter('raw_heatmap_topic').value

        self.cv_bridge = CvBridge()

        queue_size = 1
        delay_btw_msgs = 2
        tss = ApproximateTimeSynchronizer(
            [
                Subscriber(self, ImageMsg, "/lander_state"),
                Subscriber(self, ImageMsg, img_topic),
                Subscriber(self, ImageMsg, depth_topic),
                Subscriber(self, ImageMsg, heatmap_topic),
                Subscriber(self, ImageMsg, raw_heatmap_topic)
            ],
                queue_size=queue_size,
                slop=delay_btw_msgs,
                allow_headerless=False
                )
        
        tss.registerCallback(self.save_messages)
        self.get_logger().warn(f'Node started, waiting for messages...')


    def save_messages(self, statemsg, rgbmsg, depthmsg, heatmapmsg, rawheatmapmsg):
        elapsed_time_msec = int(float(statemsg.header.frame_id.split('-')[-1].split(':')[1])*1000)
        rgb = self.cv_bridge.imgmsg_to_cv2(rgbmsg, desired_encoding='passthrough')
        depth = self.cv_bridge.imgmsg_to_cv2(depthmsg, desired_encoding='passthrough')
        heatmap = self.cv_bridge.imgmsg_to_cv2(heatmapmsg, desired_encoding='passthrough')
        rawheatmap = self.cv_bridge.imgmsg_to_cv2(rawheatmapmsg, desired_encoding='passthrough')

        cv2.imwrite(self.folder+"/rgb_"+str(elapsed_time_msec)+".png", rgb)
        cv2.imwrite(self.folder+"/depth_"+str(elapsed_time_msec)+".png", depth)
        cv2.imwrite(self.folder+"/heatmap_"+str(elapsed_time_msec)+".png", heatmap)
        cv2.imwrite(self.folder+"/rawheatmap_"+str(elapsed_time_msec)+".png", rawheatmap)

        self.get_logger().warn(f"{statemsg.header.frame_id}")
        self.get_logger().warn(f"Image batch {elapsed_time_msec} saved!")

    def on_shutdown_cb(self):
        self.get_logger().warn(f'Shutting down...')

def main():
    folder="saved_imgs"
    if len(sys.argv)>1:
        for arg in sys.argv[1:]:
            if "folder" in arg:
                folder = arg.split("=")[-1]
    print(f"Saving images to: {folder}")
    rclpy.init()
    image_saving_module = ImageSaving(folder)
    try:
        rclpy.spin(image_saving_module)
    except KeyboardInterrupt:
        pass
    finally:
        # image_saving_module.on_shutdown_cb()
        rclpy.shutdown()

if __name__ == '__main__':
    main()