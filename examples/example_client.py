import math
import numpy as np
import cv2
from PIL import Image


from ros2_open_voc_landing_heatmap_srv.srv import GetLandingHeatmap
from sensor_msgs.msg import Image as ImageMsg
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.declare_parameter('topic', '/carla/flying_sensor/rgb_down/image')
        topic = self.get_parameter('topic').value

        self.cli = self.create_client(GetLandingHeatmap, 'generate_landing_heatmap')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('generate_landing_heatmap service not available, waiting again...')
        self.req = GetLandingHeatmap.Request()
        self.cv_bridge = CvBridge()
        self.img = None
        self.img_sub = self.create_subscription(
            ImageMsg,
            topic,
            self.img_sub_cb,
            1)

    def img_sub_cb(self, msg):
        self.img = msg
        self.destroy_subscription(self.img_sub)

    def send_request(self, image_msg, prompts, erosion_size):
        #request.image, request.prompts, request.erosion_size
        self.req.image = image_msg
        # the service expects a string of prompts separated by ';'
        self.req.prompts = ";".join(prompts)
        self.req.erosion_size = int(erosion_size)
        self.future = self.cli.call_async(self.req)

        # Non-blocking example
        while (not self.future.done()):
            self.get_logger().info('waiting for result...')
            rclpy.spin_once(self)
        
        # Blocking example
        # self.get_logger().info('waiting for result...')
        # rclpy.spin_until_future_complete(self, self.future)

        return self.future.result()


def main():
    rclpy.init()

    minimal_client = MinimalClientAsync()

    # test_img = np.random.rand(256,256,3)*255
    # image_msg = minimal_client.cv_bridge.cv2_to_imgmsg((test_img).astype('uint8'), encoding='passthrough')
    while minimal_client.img is None:
        rclpy.spin_once(minimal_client)
    
    image_msg = minimal_client.img

    response = minimal_client.send_request(image_msg, 
                                           ["building", "tree", "road", "water", "transmission lines", "lamp post", "vehicle", "people"],
                                           7)

    minimal_client.get_logger().info(f'Success: {response.success}')
    
    heatmap = minimal_client.cv_bridge.imgmsg_to_cv2(response.heatmap, desired_encoding='mono8') # PIL is expecting RGB

    heatmap_center = heatmap.shape[0]//2, heatmap.shape[1]//2
    max_distance = math.sqrt(heatmap_center[0]**2+heatmap_center[1]**2)

    # descending order, best landing candidates
    y,x = np.dstack(np.unravel_index(np.argsort(heatmap.ravel()), heatmap.shape))[0][::-1][0]
    rel_dist = np.sqrt((y-(heatmap_center[0]-1)/2)**2+(x-(heatmap_center[0]-1)/2)**2)/max_distance
    angle = math.degrees(math.atan2(-y,x))
    print(f"Max pixel: {y,x}, value: {heatmap[y,x]} - Relative distance: {rel_dist:.2f} - Heading: {angle:.2f}")

    cv2.imwrite("heatmap.jpg", heatmap)

    heatmap = cv2.arrowedLine(cv2.cvtColor(heatmap,cv2.COLOR_GRAY2RGB), 
                              (heatmap_center[1],heatmap_center[0]),
                              (x,y),
                              color=(0,255,0), thickness=3, tipLength=.5)

    cv2.namedWindow('heatmap',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('heatmap', heatmap.shape[1], heatmap.shape[0])
    cv2.imshow('heatmap',heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()