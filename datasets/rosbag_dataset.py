'''
Author: Bahaa Aldeeb
Setup:
    - Install ROS related stuff:
        - pip install pycryptodomex gnupg rospkg
    - Install numpy & numpy-quaternion
'''

import rospy
import rosbag
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped

import numpy as np
from quaternion import as_rotation_matrix, quaternion

def _ros_camera_info_to_np_intrinsic(info:CameraInfo):
    return np.array([
        [info.K[0], info.K[1], info.K[2]],
        [info.K[3], info.K[4], info.K[5]],
        [info.K[6], info.K[7], info.K[8]]
    ])

def _ros_image_to_np(image:Image, depth_to_meters=1e-3):
    H, W = image.height, image.width
    if image.encoding == 'rgb8':
        rgb = np.frombuffer(image.data, dtype=np.byte)
        img = rgb.reshape(H, W, 3).astype(np.uint8)
    elif image.encoding == 'rgba8':
        rgb = np.frombuffer(image.data, dtype=np.byte)
        img = rgb.reshape(H, W, 4).astype(np.uint8)
    elif image.encoding == 'bgra8':
        rgb = np.frombuffer(image.data, dtype=np.byte)
        img = rgb.reshape(H, W, 4)[:, :, (2,1,0)].astype(np.uint8)
    elif image.encoding == '16UC1':
        d = np.frombuffer(image.data, dtype=np.uint16).reshape(H, W)
        img = d.astype(np.float32) * depth_to_meters
    elif image.encoding == 'bgra8':
        rgbd = np.frombuffer(image.data, dtype=np.byte)
        rgbd = rgbd.reshape(H, W, 4)[:, :, (2,1,0)].astype(np.uint8)
        img = rgbd[:,:,3].astype(np.uint16).astype(np.float32) * depth_to_meters
    else: 
        raise RuntimeError(f'Image to Numpy is not setup to handle {image.encoding}.')
    return img


def _ros_pose_to_np_se3_matrix(pose:PoseWithCovarianceStamped):
    mat = np.eye(4)
    # quaternion to rotation matrix
    q = pose.pose.pose.orientation
    q = quaternion(q.w, q.x, q.y, q.z)
    mat[:3,:3] = as_rotation_matrix(q)
    # mat[:3,:3] = np.array([
    #     [1 - 2*q.y**2 - 2*q.z**2, 2*q.x*q.y - 2*q.z*q.w, 2*q.x*q.z + 2*q.y*q.w],
    #     [2*q.x*q.y + 2*q.z*q.w, 1 - 2*q.x**2 - 2*q.z**2, 2*q.y*q.z - 2*q.x*q.w],
    #     [2*q.x*q.z - 2*q.y*q.w, 2*q.y*q.z + 2*q.x*q.w, 1 - 2*q.x**2 - 2*q.y**2]
    # ])
    # translation
    mat[:3,3] = np.array([
        pose.pose.pose.position.x,
        pose.pose.pose.position.y,
        pose.pose.pose.position.z
    ])
    return mat

def get_topics_in_path(path:str):
    bag = rosbag.Bag(path)
    return list(bag.get_type_and_topic_info()[1])

class RosbagReader:
    ''' 
    Takes a ros bag and returns one item at a time.
    Assumes that the bag topics are logged in a normally distributed fashion. 
    ''' 
    def __init__(self, path, topics=None, topic_name_map=None):
        self.bag = rosbag.Bag(path)

        if topics is not None:
            available_topics = self.get_topics()
            assert all([t in available_topics for t in topics]), "Some topics are not available in the bag."
            self._topics = topics
        else:
            self._topics = self.get_topics()

        _, self._len, self._time_slots = self._get_topic_with_least_messages()
        self._start, self._end = self.bag.get_start_time(), self.bag.get_end_time()
        time_steps = (self._end - self._start) / self.__len__()

        self._time_eps = rospy.Time.from_sec(time_steps / 2) # TODO: make into parameter

        self._postprocess_type = {
            'sensor_msgs/Image': _ros_image_to_np,
            'geometry_msgs/PoseWithCovarianceStamped': _ros_pose_to_np_se3_matrix,
            'sensor_msgs/CameraInfo': _ros_camera_info_to_np_intrinsic,
        }

        # The output will use these names for topics 
        if topic_name_map is not None:
            self._name_of_topic = topic_name_map
        for t in self._topics:
            if t not in self._name_of_topic:
                self._name_of_topic[t] = t

    def clear_postprocess(self):
        self._postprocess_type = {}
    def unregister_postprocess(self, msg_type:str):
        del self._postprocess_type[msg_type]
    def register_postprocess(self, msg_type:str, func:callable):
        self._postprocess_type[msg_type] = func

    def __len__(self): return self._len

    def _get_topic_with_least_messages(self):
        topic = self._topics[0]
        count = self.bag.get_message_count(topic)
        for t in self._topics[1:]:
            c = self.bag.get_message_count(t)
            if c < count: count, topic = c, t
        time_sec = [m[2] for m in self.bag.read_messages(topics=[topic])]
        return topic, count, time_sec  

    def get_topics(self):
        return list(self.bag.get_type_and_topic_info()[1])

    def _msgs_at(self, time:rospy.Time, rigourous=True):
        '''Ensures that messages are returned by proximity from intended time.'''
        s = time.to_time() - self._time_eps.to_time()
        e = time.to_time() + self._time_eps.to_time()
        msgs = self.bag.read_messages(topics = self._topics, 
                                      start_time = rospy.Time.from_sec(s), 
                                      end_time   = rospy.Time.from_sec(e))
        if rigourous:
            msgs = list(msgs)
            dt = [abs(m[2] - time) for m in msgs]
            idxs = sorted(range(len(dt)), key=lambda k: dt[k])
            for i in idxs:
                yield msgs[i]
        else:
            for m in msgs:
                yield m

    def __getitem__(self, idx):
        t = self._time_slots[idx]
        data = {}
        for msg in self._msgs_at(t):
            if msg[0] in data: continue # if topic already obtained
            msg_name = self._name_of_topic[msg[0]]
            if msg[1]._type in self._postprocess_type:
                data[msg_name] = self._postprocess_type[msg[1]._type](msg[1])
            else:
                data[msg_name] = msg[1]
            if len(data) == len(self._topics): break # All topics collected
        return data

    def __del__(self):
        self.bag.close()


def collate_fn(batch):
    rgb = np.stack([v['color'].transpose(2,0,1) for v in batch])/255.0
    
    # Hack to satisfy mrcnn
    for b in batch: b['boxes'] = np.empty([0, 4]).astype(np.float32)
    for b in batch: b['nocs'] =  np.empty([3, rgb.shape[-2], rgb.shape[-1]])
    for b in batch: b['labels'] = np.empty([0]).astype(np.int64)
    for b in batch: b['masks'] = np.empty([0, rgb.shape[-2], rgb.shape[-1]])
    
    # Torchify
    import torch 
    rgb = torch.from_numpy(rgb).float()
    for b in batch:
        for k,v in b.items():
            if isinstance(v, np.ndarray):
                b[k] = torch.from_numpy(v)

    return rgb, batch



if __name__ == '__main__':
    '''This is an example of how to use the RosbagReader class.'''
    import matplotlib.pyplot as plt

    # Specify bag path
    bag_path = '/media/baldeeb/ssd2/Data/kinect/images_poses_camerainfo.bag'

    # Determine the topics you want to acquire and their desired names
    print(get_topics_in_path(bag_path))
    topic_names = {'/rtabmap/rtabmap/localization_pose': 'pose',
                   '/k4a/depth_to_rgb/camera_info': 'intrinsics',
                   '/k4a/depth_to_rgb/image_raw': 'depth',
                   '/k4a/rgb/image_raw': 'color',}
    
    # Create the dataset with this information
    dataset = RosbagReader(bag_path, 
                           topics=list(topic_names.keys()),
                           topic_name_map=topic_names)
    
    # Voila!
    for name, data in dataset[0].items():
        try:
            plt.title(name); plt.imshow(data); plt.show()
        except:
            print(name)

    # NOTE: You can also register your own postprocessing functions
    # By default: 
    #   - any image type will be processed to numpy.
    #   - any pose with covariance will be processed to a 4x4 matrix.
