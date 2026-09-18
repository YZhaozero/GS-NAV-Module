import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import asyncio
import websockets
import json
import transforms3d
import os
import copy
from std_msgs.msg import Header

# --- 导入处理: pypcd2 ---
try:
    from pypcd2 import PointCloud
except ImportError:
    try:
        from pypcd2.point_cloud import PointCloud
    except ImportError:
        PointCloud = None 

# --- 辅助函数：增强版 PCD 读取器 ---
def read_pcd_fallback(path):
    meta = {}
    header_lines = 0
    with open(path, 'rb') as f:
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines += 1
            if not line: break
            parts = line.split()
            if not parts: continue
            if parts[0] == 'DATA':
                meta['DATA'] = parts[1]
                break
            if parts[0] == 'POINTS':
                meta['POINTS'] = int(parts[1])
            if parts[0] == 'FIELDS':
                meta['FIELDS'] = parts[1:]
    
    if 'DATA' not in meta or 'POINTS' not in meta:
        return None

    points = None
    try:
        if meta['DATA'] == 'ascii':
            raw = np.loadtxt(path, skiprows=header_lines, dtype=np.float32)
            if raw.ndim == 2 and raw.shape[1] >= 3:
                points = raw[:, :3]
        elif meta['DATA'] == 'binary':
            with open(path, 'rb') as f:
                for _ in range(header_lines):
                    f.readline()
                buffer = f.read()
                data = np.frombuffer(buffer, dtype=np.float32)
                num_points = meta['POINTS']
                if num_points > 0:
                    stride = data.size // num_points
                else:
                    stride = 0
                
                if stride >= 3:
                    reshaped_data = data[:num_points*stride].reshape(num_points, stride)
                    points = reshaped_data[:, :3]
                else:
                    return None
    except Exception as e:
        print(f"Fallback reader error: {e}")
        return None
    return points

# --- 辅助函数：ROS 点云变换 ---
def transform_cloud(cloud_msg, transform_matrix):
    points_gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
    try:
        points_list = list(points_gen)
        if not points_list:
            return cloud_msg
        structured_arr = np.array(points_list)
        if structured_arr.dtype.names:
            points = np.column_stack((structured_arr['x'], structured_arr['y'], structured_arr['z']))
        else:
            points = np.array(points_list, dtype=np.float32)
        points = points.astype(np.float32)
    except Exception as e:
        print(f"Numpy conversion error: {e}")
        return cloud_msg

    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    points_hom = np.hstack((points, ones))
    points_transformed = (transform_matrix @ points_hom.T).T
    
    header = cloud_msg.header
    header.frame_id = "map" 
    return pc2.create_cloud_xyz32(header, points_transformed[:, :3])

def msg_to_matrix(transform_msg):
    t = transform_msg.transform.translation
    r = transform_msg.transform.rotation
    mat_r = transforms3d.quaternions.quat2mat([r.w, r.x, r.y, r.z])
    mat_t = np.array([t.x, t.y, t.z])
    return transforms3d.affines.compose(mat_t, mat_r, [1, 1, 1])

# --- 核心节点类 ---
class ManualRelocalizationNode(Node):
    def __init__(self):
        super().__init__('manual_relocalization_node')
        
        default_pcd = '/home/zy/ws/car_simple_sim/src/car_localization/localizer/PCD/dilo_map.pcd'
        self.declare_parameter('target_pcd_path', default_pcd)
        self.declare_parameter('init_x', 0.0)
        self.declare_parameter('init_y', 0.0)
        self.declare_parameter('init_yaw', 0.0)

        self.source_cloud_msg = None
        self.captured_frame_id = None 
        
        self.pcd_msg_cache = None        
        self.vis_source_cache = None     
        self.aligned_msg_cache = None    
        
        self.current_transform = np.eye(4) 
        self.target_pcd_path = self.get_parameter('target_pcd_path').value
        
        if self.target_pcd_path:
            self.load_timer = self.create_timer(1.0, self.timer_load_pcd_callback)

        self.republish_timer = self.create_timer(1.0, self.timer_republish_all)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos_profile_lidar = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.sub_lidar = self.create_subscription(
            PointCloud2, '/dlio/odom_node/pointcloud/deskewed', self.lidar_callback, qos_profile_lidar)
        
        qos_profile_map = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.pub_source = self.create_publisher(PointCloud2, '/manual_reloc/source_cloud', qos_profile_map)         
        self.pub_target = self.create_publisher(PointCloud2, '/manual_reloc/target_cloud', qos_profile_map)          
        self.pub_aligned = self.create_publisher(PointCloud2, '/manual_reloc/aligned_source_cloud', 10) 
        
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        self.init_transform()
        self.get_logger().info("ROS 2 Manual Reloc Node Started (6-DOF Support).")

    def init_transform(self):
        x = self.get_parameter('init_x').value
        y = self.get_parameter('init_y').value
        yaw = self.get_parameter('init_yaw').value
        T = transforms3d.affines.compose([x, y, 0], transforms3d.euler.euler2mat(0, 0, yaw), [1, 1, 1])
        self.current_transform = T

    def timer_load_pcd_callback(self):
        if self.target_pcd_path:
            if self.load_pcd_file(self.target_pcd_path):
                self.get_logger().info("Initial PCD loaded, stopping load timer.")
                self.load_timer.cancel()

    def timer_republish_all(self):
        if self.pcd_msg_cache is not None:
            self.pub_target.publish(self.pcd_msg_cache)
        if self.vis_source_cache is not None:
            self.pub_source.publish(self.vis_source_cache)
        if self.aligned_msg_cache is not None:
            self.pub_aligned.publish(self.aligned_msg_cache)

    def lidar_callback(self, msg):
        self.latest_scan = msg

    def save_current_frame(self):
        if hasattr(self, 'latest_scan'):
            self.source_cloud_msg = self.latest_scan
            self.captured_frame_id = self.source_cloud_msg.header.frame_id
            self.get_logger().info(f"Source Cloud Captured! Frame: {self.captured_frame_id}")
            
            vis_msg = copy.deepcopy(self.source_cloud_msg)
            vis_msg.header.frame_id = "map" 
            self.vis_source_cache = vis_msg
            self.pub_source.publish(vis_msg)
            self.update_visualization()
            return True
        self.get_logger().warn("No lidar data received yet!")
        return False

    def load_pcd_file(self, path):
        if not path:
            path = self.get_parameter('target_pcd_path').value
        if not path or not os.path.exists(path):
            return False

        points = read_pcd_fallback(path)
        if points is not None:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "map"
            cloud_msg = pc2.create_cloud_xyz32(header, points)
            self.pcd_msg_cache = cloud_msg
            self.pub_target.publish(cloud_msg)
            return True
        else:
            return False

    # [核心修改] 支持 6-DOF (XYZ 平移 + Roll/Pitch/Yaw 旋转)
    def apply_transform_increment(self, dx, dy, dz, d_roll, d_pitch, d_yaw):
        delta_T = transforms3d.affines.compose(
            [dx, dy, dz], 
            transforms3d.euler.euler2mat(d_roll, d_pitch, d_yaw), 
            [1, 1, 1]
        )
        self.current_transform = np.dot(self.current_transform, delta_T)
        self.update_visualization()

    def update_visualization(self):
        if self.source_cloud_msg is None:
            return
        aligned_msg = transform_cloud(self.source_cloud_msg, self.current_transform)
        self.aligned_msg_cache = aligned_msg
        self.pub_aligned.publish(aligned_msg)

    def broadcast_map_odom_tf(self):
        if self.source_cloud_msg is None or self.captured_frame_id is None:
            self.get_logger().error("Cannot publish TF: No source cloud captured yet.")
            return

        cloud_frame_id = self.captured_frame_id
        
        try:
            tf_odom_cloud = self.tf_buffer.lookup_transform(
                "odom", cloud_frame_id, rclpy.time.Time())
            
            mat_odom_cloud = msg_to_matrix(tf_odom_cloud)
            mat_map_cloud = self.current_transform
            mat_map_odom = np.dot(mat_map_cloud, np.linalg.inv(mat_odom_cloud))
            
            trans, rot, _, _ = transforms3d.affines.decompose44(mat_map_odom)
            qx, qy, qz, qw = transforms3d.quaternions.mat2quat(rot) 

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "map"
            t.child_frame_id = "odom"
            t.transform.translation.x = trans[0]
            t.transform.translation.y = trans[1]
            t.transform.translation.z = trans[2]
            t.transform.rotation.w = qx
            t.transform.rotation.x = qy
            t.transform.rotation.y = qz
            t.transform.rotation.z = qw

            self.tf_static_broadcaster.sendTransform(t)
            self.get_logger().info(f"Static TF Published: map -> odom (Based on cloud frame: {cloud_frame_id})")

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f"Failed to lookup transform odom->{cloud_frame_id}: {e}")
            if cloud_frame_id == "odom":
                self.get_logger().info("Cloud frame IS odom, using alignment directly.")
                self._broadcast_direct_transform()

    def _broadcast_direct_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom" 
        trans, rot, _, _ = transforms3d.affines.decompose44(self.current_transform)
        qx, qy, qz, qw = transforms3d.quaternions.mat2quat(rot) 
        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = trans[2]
        t.transform.rotation.w = qx
        t.transform.rotation.x = qy
        t.transform.rotation.y = qz
        t.transform.rotation.z = qw
        self.tf_static_broadcaster.sendTransform(t)
        self.get_logger().info("Static TF Published (Direct): map -> odom")

async def websocket_handler(websocket, node):
    async for message in websocket:
        try:
            data = json.loads(message)
            cmd = data.get("cmd")
            if cmd == "save_source":
                success = node.save_current_frame()
                await websocket.send(json.dumps({"status": "success" if success else "no_data"}))
            elif cmd == "move":
                # [核心修改] 解析 6 个自由度的参数
                node.apply_transform_increment(
                    data.get("dx", 0.0), 
                    data.get("dy", 0.0), 
                    data.get("dz", 0.0),  # New
                    data.get("droll", 0.0),  # New
                    data.get("dpitch", 0.0), # New
                    data.get("dyaw", 0.0)
                )
                await websocket.send(json.dumps({"status": "moved"}))
            elif cmd == "confirm":
                node.broadcast_map_odom_tf()
                await websocket.send(json.dumps({"status": "tf_published"}))
            elif cmd == "load_pcd":
                path = data.get("path", "")
                success = node.load_pcd_file(path)
                await websocket.send(json.dumps({"status": "loaded" if success else "failed"}))
        except Exception as e:
            node.get_logger().error(f"WebSocket Error: {e}")

async def spin_ros(node):
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.0)
        await asyncio.sleep(0.01)

async def start_app(node):
    async with websockets.serve(lambda ws: websocket_handler(ws, node), "0.0.0.0", 8765):
        node.get_logger().info("WebSocket Server Running on ws://0.0.0.0:8765")
        await spin_ros(node)

def main(args=None):
    rclpy.init(args=args)
    node = ManualRelocalizationNode()
    try:
        asyncio.run(start_app(node))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Main Loop Error: {e}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()