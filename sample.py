#!/usr/bin/env python3
import math
import asyncio
import signal
import sys
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PointStamped

SAFE_DIST = 1.0  # meter

# ==== (Opsional) Unitree SDK2 Python ====
USE_UNITREE_SDK = True
try:
    # Nama modul/kls bisa sedikit beda antar versi SDK2; sesuaikan kalau perlu.
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelFactoryFinalize
    from unitree_sdk2py.clients.sport import SportClient
    from unitree_sdk2py.clients.obstacle_avoid import ObstacleAvoidClient
    from unitree_sdk2py.clients.odometer import OdometerClient
except Exception as e:
    USE_UNITREE_SDK = False
    print(f"[WARN] Unitree SDK2Py not available ({e}). Will fallback to /cmd_vel.", file=sys.stderr)


def sector_min(msg: LaserScan, start_deg: float, end_deg: float) -> float:
    """Ambil jarak minimum dalam sektor [start_deg, end_deg] relatif ke +X depan robot.
       LaserScan angle_min..angle_max dalam rad. Kita map ke derajat lalu pilih indeks."""
    # Normalisasi rentang derajat -> rad
    start_rad = math.radians(start_deg)
    end_rad   = math.radians(end_deg)

    # Pastikan start <= end
    if start_rad > end_rad:
        start_rad, end_rad = end_rad, start_rad

    # Hitung index rentang
    def clamp(i, lo, hi):
        return max(lo, min(i, hi))

    n = len(msg.ranges)
    # idx = round((angle - angle_min)/angle_increment)
    start_idx = clamp(int((start_rad - msg.angle_min) / msg.angle_increment), 0, n-1)
    end_idx   = clamp(int((end_rad   - msg.angle_min) / msg.angle_increment), 0, n-1)
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx

    window = msg.ranges[start_idx:end_idx+1] if end_idx >= start_idx else []
    if not window:
        return float('inf')

    # Bersihkan nilai NaN/Inf
    vals = [v for v in window if not math.isnan(v) and v > 0.0 and math.isfinite(v)]
    if not vals:
        return float('inf')
    return min(vals)


class SafeBubbleNode(Node):
    def __init__(self):
        super().__init__('safe_bubble_go2')

        # Subscribes
        self.scan_sub = self.create_subscription(LaserScan, '/utlidar/range_info', self.lidar_cb, 10)
        self.cmd_in_sub = self.create_subscription(Twist, '/cmd_vel_input', self.cmd_in_cb, 10)

        # Publishers (fallback & debug)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_point_pub = self.create_publisher(PointStamped, '/safe_bubble/debug_point', 10)

        self.latest_scan: Optional[LaserScan] = None
        self.latest_cmd: Twist = Twist()

        # Unitree SDK2 Clients
        self.sdk_ok = False
        if USE_UNITREE_SDK:
            try:
                ChannelFactoryInitialize(0)  # 0: default
                self.sport = SportClient()
                self.avoid = ObstacleAvoidClient()
                self.odom  = OdometerClient()
                # (opsional) aktifkan/konfig obstacle avoid bawaan di level robot jika diinginkan
                # self.avoid.set_enable(True)
                self.sdk_ok = True
                self.get_logger().info("Unitree SDK2Py initialized ✅")
            except Exception as e:
                self.get_logger().warn(f"Unitree SDK2Py init failed: {e}. Fallback to /cmd_vel.")
                self.sdk_ok = False

        # Timer loop (50 Hz) untuk apply command aman (jaga low-latency)
        self.timer = self.create_timer(0.02, self.loop)

        self.get_logger().info("SafeBubbleNode started (1.0 m safety) ✅")

    # ========= Callbacks =========
    def lidar_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def cmd_in_cb(self, msg: Twist):
        self.latest_cmd = msg

    # ========= Core Logic =========
    def compute_safety_filtered_cmd(self) -> Twist:
        """
        Terapkan aturan:
        - Jika akan maju (linear.x > 0) dan FRONT < SAFE_DIST => block maju
        - Jika akan mundur (linear.x < 0) dan BACK < SAFE_DIST (jika lidar cover) => block mundur (opsional; default tidak)
        - Jika akan belok kiri (angular.z > 0) dan LEFT < SAFE_DIST => block putar kiri
        - Jika akan belok kanan (angular.z < 0) dan RIGHT < SAFE_DIST => block putar kanan
        """
        cmd = Twist()
        cmd.linear.x  = self.latest_cmd.linear.x
        cmd.linear.y  = self.latest_cmd.linear.y  # kalau kamu pakai holonomic cmd, bisa dipakai juga
        cmd.angular.z = self.latest_cmd.angular.z

        if self.latest_scan is None:
            return cmd  # belum ada data, biarkan lewat (atau bisa juga set 0 demi aman)

        scan = self.latest_scan

        # Definisi sektor (deg) relatif ke +X depan robot
        # front: -10..+10, left: +60..+120, right: -120..-60 (sesuaikan FoV lidar)
        d_front = sector_min(scan, -10.0, +10.0)
        d_left  = sector_min(scan,  +60.0, +120.0)
        d_right = sector_min(scan, -120.0,  -60.0)

        # — blokir gerakan yang membuat robot mendekat —
        if cmd.linear.x > 0.0 and d_front < SAFE_DIST:
            self.get_logger().warn(f"Front blocked @ {d_front:.2f} m → stop forward")
            cmd.linear.x = 0.0

        # (opsional) kalau lidar nge-cover belakang dan kamu mau batasi mundur:
        # d_back = sector_min(scan, 170.0, -170.0)  # near ±180°
        # if cmd.linear.x < 0.0 and d_back < SAFE_DIST:
        #     self.get_logger().warn(f"Back blocked @ {d_back:.2f} m → stop backward")
        #     cmd.linear.x = 0.0

        if cmd.angular.z > 0.0 and d_left < SAFE_DIST:
            self.get_logger().warn(f"Left blocked @ {d_left:.2f} m → stop left-turn")
            cmd.angular.z = 0.0

        if cmd.angular.z < 0.0 and d_right < SAFE_DIST:
            self.get_logger().warn(f"Right blocked @ {d_right:.2f} m → stop right-turn")
            cmd.angular.z = 0.0

        # (opsional) jika kamu pakai strafing (linear.y), bisa juga dibatasi:
        if cmd.linear.y > 0.0 and d_left < SAFE_DIST:
            self.get_logger().warn(f"Left side close @ {d_left:.2f} m → block +Y")
            cmd.linear.y = 0.0
        if cmd.linear.y < 0.0 and d_right < SAFE_DIST:
            self.get_logger().warn(f"Right side close @ {d_right:.2f} m → block -Y")
            cmd.linear.y = 0.0

        # debug titik “arah gerak” (sederhana)
        self.publish_debug_point(cmd)

        return cmd

    def publish_debug_point(self, cmd: Twist):
        """Publikasi satu titik di depan sebagai indikasi arah (untuk RViz)."""
        p = PointStamped()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = "base_link"
        # Titik 0.5 m ke arah resultant (linear.x vs angular.z → sangat sederhana)
        # Hanya buat visual, tidak presisi kinematika.
        ahead = 0.5
        p.point.x = ahead
        p.point.y = 0.0
        self.debug_point_pub.publish(p)

    # ========= Main loop =========
    def loop(self):
        safe_cmd = self.compute_safety_filtered_cmd()
        try:
            if self.sdk_ok:
                # Kirim ke Unitree SDK (SportClient)
                # Map Twist → (vx, vy, yaw_rate). Unitree umumnya [m/s, m/s, rad/s]
                vx = float(safe_cmd.linear.x)
                vy = float(safe_cmd.linear.y)
                wz = float(safe_cmd.angular.z)
                # Perhatikan mode (velocity/position). Contoh:
                self.sport.velocity_cmd(vx, vy, wz)  # sesuaikan API jika berbeda
            else:
                # Fallback ke ROS /cmd_vel
                self.cmd_pub.publish(safe_cmd)
        except Exception as e:
            self.get_logger().error(f"Send cmd failed: {e}")

    # ========= Graceful shutdown =========
    def destroy(self):
        if self.sdk_ok:
            try:
                self.sport.velocity_cmd(0.0, 0.0, 0.0)
                ChannelFactoryFinalize()
            except Exception:
                pass
        super().destroy_node()


# ======== entrypoint ========
def main():
    rclpy.init()

    node = SafeBubbleNode()

    # Tangani Ctrl+C dengan rapi
    loop = asyncio.get_event_loop()

    def _shutdown(*args):
        node.get_logger().info("Shutting down…")
        node.destroy()
        rclpy.shutdown()
        try:
            loop.stop()
        except Exception:
            pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown()

if __name__ == "__main__":
    main()
