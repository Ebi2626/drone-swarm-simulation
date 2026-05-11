"""Helpery PyBullet — m.in. ustawienie kamery na śledzonego drona."""
import pybullet as p


def update_camera_position(drone_state, distance, yaw_offset, pitch):
    """Ustaw kamerę PyBullet na pozycji `drone_state[0:3]` z zadanym dystansem/yaw/pitch."""
    target_pos = drone_state[0:3]
    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw_offset, 
        cameraPitch=pitch,
        cameraTargetPosition=target_pos
    )
