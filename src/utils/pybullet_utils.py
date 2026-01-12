import pybullet as p

def update_camera_position(drone_state, distance, yaw_offset, pitch):
    target_pos = drone_state[0:3]    
    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw_offset, 
        cameraPitch=pitch,
        cameraTargetPosition=target_pos
    )
