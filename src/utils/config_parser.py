import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def _parse_drone_model(model_input) -> DroneModel:
    if isinstance(model_input, DroneModel):
        return model_input
    
    if isinstance(model_input, str):
        try:
            return DroneModel[model_input]
        except KeyError:
            print(f"[WARN] Unknown drone model '{model_input}', using default CF2X.")
            return DroneModel.CF2X
            
    return DroneModel.CF2X

def _parse_physics(physics_input) -> Physics:
    if isinstance(physics_input, Physics):
        return physics_input
        
    if isinstance(physics_input, str):
        try:
            return Physics[physics_input]
        except KeyError:
            print(f"[WARN] Unknown physics '{physics_input}', using default PYB.")
            return Physics.PYB
            
    return Physics.PYB

def _parse_coordinates(coords_input):
    if coords_input is None:
        return None
    # Needed to cast to list() to fix Hydra ListConfig issue
    return np.array(list(coords_input))

def sanitize_init_params(drone_model, physics, xyzs, rpys):
    return (
        _parse_drone_model(drone_model),
        _parse_physics(physics),
        _parse_coordinates(xyzs),
        _parse_coordinates(rpys)
    )
