import omni.isaac.lab.sim as sim_utils

cone_spawn_cfg = sim_utils.ConeCfg(
    radius=0.15,
    height=0.5,
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
)
cone_spawn_cfg.func(
    "/World/Cone", cone_spawn_cfg, translation=(0.0, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
)


cone_spawn_cfg = sim_utils.ConeCfg(
    radius=0.15,
    height=0.5,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
)
cone_spawn_cfg.func(
    "/World/Cone", cone_spawn_cfg, translation=(0.0, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
)

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

anymal_spawn_cfg = sim_utils.UsdFileCfg(
    usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=0,
        fix_root_link=True,
    ),
)
anymal_spawn_cfg.func(
    "/World/ANYmal", anymal_spawn_cfg, translation=(0.0, 0.0, 0.8), orientation=(1.0, 0.0, 0.0, 0.0)
)