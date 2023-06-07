import mujoco
import time
import mujoco.viewer

import mediapy as media
import matplotlib.pyplot as plt

m = mujoco.MjModel.from_xml_path("/home/willow/Robotics/RL_Franka_Pushing/environments/INB0104/Robot_C.xml")
d = mujoco.MjData(m)
mujoco.mj_step(m, d)

print([m.geom(i).name for i in range(m.ngeom)])
print(m.geom("ball_object_pushing"))

print("d.qpos: ", d.qpos)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 120:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)







