### Set up :

We will use gymnasium for our RL based learning and mujoco as the physics simulator/environment.

pip3 install mujoco
pip3 install gymnasium

git clone https://github.com/WillMandil001/RL_Franka_Pushing.git

Then you need to build the custom environment (INB01014-v0)

pip3 install -e gym-INB0104

### Running the current simulation:
cd src
python3 gym-env-test.py

### editing and adding to the simulation:
- The Gymnasium RL training script is found at /src/gym-env-test.py
- The gymnasium simulation environment is found in /gym-INB0104/envs/INB0104
- The INB0104 .xml head script is found in /environments/INB0104/Robot_C.xml

# ToDo:
- we need to set up the reward to be correct.
- we need to set up the DeepRL velocity controller. (camera RGB image as input)
- Train the DeepRL policy
- implement DeepRL network on the real robot and setup real world velocity controller
- add safety control to the real world controller
- test the system and asses sim-2-real gap for the camera and the robot.
- implement Real World fine-tuning.

