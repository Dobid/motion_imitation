import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bullet_client

NUM_JOINTS_SIM = 16

ref_col = [1, 1, 1, 1]
pyb = bullet_client.BulletClient(connection_mode=pybullet.GUI)
pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
urdf_file = "laikago/laikago_toes_limits.urdf"
ref_model = pyb.loadURDF(urdf_file, useFixedBase=True)

pyb.changeDynamics(ref_model, -1, linearDamping=0, angularDamping=0)
pyb.setCollisionFilterGroupMask(ref_model, -1, collisionFilterGroup=0, collisionFilterMask=0)
pyb.changeDynamics(ref_model, -1, activationState=pyb.ACTIVATION_STATE_SLEEP + 
                                                  pyb.ACTIVATION_STATE_ENABLE_SLEEPING +
                                                  pyb.ACTIVATION_STATE_DISABLE_WAKEUP)
pyb.changeVisualShape(ref_model, -1, rgbaColor=ref_col)

num_joints = pyb.getNumJoints(ref_model)
num_joints_sim = NUM_JOINTS_SIM