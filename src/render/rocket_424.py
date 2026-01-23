from dataclasses import MISSING
import numpy as np
import pybullet
from pybullet_utils import bullet_client
import time
import json
import importlib
import meshcat
from pathlib import Path
import umsgpack
from playsound import playsound

colors = {
    'grey': [80/255, 79/255, 88/255, 1.0],
    'purple': [75/255, 47/255, 174/255, 1.0],
    'teal': [10/255, 116/255, 150/255, 1.0],
}

class Simulator:
    def __init__(
                self,
                display=True,
                seed=None,
                dt=0.01,
            ):

        # Time step
        self.dt = dt

        # Aileron angles
        self.delta_1 = 0.
        self.delta_2 = 0.
        self.delta_3 = 0.
        self.delta_4 = 0.

        # Parameters
        self.maximum_aileron_deflection = np.radians(8)        # <-- Max aileron deflection angle in radians
        
        # Connect to and configure pybullet
        self.display_meshcat = display
        self.bullet_client = bullet_client.BulletClient(
            connection_mode=pybullet.DIRECT,
        )
        self.bullet_client.setGravity(0, 0, -9.81)   # <-- +z is up (doesn't work still??)
        self.bullet_client.setPhysicsEngineParameter(
            fixedTimeStep=self.dt,
            numSubSteps=4,
            restitutionVelocityThreshold=0.05,
            enableFileCaching=0,
        )
        
        # Load platform
        self.platform_id = self.bullet_client.loadURDF(
            str(Path('./urdf/ground.urdf')),
            basePosition=np.array([0., 0., 1.56]), # experimental value, prone to change with new rocket cad
            baseOrientation=self.bullet_client.getQuaternionFromEuler([0., 0., np.pi/2]),
            flags=(self.bullet_client.URDF_USE_INERTIA_FROM_FILE),
            useFixedBase=1,
        )
        # Grey #504f58
        self.bullet_client.changeVisualShape(
            objectUniqueId=self.platform_id,
            linkIndex=-1,  # base link
            rgbaColor = colors['grey']  # Grey #504f58
        )
        
        # Load robot
        self.robot_id = self.bullet_client.loadURDF(
            str(Path('./urdf/rocket.urdf')),
            basePosition=np.array([0., 0., 0.]),
            baseOrientation=self.bullet_client.getQuaternionFromEuler([0., -np.pi/2, 0.]),
            flags=(self.bullet_client.URDF_USE_INERTIA_FROM_FILE))
        # Purple #200e45
        self.bullet_client.changeVisualShape(
            objectUniqueId=self.robot_id,
            linkIndex=-1,  # base link
            rgbaColor = colors['purple']  # Medium Purple #4B2FAE  # Lighter Purple #7D5FFF
        )
        # Set contact and damping parameters
        for object_id in [self.robot_id]:
            for joint_id in range(-1, self.bullet_client.getNumJoints(object_id)):
                self.bullet_client.changeDynamics(
                    object_id,
                    joint_id,
                    lateralFriction=1.0,
                    spinningFriction=1.0,
                    rollingFriction=0.0,
                    restitution=0.5,
                    contactDamping=-1,
                    contactStiffness=-1,
                    linearDamping=0.,
                    angularDamping=0.,
                )
        # Maybe need??? temp comment out to make ailerons appear in correct place

        self.joint_name_to_index = {
            self.bullet_client.getJointInfo(self.robot_id, i)[1].decode(): i
            for i in range(self.bullet_client.getNumJoints(self.robot_id))
        }

        # Initialize meshcat if necessary
        if self.display_meshcat:
            self.meshcat_init()

        # Set default camera view
        self.camera_launchview()

    def get_sensor_measurements(self):
        # Position and orientation
        pos, ori = self.bullet_client.getBasePositionAndOrientation(self.robot_id)
        rpy = self.bullet_client.getEulerFromQuaternion(ori)

        # Linear and angular velocity
        vel = self.bullet_client.getBaseVelocity(self.robot_id)
        v_world = np.array(vel[0])
        w_world = np.array(vel[1])
        R_body_in_world = np.reshape(np.array(self.bullet_client.getMatrixFromQuaternion(ori)), (3, 3))
        v_body = R_body_in_world.T @ v_world
        w_body = R_body_in_world.T @ w_world

        # Get components of everything
        p_x, p_y, p_z = pos
        phi, theta, psi = rpy
        v_x, v_y, v_z = v_body
        w_x, w_y, w_z = w_body

        # w_z += ... # TODO: add gyro noise: allan variance -> angle random walk 

        return p_x, p_y, p_z, psi, theta, phi, v_x, v_y, v_z, w_x, w_y, w_z

        # pos, ori = self.bullet_client.getBasePositionAndOrientation(self.robot_id)
        # vel = self.bullet_client.getBaseVelocity(self.robot_id)
        # w_world = np.array(vel[1])
        # R_body_in_world = np.reshape(np.array(self.bullet_client.getMatrixFromQuaternion(ori)), (3, 3))
        # w_body = R_body_in_world.T @ w_world
        # w_x, w_y, w_z = w_body
        # return w_z
    
    def set_actuator_commands(
                self,
                delta_1_command,
                delta_2_command,
                delta_3_command,
                delta_4_command,
            ):
        
        if not np.isscalar(delta_1_command):
            raise Exception('delta_1_command must be a scalar')
        
        if not np.isscalar(delta_2_command):
            raise Exception('delta_2_command must be a scalar')
        
        if not np.isscalar(delta_3_command):
            raise Exception('delta_3_command must be a scalar')
        
        if not np.isscalar(delta_4_command):
            raise Exception('delta_4_command must be a scalar')
        
        self.delta_1 = np.clip(delta_1_command, -self.maximum_aileron_deflection, self.maximum_aileron_deflection)
        self.delta_2 = np.clip(delta_2_command, -self.maximum_aileron_deflection, self.maximum_aileron_deflection)
        self.delta_3 = np.clip(delta_3_command, -self.maximum_aileron_deflection, self.maximum_aileron_deflection)
        self.delta_4 = np.clip(delta_4_command, -self.maximum_aileron_deflection, self.maximum_aileron_deflection)

        return self.delta_1, self.delta_2, self.delta_3, self.delta_4
    
    def reset(
            self,
            initial_conditions=None,
        ):

        if initial_conditions is None:
            initial_conditions = {
                'p_x': 0.,
                'p_y': 0.,
                'p_z': 0.,
                'psi': 0.,
                'theta': -np.pi/2,
                'phi': 0.,
                'v_x': 0.,
                'v_y': 0.,
                'v_z': 0.,
                'w_x': 0.,
                'w_y': 0.,
                'w_z': 0.,
            }

        # Set position and orientation
        self.bullet_client.resetBasePositionAndOrientation(
            self.robot_id,
            [
                initial_conditions['p_x'],
                initial_conditions['p_y'],
                initial_conditions['p_z'],
            ],
            self.bullet_client.getQuaternionFromEuler([
                initial_conditions['phi'],
                initial_conditions['theta'],
                initial_conditions['psi'],
            ]),
        )

        # Get position and orientation again, because it's an easy way to compute the rotation
        # matrix that describes the orientation of the body frame in the world frame
        pos, ori = self.bullet_client.getBasePositionAndOrientation(self.robot_id)
        R_body_in_world = np.reshape(np.array(self.bullet_client.getMatrixFromQuaternion(ori)), (3, 3))
        # Set linear and angular velocity
        # - Define linear velocity in body frame
        v_body = np.array([
            initial_conditions['v_x'],
            initial_conditions['v_y'],
            initial_conditions['v_z'],
        ])
        # - Compute linear velocity in world frame
        v_world = R_body_in_world @ v_body
        # - Define angular velocity in body frame
        w_body = np.array([
            initial_conditions['w_x'],
            initial_conditions['w_y'],
            initial_conditions['w_z'],
        ])
        # - Compute angular velocity in world frame
        w_world = R_body_in_world @ w_body
        # - Set linear and angular velocity in world frame
        self.bullet_client.resetBaseVelocity(
            self.robot_id,
            linearVelocity=v_world,
            angularVelocity=w_world,
        )

        # Set aileron angles
        self.delta_1 = 0.
        self.delta_2 = 0.
        self.delta_3 = 0.
        self.delta_4 = 0.
        
        # Update display
        self._update_display()
    
    def run(
            self,
            controller,
            maximum_time=5.0,
            data_filename=None,
            video_filename=None,
            print_debug=False,
        ):

        self.data = {
            't': [],
            'p_x': [],
            'p_y': [],
            'p_z': [],
            'psi': [],
            'theta': [],
            'phi': [],
            'v_x': [],
            'v_y': [],
            'v_z': [],
            'w_x': [],
            'w_y': [],
            'w_z': [],
            'delta_1_command': [],
            'delta_2_command': [],
            'delta_3_command': [],
            'delta_4_command': [],
            'delta_1': [],
            'delta_2': [],
            'delta_3': [],
            'delta_4': [],
            'f_x': [],
            'f_y': [],
            'f_z': [],
            'tau_x': [],
            'tau_y': [],
            'tau_z': [],
        }
        self.variables_to_log = getattr(controller, 'variables_to_log', [])
        for key in self.variables_to_log:
            if key in self.data.keys():
                raise Exception(f'Trying to log duplicate variable {key} (choose a different name)')
            self.data[key] = []

        # Always start from zero time
        self.t = 0.
        self.time_step = 0
        self.maximum_time_steps = 1 + int(maximum_time / self.dt)
        self.start_time = time.time()

        if video_filename is not None:
            # Import imageio
            imageio = importlib.import_module('imageio')

            # Open video
            fps = 25
            if int(1 / self.dt) % fps != 0:
                raise Exception(f'To create a video, 1 / dt ({1 / self.dt}) must be an ' + \
                                 'integer that is divisible by fps ({fps})')
            if print_debug:
                print(f'Creating a video with name {video_filename} and fps {fps}')
            w = imageio.get_writer(video_filename,
                                   format='FFMPEG',
                                   mode='I',
                                   fps=fps)

            # Add first frame to video
            rgba = self.snapshot()
            w.append_data(rgba)

        while True:
            all_done = self.step(controller)

            self._update_display()

            if video_filename is not None:
                if self.time_step % 100 == 0:
                    if print_debug:
                        print(f' {self.time_step} / {self.maximum_time_steps}')

                # Add frame to video
                if self.time_step % int(1 / (self.dt * fps)) == 0:
                    rgba = self.snapshot()
                    w.append_data(rgba)

            if all_done:
                break

            if (self.maximum_time_steps is not None) and (self.time_step == self.maximum_time_steps):
                break

        if video_filename is not None:
            # Close video
            w.close()

        if data_filename is not None:
            with open(data_filename, 'w') as f:
                json.dump(self.data, f)

        stop_time = time.time()
        stop_time_step = self.time_step

        elapsed_time = stop_time - self.start_time
        elapsed_time_steps = stop_time_step
        if (elapsed_time > 0) and print_debug:
            print(f'Simulated {elapsed_time_steps} time steps in {elapsed_time:.4f} seconds ' + \
                  f'({(elapsed_time_steps / elapsed_time):.4f} time steps per second)')

        # convert lists to numpy arrays
        data = self.data.copy()
        for key in data.keys():
            data[key] = np.array(data[key])

        return data

    def step(self, controller):
        # Never stop early
        all_done = False

        # Get the current time
        self.t = self.time_step * self.dt

        # Get the sensor measurements; #FIXME: these values are outputted from rocketpy simulation
        p_x, p_y, p_z, psi, theta, phi, v_x, v_y, v_z, w_x, w_y, w_z = self.get_sensor_measurements()
        # w_z = self.get_sensor_measurements()
        
        # Get the actuator commands; #FIXME: these values are outputted from rocketpy simulation
        # controller.run() will return the data outputted from the control software
        delta_1_command, delta_2_command, delta_3_command, delta_4_command = controller.run(
            self.t,
            p_x, p_y, p_z, psi, theta, phi, v_x, v_y, v_z, w_x, w_y, w_z,
        )

        # Apply the actuator commands
        delta_1, delta_2, delta_3, delta_4 = self.set_actuator_commands(delta_1_command, delta_2_command, delta_3_command, delta_4_command)
        
        # Get aerodynamic forces and torques
        f_x, f_y, f_z, tau_x, tau_y, tau_z = 0, 0, 0, 0, 0, 0 # <-- #FIXME: replace with your own model!

        # Apply aerodynamic forces
        self.bullet_client.applyExternalForce(
            self.robot_id,
            -1,
            np.array([f_x, f_y, f_z]),
            np.array([0., 0., 0.]),
            self.bullet_client.LINK_FRAME,
        )

        # Apply aerodynamic torques
        self.bullet_client.applyExternalTorque(
            self.robot_id,
            -1,
            np.array([tau_x, tau_y, tau_z]),
            self.bullet_client.LINK_FRAME,
        )

        # Log data
        self.data['t'].append(self.t)
        self.data['p_x'].append(p_x)
        self.data['p_y'].append(p_y)
        self.data['p_z'].append(p_z)
        self.data['psi'].append(psi)
        self.data['theta'].append(theta)
        self.data['phi'].append(phi)
        self.data['v_x'].append(v_x)
        self.data['v_y'].append(v_y)
        self.data['v_z'].append(v_z)
        self.data['w_x'].append(w_x)
        self.data['w_y'].append(w_y)
        self.data['w_z'].append(w_z)

        # MY ADDED LOGGING FOR AILERONS
        self.data['delta_1_command'].append(delta_1_command)
        self.data['delta_2_command'].append(delta_2_command)
        self.data['delta_3_command'].append(delta_3_command)
        self.data['delta_4_command'].append(delta_4_command)

        # MY ADDED LOGGING FOR AILERONS
        self.data['delta_1'].append(delta_1)
        self.data['delta_2'].append(delta_2)
        self.data['delta_3'].append(delta_3)
        self.data['delta_4'].append(delta_4)

        self.data['f_x'].append(f_x)
        self.data['f_y'].append(f_y)
        self.data['f_z'].append(f_z)
        self.data['tau_x'].append(tau_x)
        self.data['tau_y'].append(tau_y)
        self.data['tau_z'].append(tau_z)
        for key in self.variables_to_log:
            val = getattr(controller, key, np.nan)
            if not np.isscalar(val):
                val = val.flatten().tolist()
            self.data[key].append(val)

        # Try to stay real-time
        if self.display_meshcat:
            t = self.start_time + (self.dt * (self.time_step + 1))
            time_to_wait = t - time.time()
            while time_to_wait > 0:
                time.sleep(0.75 * time_to_wait)
                time_to_wait = t - time.time()

        # Take a simulation step
        self.bullet_client.stepSimulation()

        # Increment time step
        self.time_step += 1

        # In your simulation step/controller:
        desired_angles = [self.delta_1, self.delta_2, self.delta_3, self.delta_4]  # radians
        for i, joint_name in enumerate(['aileron1_joint', 'aileron2_joint', 'aileron3_joint', 'aileron4_joint']):
            joint_index = self.joint_name_to_index[joint_name]
            self.bullet_client.setJointMotorControl2(
                self.robot_id,
                joint_index,
                self.bullet_client.POSITION_CONTROL,
                targetPosition=desired_angles[i]
            )

        return all_done
    
    def meshcat_snapshot(self):
        # Get image from visualizer
        rgba = np.asarray(self.vis.get_image())

        # Shrink width and height to be multiples of 16
        height, width, channels = rgba.shape
        m = 16
        return np.ascontiguousarray(rgba[
            :(m * np.floor(height / m).astype(int)),
            :(m * np.floor(width / m).astype(int)),
            :,
        ])
    
    def snapshot(self):
        if self.display_meshcat:
            return self.meshcat_snapshot()
        else:
            raise Exception('you must set display=True in order to take a snapshot')

    def _update_display(self):
        if self.display_meshcat:
            if self.is_catview:
                pos, ori = self.bullet_client.getBasePositionAndOrientation(self.robot_id)
                x, y, z = pos
                y *= -1
                z *= -1
                self.vis.set_cam_pos([x - 4, y - 2.5, z])
                self.vis.set_cam_target([x, y, z])

            self.meshcat_update()

    def camera_catview(self):
        if not self.display_meshcat:
            return
        
        self.is_catview = True
        self._update_display()
    
    def camera_launchview(self):
        if not self.display_meshcat:
            return
        
        self.is_catview = False

        self.vis.set_cam_pos([-3.8, 0., .2])
        self.vis.set_cam_target([0., 0., 0.])
        
        self._update_display()
    
    def _wxyz_from_xyzw(self, xyzw):
        return np.roll(xyzw, 1)

    def _convert_color(self, rgba):
        color = int(rgba[0] * 255) * 256**2 + int(rgba[1] * 255) * 256 + int(rgba[2] * 255)
        opacity = rgba[3]
        transparent = opacity != 1.0
        return {
            'color': color,
            'opacity': opacity,
            'transparent': transparent,
        }

    def meshcat_lights(self):
        # As of 1/21/2025, meshcat-python has a bug that does not allow
        # setting a property with characters that aren't all lower-case.
        # The reason is that, prior to sending commands, the name of the
        # property is converted to lower-case. So, we will DIY it here.

        lights = ['SpotLight', 'PointLightNegativeX', 'PointLightPositiveX']
        intensity = 0.5
        for light in lights:
            self.vis[f'/Lights/{light}'].set_property('visible', True)
            self.vis[f'/Lights/{light}/<object>'].set_property('intensity', intensity)
            cmd_data = {
                u'type': u'set_property',
                u'path': self.vis[f'/Lights/{light}/<object>'].path.lower(),
                u'property': u'castShadow',
                u'value': True,
            }
            self.vis.window.zmq_socket.send_multipart([
                cmd_data['type'].encode('utf-8'),
                cmd_data['path'].encode('utf-8'),
                umsgpack.packb(cmd_data),
            ])
            res = self.vis.window.zmq_socket.recv()
            if res != b'ok':
                raise Exception(f'bad result on meshcat_lights() for light "{light}": {res}')
        
        self.vis['/Lights/PointLightPositiveX/<object>'].set_property('intensity', 1)
        self.vis['/Lights/PointLightPositiveX/<object>'].set_property('distance', 10)
        self.vis['/Lights/PointLightNegativeX/<object>'].set_property('position', [175, 0, -10])
        self.vis['/Lights/PointLightNegativeX/<object>'].set_property('distance', 20)
        self.vis['/Lights/PointLightNegativeX/<object>'].set_property('intensity', 0.75)

    
    def meshcat_init(self):
        import meshcat.geometry as g
        import meshcat.transformations as tf
        # Create a visualizer
        self.vis = meshcat.Visualizer().open()

        # Make sure everything has been deleted from the visualizer
        self.vis.delete()

        # Add platform
        shape_data = self.bullet_client.getVisualShapeData(self.platform_id)
        if len(shape_data) != 1:
            raise Exception(f'platform has bad number of links {len(shape_data)}')
        s = shape_data[0]
        if s[1] != -1:
            raise Exception(f'base link has bad id {s[1]}')
        link_scale = s[3]
        stl_filename = s[4].decode('UTF-8')
        color = self._convert_color(s[7])
        self.vis['platform'].set_object(
            meshcat.geometry.StlMeshGeometry.from_file(stl_filename),
            meshcat.geometry.MeshPhongMaterial(
                color=color['color'],
                transparent=color['transparent'],
                opacity=color['opacity'],
                reflectivity=0.8,
            )
        )

        # Set pose of platform
        pos, ori = self.bullet_client.getBasePositionAndOrientation(self.platform_id)
        S = np.diag(np.concatenate((link_scale, [1.0])))
        Rx = meshcat.transformations.rotation_matrix(np.pi, [1., 0., 0.])
        T = meshcat.transformations.quaternion_matrix(self._wxyz_from_xyzw(ori))
        T[:3, 3] = np.array(pos)[:3]
        self.vis['platform'].set_transform(Rx @ T @ S)
        
        # Add robot
        shape_data = self.bullet_client.getVisualShapeData(self.robot_id)
        if len(shape_data) != 1:
            # raise Exception(f'robot has bad number of links {len(shape_data)}')
            pass
        s = shape_data[0]
        if s[1] != -1:
            raise Exception(f'base link has bad id {s[1]}')
        if not np.allclose(s[3], 1.):
            raise Exception(f'base link has bad scale {s[3]}')
        stl_filename = s[4].decode('UTF-8')
        color = self._convert_color(s[7])
        self.vis['robot'].set_object(
            meshcat.geometry.StlMeshGeometry.from_file(stl_filename),
            meshcat.geometry.MeshPhongMaterial(
                color=color['color'],
                transparent=color['transparent'],
                opacity=color['opacity'],
                reflectivity=0.8,
            )
        )

        # Add first aileron
        color = self._convert_color(colors['teal'])  # Teal #0A7496
        self.vis['robot']['aileron1'].set_object(
            meshcat.geometry.StlMeshGeometry.from_file(str(Path('./urdf/aileron1.stl'))),
            meshcat.geometry.MeshPhongMaterial(
                color=color['color'],
                transparent=color['transparent'],
                opacity=color['opacity'],
                reflectivity=0.8,
            )
        )

        # Add second aileron
        self.vis['robot']['aileron2'].set_object(
            meshcat.geometry.StlMeshGeometry.from_file(str(Path('./urdf/aileron2.stl'))),
            meshcat.geometry.MeshPhongMaterial(
                color=color['color'],
                transparent=color['transparent'],
                opacity=color['opacity'],
                reflectivity=0.8,
            )
        )
        
        # # Add third aileron
        self.vis['robot']['aileron3'].set_object(
            meshcat.geometry.StlMeshGeometry.from_file(str(Path('./urdf/aileron3.stl'))),
            meshcat.geometry.MeshPhongMaterial(
                color=color['color'],
                transparent=color['transparent'],
                opacity=color['opacity'],
                reflectivity=0.8,
            )
        )

        # Add fourth aileron
        self.vis['robot']['aileron4'].set_object(
            meshcat.geometry.StlMeshGeometry.from_file(str(Path('./urdf/aileron4.stl'))),
            meshcat.geometry.MeshPhongMaterial(
                color=color['color'],
                transparent=color['transparent'],
                opacity=color['opacity'],
                reflectivity=0.8,
            )
        )

        # Turn off grid
        self.vis['/Grid'].set_property('visible', True)

        # Add lights
        self.meshcat_lights()

        # Set background color
        self.vis['/Background'].set_property('top_color', [0/255, 159/255, 212/255])
        self.vis['/Background'].set_property('bottom_color', [1, 1, 1])

        # Set clipping range of camera
        self.vis['/Cameras/default/rotated/<object>'].set_property('near', 0.1)
        self.vis['/Cameras/default/rotated/<object>'].set_property('far', 500.)

        # Turn off axes
        self.vis[f'/Axes/<object>'].set_property('visible', True)

        
    def meshcat_update(self):
        # Set pose of robot
        Rx = meshcat.transformations.rotation_matrix(np.pi, [1., 0., 0.])
        pos, ori = self.bullet_client.getBasePositionAndOrientation(self.robot_id)
        T = meshcat.transformations.quaternion_matrix(self._wxyz_from_xyzw(ori))
        T[:3, 3] = np.array(pos)[:3]
        self.vis['robot'].set_transform(Rx @ T)

        # Set position of spotlight
        # self.vis['/Lights/SpotLight/<object>'].set_property('position', [pos[0], -pos[1], -(pos[2] - 1.)])
        self.vis['/Lights/PointLightPositiveX/<object>'].set_property('position', [pos[0], -pos[1], -(pos[2] - 1.)])

        # # Set pose of right elevon
        # T = meshcat.transformations.translation_matrix([-0.23404, 0.15886, 0.008353])
        # R = meshcat.transformations.rotation_matrix(self.delta_r, [-0.37352883778028634, 0.9275991332385393, -0.006004611695959905])
        # self.vis['robot']['elevon-right'].set_transform(T @ R)

        # # Set pose of left elevon
        # T = meshcat.transformations.translation_matrix([-0.23404, -0.15886, 0.008353])
        # R = meshcat.transformations.rotation_matrix(self.delta_l, [0.37352883778028634, 0.9275991332385393, 0.006004611695959905])
        # self.vis['robot']['elevon-left'].set_transform(T @ R)

        import meshcat.transformations as tf

        joint_origins = [
            ([0.0, 0.0, 0.0], [0, 0, 0]),         # aileron1
            ([0.0, 0.0, 0.0], [0, 0, 0]),         # aileron2
            ([0.0, 0.0, 0.0], [0.0, 0, 0]),       # aileron3
            ([0.0, 0.0, 0.0], [0.0, 0, 0]),       # aileron4
        ]

        for i, joint_name in enumerate(['aileron1_joint', 'aileron2_joint', 'aileron3_joint', 'aileron4_joint']):
            # joint_index = self.joint_name_to_index[joint_name]
            joint_index = self.joint_name_to_index[joint_name]
            joint_state = self.bullet_client.getJointState(self.robot_id, joint_index)
            joint_angle = joint_state[0]
            xyz, rpy = joint_origins[i]
            T_origin = tf.compose_matrix(translate=xyz, angles=rpy)
            T_joint = tf.rotation_matrix(joint_angle, [0, 1, 0])
            T = tf.concatenate_matrices(T_origin, T_joint)
            self.vis['robot'][f'aileron{i+1}'].set_transform(T)
