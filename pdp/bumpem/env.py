import os
import copy
import random
import collections
from functools import partial

import torch
import numpy as np
import mujoco
from scipy.interpolate import interp1d
from gym import utils
from gym.utils.renderer import Renderer
from gym.envs.mujoco import Viewer, RenderContextOffscreen
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.5)),
    "elevation": 0.0,
    "azimuth": 180.0,
}
DEFAULT_SIZE = 480

# Config for initializing the Skeleton Mujoco environment
ENV_CONFIG = {
    'env_id': 'Skeleton',
    'xml_file': 'assets/S2_model.xml',
    'render_mode': 'rgb_array',
    'init_type': 'init_dist', 
    'init_path': 'motions/S07DN_init_dist',
    'ref_path': None,

    'obs_size': 181,
    'frame_skip': 2,
    'max_ep_time': 6,
    
    'pert': {
        'active': True,
        'p_phs': .325,
        'imp_time': None, 
        'p_frc_frac': 0.15,
        'p_ang': 90,
        'p_dur': .3,
    },
    
    'rand_pert':{
        'active': False,
        'imp_time': [0, 2], 
        'p_frc_frac': [0.0745, 0.15],
        'p_dur': .3,
    },
}


class MujocoEnv:
    """Superclass for MuJoCo environments."""

    def __init__(
        self,
        model_path,
        frame_skip,
        render_mode=None,
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        if model_path.startswith("/"):
            self.fullpath = model_path
        else:
            self.fullpath = os.path.join(os.path.dirname(__file__), model_path)

        if not os.path.exists(self.fullpath):
            raise OSError(f"File {self.fullpath} does not exist")

        self.frame_skip = frame_skip
        self._initialize_simulation()
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self._viewers = {}
        self.viewer = None

        # defined metadata here and removed asserts, annoying to redefine render_fps everytime...
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "single_rgb_array",
                "single_depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.render_mode = render_mode
        render_frame = partial(
            self._render,
            width=width,
            height=height,
            camera_name=camera_name,
            camera_id=camera_id,
        )
        self.renderer = Renderer(self.render_mode, render_frame)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def _initialize_simulation(self):
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = mujoco.MjData(self.model)

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def set_state(self, qpos, qvel=0):
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        mujoco.mj_forward(self.model, self.data)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=n_frames)  # nframes defined as frame_skip for some reason... force it to be 1.
        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def _render(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        assert mode in self.metadata["render_modes"]

        if mode in {
            "rgb_array",
            "single_rgb_array",
            "depth_array",
            "single_depth_array",
        }:
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name,
                )

                self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode in {"rgb_array", "single_rgb_array"}:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode in {"depth_array", "single_depth_array"}:
            self._get_viewer(mode).render(width, height)
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def render(
        self,
        mode="human",
        width=None,
        height=None,
        camera_id=None,
        camera_name=None,
    ):
        if self.render_mode is not None:
            assert (
                width is None
                and height is None
                and camera_id is None
                and camera_name is None
            ), "Unexpected argument for render. Specify render arguments at environment initialization."
            return self.renderer.get_renders()
        else:
            width = width if width is not None else DEFAULT_SIZE
            height = height if height is not None else DEFAULT_SIZE
            return self._render(
                mode=mode,
                width=width,
                height=height,
                camera_id=camera_id,
                camera_name=camera_name,
            )

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def _get_viewer(self, mode, width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = Viewer(self.model, self.data)
            elif mode in {
                "rgb_array",
                "depth_array",
                "single_rgb_array",
                "single_depth_array",
            }:
                self.viewer = RenderContextOffscreen(
                    width, height, self.model, self.data
                )
            else:
                raise AttributeError(
                    f"Unexpected mode: {mode}, expected mmodes: {self.metadata['render_modes']}"
                )

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer


class Skeleton(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": -1,  # redefined
    }

    def __init__(self, config=ENV_CONFIG, **kwargs):
        args = config
        self.xml_file = args['xml_file']
        self.frame_skip = args['frame_skip']
        self.render_mode = args['render_mode']
        
        MujocoEnv.__init__(self, self.xml_file, self.frame_skip, render_mode=self.render_mode, **kwargs)
        utils.EzPickle.__init__(self, args, **kwargs)

        self.agent_obs_size = args['obs_size']
        self.xml_file = args['xml_file']

        self.ref_path =  args['ref_path']
        self.init_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            args['init_path']
        )
        assert os.path.exists(self.init_path), f"init_path {self.init_path} does not exist"

        self.p_active = args['pert']['active']
        self.p_phs = args['pert']['p_phs']
        self.p_frc_frac = args['pert']['p_frc_frac']
        self.p_frc = 9.81 * 59 * self.p_frc_frac
        self.p_ang = args['pert']['p_ang']
        self.p_dur = args['pert']['p_dur']
        self.imp_time = args['pert']['imp_time']
        
        self.rand_pert_active = False
        if 'rand_pert' in args:
            self.rand_pert_active = args['rand_pert']['active']
            self.rand_imp_time_range = args['rand_pert']['imp_time']
            self.rand_p_frc_frac_range = args['rand_pert']['p_frc_frac']
            self.p_dur = args['rand_pert']['p_dur']
            
            if self.rand_pert_active and self.p_active: 
                raise ValueError('Both perturbation and random perturbation cannot be active at the same time')

        self.init_type = args['init_type']
        self.maybe_load_init_dist(self.init_path)

        """"""
        # State trackers
        self.force_already_applied = False
        self.force_being_applied = False
        self.initial_phase_offset = None
        self.force_end_time = None
        self.force_start_time = None
        self.prev_time = None
        self.signal = 0
        self.lhs_signal = 0
        self.sim_step = 0

        self.foot_state = 'wait_lhs'
        self.completed_gait_cycles = 0
        self.lhs_time = collections.deque(maxlen=3)
        self.gait_cycle_time = None
        self.new_gait_cycle = False
        self.phase_est = None
        self.cycle_start_time = None
        self.prev_phase = None
                
        self.left_foot_first_contact_pos = None
        self.right_foot_first_contact_pos = None

        action_bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        self.action_bounds_low = action_bounds[:, 0]
        self.action_bounds_high = action_bounds[:, 1]

        self.frame_skip = args['frame_skip']
        self.max_ep_time = args['max_ep_time']
        self.render_mode = args['render_mode']
        
    def maybe_load_init_dist(self, path):
        if self.init_type == 'init_dist':
            assert path is not None
            self.init_qpos_mean = np.load(path + '/init_qpos_mean.npy')
            self.init_qpos_std = np.load(path + '/init_qpos_std.npy')
            self.init_qvel_mean = np.load(path + '/init_qvel_mean.npy')
            self.init_qvel_std = np.load(path + '/init_qvel_std.npy')
            self.init_qvel_std[np.where(self.init_qvel_std == 0)] = .1

    @property
    def done(self):
        done = self.data.body('torso').xpos[2] <= .6
        done = done or self.data.time > self.max_ep_time
        return done

    def get_obs(self):
        exlude_idxs = [0, 4, 9, 14, 18, 16, 20, 15, 19, 13, 17, 21]  # world, talus_r, talus_l, ulna_r, ulna_l, hand_r, hand_l, radius_r, radius_l, humerus_r, humerus_l, toes_r, toes_l, treadmill

        xipos = np.delete(self.data.xipos.copy(), exlude_idxs, axis=0)
        ximat = np.delete(self.data.ximat.copy(), exlude_idxs, axis=0)
        cvel = np.delete(self.data.cvel.copy(), exlude_idxs, axis=0)
        # Signal for inference
        if self.force_being_applied:
            self.signal = 1
        else:
            self.signal *= .85

        self.signal = self.signal if self.signal >= 1e-3 else 0
        
        observation = np.hstack((xipos.flatten(), ximat.flatten(), cvel.flatten(), self.signal)) ##, self.p_frc))
        return observation

    def apply_force_phase(self):
        # reset force_applied flag 3 seconds after pert
        if self.force_end_time is not None :
            if self.data.time >= self.force_end_time + 4 and self.force_already_applied:
                self.force_already_applied = False

        if self.phase_est is not None and self.prev_phase is not None and self.data.time > 3:
            current_phase = self.phase_est % 1
            prev_phase = self.prev_phase % 1
            phase_crossed = (prev_phase < self.p_phs <= current_phase) or (prev_phase > current_phase and self.p_phs in [0, 1])
            if not self.force_being_applied and not self.force_already_applied and phase_crossed:
                self.force_being_applied = True
                self.force_start_time = self.data.time
                self.force_end_time = self.force_start_time + self.p_dur

                self.data.qfrc_applied[0] = np.cos(self.p_ang * np.pi / 180) * self.p_frc
                self.data.qfrc_applied[2] = np.sin(self.p_ang * np.pi / 180) * self.p_frc
                self.model.geom_rgba = np.array([1, 1, 1, 1])

            elif self.force_being_applied and self.data.time > self.force_end_time:
                self.force_already_applied = True
                self.force_being_applied = False
                self.data.qfrc_applied[:] = 0
                self.model.geom_rgba = np.array([.7, .5, .3, 1])

    def update_foot_contact_positions(self):
        # returns the first 
        lhs = self.data.body('calcn_l').cfrc_ext[5]
        rhs = self.data.body('calcn_r').cfrc_ext[5]

        left_foot = np.array([np.inf, np.inf])
        right_foot = np.array([np.inf, np.inf])
        root_pos = np.array([np.inf, np.inf])
        
        if self.data.time > .1 and self.force_already_applied: # crude filter
            if lhs and not self.first_left_contact: 
                x_pos = self.data.body('calcn_l').xpos[0] # x, y (skel env defined in y up coordinate system)
                y_pos = -self.data.body('calcn_l').xpos[1] # x, y (skel env defined in y up coordinate system)
                left_foot = np.array([y_pos, x_pos])

                x_root = self.data.body('pelvis').xpos[0]
                y_root = -self.data.body('pelvis').xpos[1]
                root_pos = np.array([y_root, x_root])

                self.left_foot_first_contact_pos = left_foot.copy()
                self.left_foot_first_contact_root_pos = root_pos.copy()
                self.first_left_contact = True

            if rhs and self.first_left_contact and not self.first_right_contact: 
                x_pos = self.data.body('calcn_r').xpos[0] # x, y (skel env defined in y up coordinate system)
                y_pos = -self.data.body('calcn_r').xpos[1] # x, y (skel env defined in y up coordinate system)
                right_foot = np.array([y_pos, x_pos])

                x_root = self.data.body('pelvis').xpos[0]
                y_root = -self.data.body('pelvis').xpos[1]
                root_pos = np.array([y_root, x_root])

                self.right_foot_first_contact_pos = right_foot.copy()
                self.right_foot_first_contact_root_pos = root_pos.copy()
                self.first_right_contact = True

        return np.hstack((left_foot, right_foot))

    def get_episode_foot_contact_positions(self):
        return np.hstack((self.left_foot_first_contact_pos, self.right_foot_first_contact_pos))

    def get_episode_foot_contact_positions_root(self):
        return np.hstack((self.left_foot_first_contact_root_pos, self.right_foot_first_contact_root_pos))

    def apply_force_sim(self, imp_time=None):
        if self.imp_time is None: 
            self.imp_time = 0 #self.model.geom_rgba = np.array([.7, .5, .3, 1])

        if self.imp_time <= self.data.time <= self.imp_time +.3:
            self.data.qfrc_applied[0] = np.cos(self.p_ang * np.pi / 180) * self.p_frc
            self.data.qfrc_applied[2] = np.sin(self.p_ang * np.pi / 180) * self.p_frc
            self.model.geom_rgba[[1, 2, 22], :] = np.array([1, 1, 1, 1])
            self.force_being_applied = True

        else:
            self.data.qfrc_applied[:] = 0
            self.model.geom_rgba[[1, 2, 22, 23, 27], :] = np.array([.7, .5, .3, 1])
            self.force_being_applied = False
            if self.data.time > self.imp_time + .3:
                self.force_already_applied = True

        self.prev_time = copy.copy(self.data.time)  

    def apply_force_eval(self):
        if self.phase_est is None:
            return False

        if self.phase_est >= self.p_phs and self.phase_est <= self.p_phs + self.p_dur: 
            self.data.qfrc_applied[0] = np.cos(self.p_ang * np.pi / 180) * self.p_frc
            self.data.qfrc_applied[2] = np.sin(self.p_ang * np.pi / 180) * self.p_frc
            self.model.geom_rgba[[1, 2, 22, 23, 27], :] = np.array([1, 1, 1, 1])
            self.force_being_applied = True

        else:
            self.data.qfrc_applied[:] = 0
            self.model.geom_rgba[[1, 2, 22, 23, 27], :] = np.array([.7, .5, .3, 1])
            self.force_being_applied = False

        self.prev_time = copy.copy(self.data.time)
        
    def estimate_phase(self):
        if self.phase_est is not None:
            self.prev_phase = self.phase_est.copy()

        lhs = self.data.body('calcn_l').cfrc_ext[5]
        rhs = self.data.body('calcn_r').cfrc_ext[5]

        both_feet = lhs > 0 and rhs > 0

        self.lhs_signal = 0

        if self.foot_state == 'wait_lhs':
            if lhs > 0:
                self.cycle_start_time = self.data.time
                self.lhs_signal = 1

                self.lhs_time.append(self.cycle_start_time)
                self.new_gait_cycle = True
                if len(self.lhs_time) > 1:
                    self.gait_cycle_time = np.mean(np.diff(self.lhs_time))
                    # print(self.gait_cycle_time)
                self.foot_state = 'wait_rhs'
        elif self.foot_state == 'wait_rhs':
            if rhs > 0 and not both_feet:
                self.foot_state = 'wait_lhs'

        if self.gait_cycle_time is not None and self.new_gait_cycle:
            self.completed_gait_cycles += 1
            self.new_gait_cycle = False

        if self.gait_cycle_time is not None:
            self.phase_est = self.completed_gait_cycles + (self.data.time - self.cycle_start_time)/self.gait_cycle_time

    def reset_model(self):
        if self.rand_pert_active:
            self.p_frc_frac = np.random.uniform(self.rand_p_frc_frac_range[0], self.rand_p_frc_frac_range[1])
            self.p_frc = self.p_frc_frac * 9.81 * 59
            self.p_frc = np.random.choice([self.p_frc, -self.p_frc])
            self.imp_time = np.random.uniform(self.rand_imp_time_range[0], self.rand_imp_time_range[1])

        if self.init_type=='init_dist':
            # This is for checking how diverse the policy can be given the identical starting state. add some noise when we test for robustness.   
            self.initial_phase_offset = 0
            qpos = np.random.normal(self.init_qpos_mean, self.init_qpos_std)
            qvel = np.random.normal(self.init_qvel_mean, self.init_qvel_std)
        else:
            raise ValueError(f'init_type {self.init_type} not recognized')

        self.set_state(qpos, qvel)

        self.force_already_applied = False
        self.force_being_applied = False
        self.first_left_contact = False
        self.first_right_contact = False
        
        self.left_foot_first_contact_pos = np.array([np.inf, np.inf]) 
        self.right_foot_first_contact_pos = np.array([np.inf, np.inf])
        self.left_foot_first_contact_root_pos = np.array([np.inf, np.inf])
        self.right_foot_first_contact_root_pos = np.array([np.inf, np.inf])

        self.foot_state = 'wait_lhs'
        self.signal = 0
        self.lhs_signal = 0

        self.completed_gait_cycles = 0
        self.lhs_time = collections.deque(maxlen=5)
        self.gait_cycle_time = None
        self.new_gait_cycle = False
        self.data.qfrc_applied[:] = 0
        self.model.geom_rgba[[1, 2, 22, 23, 27], :] = np.array([.7, .5, .3, 1])

        self.prev_phase = None
        self.sim_step = 0

        self.force_end_time = None
        self.force_start_time = None
        self.prev_time = None
        self.phase_est = None

        observation = self.get_obs()
        return observation

    def reset(self):
        self._reset_simulation()
        obs = self.reset_model()
        self.renderer.reset()
        self.renderer.render_step()
        return obs

    def compute_torque(self, qtarget):
        qpos = self.data.qpos[6:-1].copy()
        qvel = self.data.qvel[6:-1].copy()
        kp, kd = 1, 0.05
        torque = kp * (qtarget - qpos) - kd * qvel
        return torque
    
    def step(self, action):
        self.pose = action
        for _ in range(self.frame_skip):
            torque = self.compute_torque(action)
            self.data.qvel[-1] = -1.25 # Treadmill

            self._step_mujoco_simulation(ctrl=torque, n_frames=1)
            self.sim_step += 1

            self.estimate_phase()
            if self.p_active or self.rand_pert_active:
                self.apply_force_sim(imp_time=None) # can give a simulation time, probably between 0 and (gait cycle approx .7 seconds, so perhaps set this between 0 and 3, so it can walk a little bit) 

        self.update_foot_contact_positions()
        foot_pos = self.get_episode_foot_contact_positions()

        self.renderer.render_step()
        obs = self.get_obs()
        reward = None
        done = self.done
        info = {
            'foot_pos': foot_pos,
        }

        # if self.render_mode == 'rgb_array':
        rgb = self.render(mode=self.render_mode)
        info['rgb'] = rgb[0].astype(np.uint8)

        return obs, reward, done, info
