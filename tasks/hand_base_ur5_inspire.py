from isaacgym import gymapi, gymtorch



from torch_jit_utils_ur5_inspire import *
from load_robot_ur5_inspire import ur5_inspire

import numpy as np
import torch
import os 
import math 
import sys
from copy import deepcopy

class BaseTask():
    def __init__(self, cfg, sim_params=None):
        self.cfg = cfg
        self.gym = gymapi.acquire_gym()
        self.up_axis = gymapi.UP_AXIS_Z
        self.dt = sim_params.dt
        self.physics_engine = cfg['physics_engine']
        self.asset_root = cfg["asset"]["assetRoot"]
        self.device = cfg['device']
        self.device_id = cfg['device_id']
        self.save_video = cfg['save_video']
        self.add_mask = cfg['add_mask']
        self.learn_input_mode = cfg['learn_input_mode']   # compute in utils/config.py
        self.add_proprio_obs = cfg['add_proprio_obs']
        print('Learning mode: ', self.learn_input_mode)
        print('Add proprio obs: ', self.add_proprio_obs)
        self.headless = cfg['headless']
        self.graphics_device_id = cfg['graphics_device_id']
        if not self.save_video and self.headless and 'depth' not in self.learn_input_mode:
            self.graphics_device_id = -1
        print('graphics card:', self.graphics_device_id)
        self.num_envs = cfg["num_envs"]
        self.max_episode_length = cfg["maxEpisodeLength"]
        self.control_freq_inv = cfg["controlFrequencyInv"]
        self.clip_actions = cfg['clipActions']
        self.clip_obs = cfg['clipObservations']
        # self.reset_time = cfg['reset_time']
        self.robot = ur5_inspire(self.gym, cfg['robot'], self.dt * self.control_freq_inv, self.num_envs, self.device)
        self.num_actions = self.robot.num_actions

        self.num_obs = {}
        for obs_mode in cfg['obs_mode'].keys():
            self.num_obs[obs_mode] = cfg['obs_mode'][obs_mode]
        if 'tsdf' in self.learn_input_mode:
            self.num_obs[self.learn_input_mode] = cfg['obs_mode']['tsdf']['resolution'] ** 3
        self.tsdf_size = cfg['obs_mode']['tsdf']['size']
        self.tsdf_resolution = cfg['obs_mode']['tsdf']['resolution']
        self.tsdf_origin = cfg['obs_mode']['tsdf']['origin']
        if self.add_proprio_obs:
            self.num_obs[self.learn_input_mode] += self.num_obs['proprio_state']
        
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = {}
        self.extras = {}
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.train_test_flag = 'train'
        self.explore_step = cfg['explore_step']
        self.epis_max_rew = -100 * torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        self.epis_max_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # create envs, sim
        if self.up_axis == gymapi.UP_AXIS_Z:
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
        else:
            raise NotImplementedError
        self.sim = self.gym.create_sim(self.device_id, self.graphics_device_id, self.physics_engine, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()
        self.load_scene()
        self.gym.prepare_sim(self.sim)

        self.robot.init_jacobian()
        self.num_obs["normal_state"] = (19 if cfg["task"].startswith("grasp_cube") else 29) + 2*self.robot.num_dofs
        self.num_obs["proprio_state"] = 7 + 2*self.robot.num_dofs
        # create viewer
        self.enable_viewer_sync = True
        self.viewer = None
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            
            # set the camera position based on up axis
            if self.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(1.4, -1.3, 1.7)
                cam_target = gymapi.Vec3(0, 0, 0.7)
            else:
                cam_pos = gymapi.Vec3(10, 10, 3.0)
                cam_target = gymapi.Vec3(0, 0, 0.0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # init camera
        # camera settings
        if self.save_video:
            raise ValueError("Use the MJCF viewer for inspection; training video is disabled")


        
        return 

    ###########################################################
    ######       create scenes
    ###########################################################

    def load_scene(self):
        # create ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.,0.,1.)
        plane_params.static_friction = 0.1
        plane_params.dynamic_friction = 0.1
        self.gym.add_ground(self.sim, plane_params)

        # arrange envs
        spacing = self.cfg['envSpacing']
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.space_middle = torch.zeros((self.num_envs, 3), device=self.device)
        self.space_range = torch.zeros((self.num_envs, 3), device=self.device)
        self.space_middle[:, 0] = self.space_middle[:, 1] = 0
        self.space_middle[:, 2] = spacing/2
        self.space_range[:, 0] = self.space_range[:, 1] = spacing
        self.space_middle[:, 2] = spacing/2
        num_per_row = int(np.sqrt(self.num_envs))

        # create envs
        self.preload_all_obj()
        self.robot.preload(self.sim, self.asset_root)
        self.env_ptr_list = []
        for env_id in range(self.num_envs):    
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.env_ptr_list.append(env_ptr)
            self.robot.load_to_env(env_ptr, env_id)
            self.load_obj(env_ptr, env_id)
        self.obj_actor = 1
        return 
        
    def preload_all_obj(self,):
        pass 

    def load_obj(self, env_ptr, env_id):
        raise NotImplementedError
    
    def step(self, actions, save_image_path=None):
        '''
        To add a new task, overwrite 'self.reset_idx()', 'self.compute_observations()' and 'self.compute_reward()'

        Pipeline in time T:
            -- pre_physics_step()
                1. compute actions (using self.robot.control())
                2. compute 'self.reset_buf' using 'self.rew_buf' and 'self.progress_buf' in time T-1
                3. reset some envs using 'self.reset_idx()',
                    or directly using gym.set_dof_position_target_tensor to take actions.
                    NOTE: In 'self.reset_idx()', those envs who don't need to reset 
                    should take actions using gym.set_dof_position_target_tensor.

            -- post_physics_step()
                update 'self.progress_buf'.
                -- refresh_gym_tensor(): Must refresh gym tensors before compute observation.
                -- compute_observations(): compute 'self.obs_buf'
                -- compute_reward(): compute 'self.rew_buf' using updated 'self.obs_buf'

        return:
            self.obs_buf (If self.reset_buf False, obs_{T-1} + action_T -> obs_T)
            self.rew_buf 
            self.reset_buf (If True, then this transition shouldn't be used for training!)
            self.extra (Debug)
        
        '''
        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.render(save_image_path=save_image_path)

        # compute observations, rewards, resets, ...
        self.post_physics_step(actions)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def render(self, save_image_path=None):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)
        return

    def pre_physics_step(self, actions):
        # print(actions[5,:3])
        self.pos_act = self.robot.control(actions)
        
        if self.train_test_flag == 'train':
            # self.reset_buf = (self.progress_buf >= self.max_episode_length) | self.success
            self.epis_max_step = torch.where(self.rew_buf < self.epis_max_rew, self.epis_max_step, self.progress_buf)
            self.epis_max_rew = torch.maximum(self.rew_buf, self.epis_max_rew)
            self.reset_buf = (self.progress_buf >= self.max_episode_length) | (self.progress_buf >= self.epis_max_step + self.explore_step) | self.success
            self.reset_succ = deepcopy(self.success)
            self.extras['succ_rate'] = self.success.int().sum(dim=-1, keepdim=True) / torch.clamp(self.reset_buf.int().sum(), min=1)
        elif self.train_test_flag == 'test':
            self.reset_buf = self.progress_buf >= self.max_episode_length
        else:
            raise NotImplementedError

        if self.reset_buf.sum() > 0:
            self.reset_idx(self.reset_buf)
        else:
            self.pos_act_all[self.dof_state_mask[:, :self.robot.num_dofs]] = self.pos_act
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_act_all))

        return

    def post_physics_step(self, actions):
        self.progress_buf += 1
        self.refresh_gym_tensor()
        self.compute_observations()
        obs = self.obs_buf["normal_state"]
        if obs.shape != (self.num_envs, self.num_obs["normal_state"]) or not torch.isfinite(obs).all():
            raise RuntimeError("Invalid UR5 observation shape or non-finite state")
        self.obs_buf["normal_state"] = obs.clamp(-self.clip_obs, self.clip_obs)
        self.compute_reward(actions)
        if not torch.isfinite(self.rew_buf).all():
            raise RuntimeError("Non-finite UR5 reward")
        return 

    def reset(self,type="reset"):
        to_reset = torch.ones(self.num_envs, device=self.device)
        self.reset_idx(to_reset)
        self.gym.simulate(self.sim)     # Necessary for updating rigid_body_state!!
        self.gym.fetch_results(self.sim, True)
        self.render()
        self.refresh_gym_tensor()
        self.compute_observations(type=type)
        return self.obs_buf

    def refresh_gym_tensor(self) :
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return 
    
    def reset_idx(self, to_reset):
        """
        Reset envs where to_reset[env_ids] = 1.
        For to_reset[env_ids] = 1, set dof + action + root.
        For to_reset[env_ids] != 1, set action.
        """
        raise NotImplementedError

    def compute_reward(self, action):
        raise NotImplementedError
    
    def compute_observations(self):
        raise NotImplementedError

    ###########################################################
    ######       useful functions
    ###########################################################

