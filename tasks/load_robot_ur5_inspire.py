"""Fixed UR5 + single-actuator Inspire EG2-4C2 adapter (Isaac Gym).
All simulation indices are resolved by name. Quaternions use xyzw.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import torch

ARM_NAMES = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
             'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')
MASTER = 'inspire_gripper_joint'

def quat_matrix(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x,y,z,w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1],3,3)

def damped_ik(jacobian, dpose, damping):
    eye=torch.eye(6,device=jacobian.device,dtype=jacobian.dtype)
    return (jacobian.transpose(-1,-2) @ torch.linalg.solve(jacobian @ jacobian.transpose(-1,-2)+damping**2*eye,dpose.unsqueeze(-1))).squeeze(-1)

class ur5_inspire:
    num_actions = 7
    mobile = False
    # Meaning/version, not just dimensionality: prevents accidentally loading Franka weights.
    signature = 'ur5_eg2_v1_fixed_cartesian6_opening1_all12jointstate'
    max_opening = 0.067

    def __init__(self,gym,robot_cfg,dt,num_envs,device):
        self.gym,self.cfg,self.dt,self.num_envs,self.device=gym,robot_cfg,dt,num_envs,device
        if robot_cfg['driveMode'] != 'ik': raise ValueError('UR5 adapter supports incremental Cartesian IK only')
        self.driveMode='ik'; self.asset_file=robot_cfg['assetFile']
        self.default_root=torch.tensor(robot_cfg['root'],device=device,dtype=torch.float)
        if self.default_root.shape != (7,) or abs(self.default_root[3:].norm().item()-1)>1e-4:
            raise ValueError('robot.root must be xyz + normalized xyzw quaternion')

    def preload(self,sim,asset_root):
        from isaacgym import gymapi
        self.sim=sim
        source=Path(asset_root)/self.asset_file
        xml=ET.parse(source).getroot()
        for mesh in xml.findall('.//mesh'):
            if not (source.parent/mesh.get('filename')).is_file():
                raise FileNotFoundError(mesh.get('filename'))
        options=gymapi.AssetOptions()
        options.fix_base_link=True
        options.disable_gravity=False
        options.flip_visual_attachments=False
        options.collapse_fixed_joints=False
        options.thickness=0.001
        options.armature=0.001
        options.vhacd_enabled=True
        self.robot_asset=self.gym.load_asset(sim,str(asset_root),self.asset_file,options)
        if self.robot_asset is None: raise RuntimeError('Failed to load '+str(source))
        names=self.gym.get_asset_dof_dict(self.robot_asset)
        self.num_dofs=self.gym.get_asset_dof_count(self.robot_asset)
        self.body_dict=self.gym.get_asset_rigid_body_dict(self.robot_asset)
        self.num_rigid_body=len(self.body_dict)
        joints={j.get('name'):j for j in xml.findall('joint') if j.get('type')!='fixed'}
        if set(names)!=set(joints) or self.num_dofs!=12:
            raise RuntimeError('Importer must expose six UR5 and six EG2 joints; got '+str(names))
        self.arm_ids=torch.tensor([names[n] for n in ARM_NAMES],device=self.device)
        self.master_id=names[MASTER]
        self.hand_names=[n for n in names if n not in ARM_NAMES]
        self.hand_ids=torch.tensor([names[n] for n in self.hand_names],device=self.device)
        coefficients=[]
        for name in self.hand_names:
            m=joints[name].find('mimic')
            if m is None:
                if name!=MASTER: raise ValueError('Unexpected uncoupled hand joint '+name)
                coefficients.append((1.,0.))
            else:
                if m.get('joint')!=MASTER: raise ValueError('Expected a direct EG2 master relation')
                coefficients.append((float(m.get('multiplier','1')),float(m.get('offset','0'))))
        self.multipliers=torch.tensor([c[0] for c in coefficients],device=self.device)
        self.offsets=torch.tensor([c[1] for c in coefficients],device=self.device)
        props=self.gym.get_asset_dof_properties(self.robot_asset)
        self.lower=torch.tensor(props['lower'].copy(),device=self.device)
        self.upper=torch.tensor(props['upper'].copy(),device=self.device)
        if torch.any(self.upper<=self.lower): raise ValueError('Invalid DOF limits')
        self.velocity=torch.tensor([float(joints[n].find('limit').get('velocity')) for n in sorted(names,key=names.get)],device=self.device)
        self.default_dof_pos=torch.zeros(self.num_dofs,device=self.device)
        self.default_dof_pos[self.arm_ids]=torch.tensor(self.cfg['arm_positions'],device=self.device)
        self.default_dof_pos[self.hand_ids]=self.cfg['gripper_open']*self.multipliers+self.offsets
        if torch.any(self.default_dof_pos<self.lower) or torch.any(self.default_dof_pos>self.upper): raise ValueError('Reset joint pose exceeds limits')
        self.action_tensor=self.default_dof_pos.repeat(self.num_envs,1)
        # ACTOR indices for Jacobian rows. Environment state indices are resolved separately.
        self.jac_left=self.body_dict['left_tip_ur5_inspire']-1
        self.jac_right=self.body_dict['right_tip_ur5_inspire']-1
        self.jac_tcp=self.body_dict['tcp_ur5_inspire']-1

    def load_to_env(self,env_ptr,env_id):
        from isaacgym import gymapi
        props=self.gym.get_asset_dof_properties(self.robot_asset)
        props['driveMode'][:]=gymapi.DOF_MODE_POS
        for ids,kind in ((self.arm_ids,'arm'),(self.hand_ids,'gripper')):
            indices=ids.cpu().numpy()
            props['stiffness'][indices]=self.cfg[kind+'_stiffness']
            props['damping'][indices]=self.cfg[kind+'_damping']
        pose=gymapi.Transform()
        pose.p=gymapi.Vec3(*self.cfg['root'][:3]);pose.r=gymapi.Quat(*self.cfg['root'][3:])
        # Bit 2 disables robot self collision; drawer uses bit 1, cube uses 0.
        # Distinct masks retain robot-object collision within each environment.
        self.robot_actor=self.gym.create_actor(env_ptr,self.robot_asset,pose,'ur5_inspire',env_id,2,1)
        self.gym.set_actor_dof_properties(env_ptr,self.robot_actor,props)
        shapes=self.gym.get_actor_rigid_shape_properties(env_ptr,self.robot_actor)
        for shape in shapes: shape.friction=1.0
        self.gym.set_actor_rigid_shape_properties(env_ptr,self.robot_actor,shapes)
        for index in range(self.num_rigid_body):
            self.gym.set_rigid_body_segmentation_id(env_ptr,self.robot_actor,index,1)
        if env_id==0:
            def idx(n):return self.gym.find_actor_rigid_body_index(env_ptr,self.robot_actor,n,gymapi.DOMAIN_ENV)
            self.ltip_rb_index=idx('left_tip_ur5_inspire');self.rtip_rb_index=idx('right_tip_ur5_inspire')
            self.tcp_rb_index=idx('tcp_ur5_inspire')
            if min(self.ltip_rb_index,self.rtip_rb_index,self.tcp_rb_index)<0: raise RuntimeError('Missing TCP/tip body')

    def init_jacobian(self):
        from isaacgym import gymtorch
        self.jacobian_tensor=gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim,'ur5_inspire'))
        if self.jacobian_tensor.shape[-1]!=self.num_dofs: raise RuntimeError('Expected fixed-base Jacobian')

    def update_state(self,rigid_body_tensor,dof_state_tensor):
        self.ltip_rb_tensor=rigid_body_tensor[:,self.ltip_rb_index,:]
        self.rtip_rb_tensor=rigid_body_tensor[:,self.rtip_rb_index,:]
        self.tip_rb_tensor=rigid_body_tensor[:,self.tcp_rb_index,:].clone()
        # Never average quaternions: orientation is the fixed gripper frame.
        self.tip_rb_tensor[:,:3]=(self.ltip_rb_tensor[:,:3]+self.rtip_rb_tensor[:,:3])/2
        self.tip_rb_tensor[:,7:10]=(self.ltip_rb_tensor[:,7:10]+self.rtip_rb_tensor[:,7:10])/2
        self.tip_pos=self.tip_rb_tensor[:,:3]
        self.tip_rot_9d=quat_matrix(self.tip_rb_tensor[:,3:7])
        self.gripper_length=(self.ltip_rb_tensor[:,:3]-self.rtip_rb_tensor[:,:3]).norm(dim=-1)
        self.dof_qpos_raw=dof_state_tensor[:,:self.num_dofs,0]
        self.dof_qvel_raw=dof_state_tensor[:,:self.num_dofs,1]
        self.dof_qpos_normalized=2*(self.dof_qpos_raw-self.lower)/(self.upper-self.lower)-1

    def control(self,raw_output):
        if raw_output.shape!=(self.num_envs,7) or not torch.isfinite(raw_output).all():
            raise ValueError('Expected finite (num_envs,7) actions')
        actions=raw_output.clamp(-1,1)
        linear=(self.jacobian_tensor[:,self.jac_left,:3,:]+self.jacobian_tensor[:,self.jac_right,:3,:])/2
        angular=self.jacobian_tensor[:,self.jac_tcp,3:,:]
        jac=torch.cat((linear,angular),dim=1).index_select(2,self.arm_ids)
        error=torch.cat((actions[:,:3]*self.cfg['translation_step'],actions[:,3:6]*self.cfg['rotation_step']),dim=-1)
        delta=damped_ik(jac,error,self.cfg['ik_damping'])
        limit=self.velocity[self.arm_ids].clamp(max=self.cfg['arm_speed'])*self.dt
        target=self.dof_qpos_raw.clone()
        target[:,self.arm_ids]+=torch.maximum(torch.minimum(delta,limit),-limit)
        hand_speed=torch.min(self.velocity[self.hand_ids]/self.multipliers.abs().clamp_min(1e-9)).clamp(max=self.cfg['gripper_speed'])
        master=(self.dof_qpos_raw[:,self.master_id]+actions[:,6]*hand_speed*self.dt).clamp(self.lower[self.master_id],self.upper[self.master_id])
        target[:,self.hand_ids]=master[:,None]*self.multipliers+self.offsets
        self.action_tensor=torch.maximum(torch.minimum(target,self.upper),self.lower)
        return self.action_tensor
