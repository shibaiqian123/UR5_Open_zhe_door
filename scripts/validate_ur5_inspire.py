#!/usr/bin/env python3
"""Asset/physics validation + isolated CPU control/PPO tests.
The CPU test harness is NOT an Isaac Gym simulation and never replaces training.
"""
from pathlib import Path
import argparse,copy,json,subprocess,sys,tempfile,types
import xml.etree.ElementTree as E
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
for folder in ('tasks','utils','algorithms','algorithms/algo_utils','scripts'):
    sys.path.insert(0,str(ROOT/folder))
OUT=ROOT/'assets/ur5_inspire'

def yaml_read(path):
    return json.loads(subprocess.check_output(['python3','-c','import yaml,json,sys; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))',str(path)]))

def rotation(rpy):
    x,y,z=rpy;cx,sx=np.cos(x),np.sin(x);cy,sy=np.cos(y),np.sin(y);cz,sz=np.cos(z),np.sin(z)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])

def fk(robot,values):
    poses={'base_link':np.eye(4)}
    pending=list(robot.findall('joint'))
    while pending:
        count=len(pending)
        for j in pending[:]:
            parent=j.find('parent').get('link')
            if parent not in poses:continue
            origin=j.find('origin');t=np.eye(4)
            if origin is not None:
                t[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                t[:3,:3]=rotation(np.fromstring(origin.get('rpy','0 0 0'),sep=' '))
            if j.get('type')!='fixed':
                axis=np.fromstring(j.find('axis').get('xyz'),sep=' ');q=values[j.get('name')]
                k=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
                t[:3,:3]=t[:3,:3]@(np.eye(3)+np.sin(q)*k+(1-np.cos(q))*(k@k))
            poses[j.find('child').get('link')]=poses[parent]@t;pending.remove(j)
        assert len(pending)<count,'Disconnected/cyclic URDF'
    return poses

def validate_assets():
    robot=E.parse(OUT/'robot_ur5_inspire.urdf').getroot()
    for mesh in robot.findall('.//mesh'):
        assert (OUT/mesh.get('filename')).is_file()
        assert np.all(np.fromstring(mesh.get('scale','1 1 1'),sep=' ')>0)
    for inertial in robot.findall('link/inertial'):
        assert float(inertial.find('mass').get('value'))>0
        d=inertial.find('inertia').attrib
        inertia=np.array([[float(d[a]) for a in row] for row in [('ixx','ixy','ixz'),('ixy','iyy','iyz'),('ixz','iyz','izz')]])
        assert np.linalg.eigvalsh(inertia).min()>0
    model=mujoco.MjModel.from_xml_path(str(OUT/'ur5_inspire.xml'));data=mujoco.MjData(model)
    assert (model.nq,model.nu,model.neq)==(12,7,5)
    rng=np.random.default_rng(10);maximum_error=0
    for _ in range(20):
        q={j.get('name'):rng.uniform(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in robot.findall('joint') if j.get('type')!='fixed'}
        for j in robot.findall('joint'):
            mimic=j.find('mimic')
            if mimic is not None:q[j.get('name')]=q[mimic.get('joint')]*float(mimic.get('multiplier','1'))+float(mimic.get('offset','0'))
        for name,value in q.items():data.qpos[model.joint(name).qposadr[0]]=value
        mujoco.mj_forward(model,data)
        poses=fk(robot,q)
        for name,t in poses.items():
            bid=model.body(name).id
            maximum_error=max(maximum_error,np.max(np.abs(data.xpos[bid]-t[:3,3])),np.max(np.abs(data.xmat[bid].reshape(3,3)-t[:3,:3])))
    assert maximum_error<1e-8
    hold_error=0;coupling_error=0
    hand=[i for i in range(model.nq) if model.joint(i).name.startswith('inspire_')]
    openings=[]
    for key in range(model.nkey):
        mujoco.mj_resetDataKeyframe(model,data,key);mujoco.mj_forward(model,data)
        assert all(c.dist>-.0001 for c in data.contact),'Initial ground penetration'
        initial=data.qpos.copy()
        for _ in range(2000):mujoco.mj_step(model,data)
        assert np.isfinite(data.qpos).all()
        arm=[model.joint(n).qposadr[0] for n in ('shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint')]
        hold_error=max(hold_error,float(np.max(np.abs(data.qpos[arm]-initial[arm]))))
        for target in (.02,.82):
            data.ctrl[6]=target
            for _ in range(2000):mujoco.mj_step(model,data)
            assert np.isfinite(data.qpos).all()
            coupling_error=max(coupling_error,float(np.ptp(data.qpos[hand])))
            openings.append(float(np.linalg.norm(data.xpos[model.body('left_tip_ur5_inspire').id]-data.xpos[model.body('right_tip_ur5_inspire').id])))
    assert hold_error<.02 and coupling_error<.005
    assert openings[0]<.005 and openings[1]>.06 and openings[2]<.005 and openings[3]>.06
    return {'urdf_mjcf_max_error':maximum_error,'hold_max_arm_error_rad':hold_error,'mimic_max_error_rad':coupling_error,'close_open_widths_m':openings}

def validate_reachability():
    import calibrate_ur5_inspire as k
    rng=np.random.default_rng(22);count=0
    # Workspace corners, plus randomized endpoints; cube approach/lift and drawer stroke.
    for task in ('grasp_cube','open_drawer'):
        cfg=yaml_read(ROOT/f'cfg/tasks/{task}_ur5_inspire.yaml')['asset']
        rot=np.diag([1,-1,-1]) if task=='grasp_cube' else np.array([[0,0,1],[1,0,0],[0,1,0]])
        for i in range(12):
            if task=='grasp_cube':
                pos=np.array(cfg['object_root'][:3])+np.r_[rng.uniform(-cfg['reset_range'],cfg['reset_range'],2),0]
                if i%2:pos[2]=cfg['goal'][2]
            else:
                delta=rng.uniform(-cfg['reset_range'],cfg['reset_range'],3)
                yaw=rng.uniform(-cfg['reset_yaw'],cfg['reset_yaw']);r=rotation([0,0,yaw])
                pos=np.array(cfg['object_root'][:3])+delta+r@np.array([-.04-(.08 if i%2 else 0),0,0])
                rot=r@np.array([[0,0,1],[1,0,0],[0,1,0]])
            q,p,_=k.solve(pos,rot)
            assert np.linalg.norm(p-pos)<1e-4;count+=1
    return {'sampled_collision_free_ik_targets':count,'note':'Endpoint reachability, not a collision-free trajectory guarantee'}

def validate_policy_distribution():
    """Check the actual distributions built by all three policy code paths."""
    import torch
    from unittest.mock import patch
    from torch.distributions import MultivariateNormal, kl_divergence
    import actor_critic_ur5_inspire as policy

    torch.set_num_threads(1)
    torch.manual_seed(73)
    cfg={'network':{'name':'MLP','hid_dim':[32,32],'activation':'tanh'},
         'action_std':0.5,'clipAction':1.0,'action_activate':'tanh'}
    model=policy.ActorCritic(43,7,cfg)
    with torch.no_grad():
        for parameter in model.actor.parameters():
            parameter.zero_()
    obs=torch.zeros(32,43)
    captured=[]

    def capture(*args,**kwargs):
        distribution=MultivariateNormal(*args,**kwargs)
        captured.append(distribution)
        return distribution

    with patch.object(policy,'MultivariateNormal',side_effect=capture):
        actions,old_log_prob,_,_,_=model.random_act_cri(obs)
        assert actions.abs().max()<0.99, 'Use non-saturated actions for inverse tanh'
        new_log_prob,_,_,_,_=model.update_act_cri(obs,actions)
        model.random_act(obs)
    assert len(captured)==3
    for distribution in captured:
        torch.testing.assert_close(distribution.stddev,torch.full((32,7),0.5))
        torch.testing.assert_close(distribution.variance,torch.full((32,7),0.25))
    torch.testing.assert_close(old_log_prob,new_log_prob,rtol=1e-5,atol=1e-5)

    # Exercise nonuniform variances and changed means with the real policy distributions.
    with torch.no_grad():
        model.log_std.copy_(torch.linspace(-1.2,0.2,7))
    with patch.object(policy,'MultivariateNormal',side_effect=capture):
        _,_,_,mu,log_std=model.update_act_cri(obs,actions)
    old=captured[0]
    current=MultivariateNormal(mu+torch.linspace(-0.3,0.3,7),
                              scale_tril=captured[-1].scale_tril)
    old_log_std=old.stddev.log()
    kl=(log_std-old_log_std+
        (old_log_std.exp().square()+(old.mean-current.mean).square())/
        (2*log_std.exp().square())-0.5).sum(-1)
    torch.testing.assert_close(kl,kl_divergence(old,current),rtol=1e-5,atol=1e-5)
    return {'policy_distribution':'PASS (all three paths: std=0.5, variance=0.25)',
            'log_probability':'PASS (sampling/update agree before parameter updates)',
            'kl':'PASS (PPO formula matches distribution KL with changed means/stds)'}

def validate_control_and_ppo():
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    from load_robot_ur5_inspire import ur5_inspire,ARM_NAMES,MASTER,collision_filters
    # Only supply the AssetOptions type for loader unit testing. No simulated physics.
    original=sys.modules.get('isaacgym')
    stub=types.ModuleType('isaacgym');stub.gymapi=types.SimpleNamespace(AssetOptions=types.SimpleNamespace)
    stub.gymtorch=types.SimpleNamespace(unwrap_tensor=lambda x:x)
    sys.modules['isaacgym']=stub
    robot_xml=E.parse(OUT/'robot_ur5_inspire.urdf').getroot()
    masks,excluded=collision_filters(robot_xml)
    assert masks['inspire_left_pad'] & masks['inspire_right_pad']==0
    assert masks['upper_arm_link'] & masks['inspire_left_pad']==0
    assert masks['upper_arm_link'] & masks['forearm_link']!=0
    assert masks['inspire_gripper_base'] & masks['wrist_3_link']!=0
    assert all(mask & 1==0 for mask in masks.values()), 'Robot/object collision disabled'
    # Every collision body must be represented, independent of importer ordering.
    assert set(masks)=={n.get('name') for n in robot_xml.findall('link') if n.find('collision') is not None}

    joints=[j for j in robot_xml.findall('joint') if j.get('type')!='fixed'][::-1]
    names={j.get('name'):i for i,j in enumerate(joints)}
    bodies={l.get('name'):i for i,l in enumerate(robot_xml.findall('link'))}
    class LoaderHarness:
        def load_asset(self,*a):return object()
        def get_asset_dof_dict(self,*a):return names
        def get_asset_dof_count(self,*a):return 12
        def get_asset_rigid_body_dict(self,*a):return bodies
        def get_asset_dof_properties(self,*a):
            return {key:np.array([float(j.find('limit').get(key)) for j in joints],dtype=np.float32) for key in ('lower','upper')}
    cfg=yaml_read(ROOT/'cfg/tasks/grasp_cube_ur5_inspire.yaml')
    robot=ur5_inspire(LoaderHarness(),cfg['robot'],1/30,3,'cpu');robot.preload(None,ROOT/'assets')
    class ShapeHarness(LoaderHarness):
        def __init__(self):
            self.shapes=[];self.ranges=[]
            for name in bodies:
                count=2 if name in masks else 0
                self.ranges.append(types.SimpleNamespace(start=len(self.shapes),count=count))
                self.shapes.extend(types.SimpleNamespace(filter=-1,friction=0.) for _ in range(count))
        def get_asset_dof_properties(self,*args):
            props=super().get_asset_dof_properties()
            for key in ('driveMode','effort','stiffness','damping'):props[key]=np.zeros(12)
            return props
        def create_actor(self,*args):return 0
        def set_actor_dof_properties(self,env,actor,props):self.props=props
        def get_actor_rigid_shape_properties(self,*args):return self.shapes
        def get_actor_rigid_body_shape_indices(self,*args):return self.ranges
        def set_actor_rigid_shape_properties(self,env,actor,shapes):self.shapes=shapes
        def set_rigid_body_segmentation_id(self,*args):pass
        def find_actor_rigid_body_index(self,env,actor,name,domain):return bodies[name]
    shape_gym=ShapeHarness()
    stub.gymapi.Transform=types.SimpleNamespace
    stub.gymapi.Vec3=lambda *args:args;stub.gymapi.Quat=lambda *args:args
    stub.gymapi.DOF_MODE_POS=1;stub.gymapi.DOMAIN_ENV=0
    robot.gym=shape_gym;robot.load_to_env(None,0)
    np.testing.assert_allclose(shape_gym.props['effort'],robot.effort.numpy())
    for name,index in bodies.items():
        shape_range=shape_gym.ranges[index]
        for shape in shape_gym.shapes[shape_range.start:shape_range.start+shape_range.count]:
            assert shape.filter==masks[name] and shape.filter & 1==0
            assert shape.friction==1.0

    for attr,name in [('ltip_rb_index','left_tip_ur5_inspire'),('rtip_rb_index','right_tip_ur5_inspire'),('tcp_rb_index','tcp_ur5_inspire')]:setattr(robot,attr,bodies[name])
    model=mujoco.MjModel.from_xml_path(str(OUT/'ur5_inspire.xml'));data=mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,0);mujoco.mj_forward(model,data)
    states=torch.zeros((3,len(bodies),13));dofs=torch.zeros((3,12,2))
    for name,index in bodies.items():
        bid=model.body(name).id;states[:,index,:3]=torch.tensor(data.xpos[bid]);states[:,index,3:7]=torch.tensor(data.xquat[bid][[1,2,3,0]])
    for name,index in names.items():dofs[:,index,0]=float(data.qpos[model.joint(name).qposadr[0]])
    robot.jacobian_tensor=torch.zeros((3,len(bodies)-1,6,12))
    for body in ('left_tip_ur5_inspire','right_tip_ur5_inspire','tcp_ur5_inspire'):
        jp=np.zeros((3,model.nv));jr=jp.copy();mujoco.mj_jacBody(model,data,jp,jr,model.body(body).id)
        for name,index in names.items():
            robot.jacobian_tensor[:,bodies[body]-1,:,index]=torch.tensor(np.r_[jp[:,model.joint(name).dofadr[0]],jr[:,model.joint(name).dofadr[0]]])
    robot.update_state(states,dofs)
    action=torch.zeros((3,7));action[:,0]=1;action[:,6]=-1
    targets=robot.control(action)
    assert torch.isfinite(targets).all() and targets.shape==(3,12)
    assert torch.all(targets>=robot.lower) and torch.all(targets<=robot.upper)
    assert torch.allclose(targets[:,robot.hand_ids],targets[:,robot.master_id,None].expand(-1,6))
    assert torch.max(torch.abs(targets[:,robot.arm_ids]-dofs[:,robot.arm_ids,0]))<=robot.cfg['arm_speed']*robot.dt+1e-6
    before=robot.tip_pos[0].numpy().copy()
    for name,index in names.items():data.qpos[model.joint(name).qposadr[0]]=float(targets[0,index])
    mujoco.mj_forward(model,data)
    after=(data.xpos[model.body('left_tip_ur5_inspire').id]+data.xpos[model.body('right_tip_ur5_inspire').id])/2
    assert after[0]>before[0],'IK direction wrong'
    try:robot.control(torch.full((3,7),float('nan')));raise AssertionError('NaN accepted')
    except ValueError:pass
    # Exercise the actual task observation/reward functions without a Gym backend.
    from grasp_cube_ur5_inspire import grasp_cube_ur5_inspire
    task=grasp_cube_ur5_inspire.__new__(grasp_cube_ur5_inspire)
    task.robot=robot;task.rigid_body_tensor=states;task.dof_state_tensor=dofs
    task.num_envs=3;task.device='cpu';task.obj_actor=1;task.root_tensor=torch.zeros((3,2,13))
    task.root_tensor[:,1,:7]=torch.tensor(cfg['asset']['object_root'])
    task.pose_lower_limit=torch.tensor([.15,-.3,0,-1,-1,-1,-1]);task.pose_upper_limit=torch.tensor([.75,.3,.5,1,1,1,1])
    task.learn_input_mode='normal_state';task.add_proprio_obs=False;task.obs_buf={};task.extras={}
    task.success_pos=torch.tensor(cfg['asset']['goal'])[None];task.goal_thresh=.025
    task.obj_default_root=torch.tensor(cfg['asset']['object_root']);task.progress_buf=torch.zeros(3)
    task.compute_observations();task.compute_reward(action)
    assert task.obs_buf['normal_state'].shape==(3,43) and torch.isfinite(task.rew_buf).all()
    from open_drawer_ur5_inspire import open_drawer_ur5_inspire
    drawer=open_drawer_ur5_inspire.__new__(open_drawer_ur5_inspire)
    drawer.__dict__.update(task.__dict__)
    drawer.dof_state_tensor_all=torch.cat([dofs,torch.zeros((3,1,2))],dim=1).reshape(-1,2)
    drawer.dof_state_mask=torch.arange(39).reshape(3,13)
    drawer.rigid_body_tensor_all=states.reshape(-1,13);drawer.rigid_body_mask=torch.arange(3*len(bodies)).reshape(3,-1)
    corners=json.loads((OUT/'objects_ur5_inspire/bbox_info_ur5_inspire.json').read_text())['bbox_world'][2]
    drawer.part_bbox_init=torch.tensor(corners)[None].repeat(3,1,1);drawer.part_axis_dir_init=torch.tensor([[-1.,0,0]]).repeat(3,1)
    drawer.root_tensor=task.root_tensor.clone();drawer.root_tensor[:,1,:7]=torch.tensor([.54,0,.35,0,0,0,1])
    drawer.part_joint_lower_limits=torch.zeros(3);drawer.part_joint_upper_limits=torch.ones(3)*.16
    drawer.suc_prop=.5;drawer.succ_objid_lst=torch.zeros(1,dtype=torch.bool);drawer.obj_lstid_lst=torch.zeros(3,dtype=torch.long)
    drawer.compute_observations();drawer.compute_reward(action)
    assert drawer.obs_buf['normal_state'].shape==(3,53) and torch.isfinite(drawer.rew_buf).all()
    class ResetHarness:
        def __init__(self):self.calls=[]
        def set_dof_state_tensor_indexed(self,sim,state,ids,count):self.calls.append(('dof',ids.clone()))
        def set_actor_root_state_tensor_indexed(self,sim,state,ids,count):self.calls.append(('root',ids.clone()))
        def set_dof_position_target_tensor(self,sim,targets):self.targets=targets.clone()
    for obj in (task,drawer):
        obj.gym=ResetHarness();obj.sim=None;obj.random_reset=False
        obj.global_indices=torch.arange(6,dtype=torch.int32).reshape(3,2)
        obj.pos_act=torch.rand((3,12));preserved=obj.pos_act[[0,2]].clone()
        obj.pos_act_all=torch.zeros(36 if obj is task else 39)
        if obj is task:obj.dof_state_mask=torch.arange(36).reshape(3,12)
        obj.progress_buf=torch.ones(3,dtype=torch.long)*4
        obj.success=torch.ones(3,dtype=torch.bool)
        obj.epis_max_rew=torch.ones(3);obj.epis_max_step=torch.ones(3,dtype=torch.long)
        obj.reset_idx(torch.tensor([False,True,False]))
        assert torch.equal(obj.pos_act[[0,2]],preserved),'Reset changed another environment target'
        assert torch.equal(obj.pos_act[1],robot.default_dof_pos)
        assert obj.progress_buf.tolist()==[4,0,4]
        expected=[2] if obj is task else [2,3]
        assert obj.gym.calls[0][1].tolist()==expected
        assert obj.gym.calls[1][1].tolist()==[2,3]
        obj.reset_idx(torch.ones(3,dtype=torch.bool))
        assert torch.equal(obj.pos_act,robot.default_dof_pos.repeat(3,1))
    # A blocked finger must not erase its commanded target on the next step.
    robot.reset_targets(torch.arange(3))
    close=torch.zeros((3,7));close[:,6]=-1
    first=robot.control(close)[:,robot.master_id].clone()
    second=robot.control(close)[:,robot.master_id].clone()
    assert torch.all(second<first), 'Gripper command follows measured q instead of persistent target'
    held=robot.control(torch.zeros_like(close))[:,robot.master_id].clone()
    assert torch.equal(held,second), 'Zero gripper action must hold the target'
    robot.reset_targets(torch.tensor([1]))
    assert robot.master_target[1]==robot.default_dof_pos[robot.master_id]
    assert torch.equal(robot.master_target[[0,2]],second[[0,2]])
    assert torch.all(robot.effort>0) and robot.effort.numel()==12
    for _ in range(40):robot.control(close)
    assert torch.all(robot.master_target>=robot.lower[robot.master_id])
    assert torch.all(robot.action_tensor>=robot.lower)
    if original is not None:sys.modules['isaacgym']=original
    else:del sys.modules['isaacgym']
    # A small tensor environment verifies the real PPO update + checkpoint interface.
    from ppo_ur5_inspire import ppo
    class TensorHarness:
        def __init__(self,width):
            self.num_envs=4;self.num_actions=7;self.num_obs={'normal_state':width};self.max_episode_length=4
            self.robot=types.SimpleNamespace(signature=ur5_inspire.signature);self.x=torch.zeros(4,width)
            self.reset_succ=torch.zeros(4,dtype=torch.bool);self.train_test_flag='train'
        def reset(self):self.x.zero_();return {'normal_state':self.x.clone()}
        def step(self,a,save_image_path=None):
            self.x[:,:7]+=a*.01;self.rew_buf=1-a.square().mean(-1)
            return {'normal_state':self.x.clone()},self.rew_buf,torch.zeros(4,dtype=torch.bool),{'succ_rate':torch.zeros(1)}
    with tempfile.TemporaryDirectory(prefix='ppo_ur5_inspire_') as tmp:
        for width in (43,53):
            ac=yaml_read(ROOT/'cfg/algos/ppo_ur5_inspire.yaml')
            ac.update(num_envs=4,n_steps=4,n_updates=1,n_minibatches=2,max_iterations=2,eval_frequence=100,save_frequence=2,device='cpu',resume=None,test_only=False,save_pose=False,save_video=False,succ_value=None)
            ac['model']['network']['hid_dim']=[32,32];ac['model']['clipAction']=1.
            logger=types.SimpleNamespace(save_ckpt_dir=tmp,info=lambda *args:None)
            runner=ppo(TensorHarness(width),ac,logger);initial=copy.deepcopy(runner.actor_critic.state_dict());runner.run()
            assert any(not torch.equal(initial[k],v) for k,v in runner.actor_critic.state_dict().items())
            ac['resume']=str(Path(tmp)/'model_2.pth');resumed=ppo(TensorHarness(width),ac,logger)
            assert resumed.curr_iter==2
            for k,v in runner.actor_critic.state_dict().items():assert torch.equal(v,resumed.actor_critic.state_dict()[k])
            checkpoint=torch.load(ac['resume']);checkpoint['robot_signature']='franka';torch.save(checkpoint,ac['resume'])
            try:ppo(TensorHarness(width),ac,logger);raise AssertionError('Franka checkpoint accepted')
            except ValueError:pass
    return {'control':'PASS (indices, limits, effort, persistent targets, isolated target resets, coupling, IK, NaN rejection)', 'collision_filters':'PASS (named collision bodies, adjacent exclusions, pad contact and external collisions retained)', 'task_tensors':'PASS (43/53 dimensions, finite rewards, partial/full reset isolation)', 'ppo_cpu_harness':'PASS (two updates each, checkpoint roundtrip, incompatible checkpoint rejected)', 'isaac_gym_simulation':'NOT RUN: unavailable; CPU harness does not validate PhysX'}

def validate_contact_metrics():
    """Negative cases prevent a fallen object or GPU-only evidence passing."""
    from validate_contact_ur5_inspire import evaluate_grasp,evaluate_stability
    record=dict(q=np.zeros((1,12)),tcp=np.array([[.45,0,.20]]),obj=np.array([[.45,0,.20]]),
                width=np.array([.05]),coupling=np.array([.001]),drawer=np.array([.10]),
                pairs=[[True,True]],overlap=[.001])
    records=[copy.deepcopy(record) for _ in range(10)]
    checks,_=evaluate_grasp(records,records,.001,'grasp_cube','cpu')
    assert all(checks.values())
    checks,_=evaluate_grasp(records,records,.001,'open_drawer','cpu')
    assert all(checks.values())
    checks,_=evaluate_grasp(records,records,.001,'grasp_cube','cuda')
    assert checks['bilateral_target_contact'] is None and checks['penetration'] is None
    bad=copy.deepcopy(records);bad[-1]['obj'][0,2]=.10
    checks,_=evaluate_grasp(bad,bad,.001,'grasp_cube','cpu')
    assert not checks['height'] and not checks['drop'] and not checks['slip']
    bad=copy.deepcopy(records)
    for t in bad:t['pairs']=[[True,False]]
    checks,_=evaluate_grasp(bad,bad,.001,'grasp_cube','cpu')
    assert not checks['bilateral_target_contact']
    bad=copy.deepcopy(records);bad[-1]['overlap']=[.003];bad[-1]['coupling'][0]=.03
    checks,_=evaluate_grasp(bad,bad,.001,'grasp_cube','cpu')
    assert not checks['penetration'] and not checks['coupling']
    bad=copy.deepcopy(records);bad[-1]['drawer'][0]=.07
    checks,_=evaluate_grasp(bad,bad,.001,'open_drawer','cpu')
    assert not checks['stroke'] and not checks['retreat']
    bad=copy.deepcopy(records);bad[-1]['q'][0,0]=.03;bad[-1]['tcp'][0,0]+=.01
    checks,_=evaluate_stability(bad,list(range(6)),'cpu')
    assert not checks['joint_drift'] and not checks['tcp_drift']
    return 'PASS (missing contact, slip, drop, penetration, coupling, jitter and GPU evidence gaps)'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-reachability',action='store_true');args=parser.parse_args()
    report={'assets':validate_assets()}
    if not args.skip_reachability:report['reachability']=validate_reachability()
    report['policy']=validate_policy_distribution()
    report['contact_metrics']=validate_contact_metrics()
    report['code']=validate_control_and_ppo()
    path=OUT/'validation_ur5_inspire.json';path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
