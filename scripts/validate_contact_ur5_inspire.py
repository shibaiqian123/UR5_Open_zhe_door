#!/usr/bin/env python3
"""Scripted PhysX validation; never attach, teleport, or drive task objects.
CPU contact pairs are required for a contact PASS; GPU reports evidence gaps.
"""
from pathlib import Path
import argparse,csv,json,sys,math,traceback
ROOT=Path(__file__).resolve().parents[1]
for folder in ('tasks','utils'):
    sys.path.insert(0,str(ROOT/folder))

def contact_overlap_ok(records,device):
    return None if device!='cpu' else max(x for t in records for x in t['overlap'])<=.002

def evaluate_stability(records,arm_ids,device):
    import numpy as np
    qs=np.stack([t['q'][:,arm_ids] for t in records]);ts=np.stack([t['tcp'] for t in records])
    objs=np.stack([t['obj'] for t in records])
    drift=float(np.abs(qs-qs[0]).max());tdrift=float(np.linalg.norm(ts-ts[0],axis=-1).max())
    object_drift=float(np.linalg.norm(objs-objs[0],axis=-1).max())
    return {'joint_drift':drift<=.02,'tcp_drift':tdrift<=.005,'object_drift':object_drift<=.01,
            'penetration':contact_overlap_ok(records,device)},dict(joint_drift=drift,tcp_drift=tdrift,object_drift=object_drift)

def evaluate_grasp(records,all_records,reach_error,task,device):
    import numpy as np
    ts=np.stack([t['tcp'] for t in records]);os_=np.stack([t['obj'] for t in records])
    widths=np.stack([t['width'] for t in records]);ds=np.stack([t['drawer'] for t in records])
    jitter=float(np.linalg.norm(np.ptp(ts,axis=0),axis=-1).max());wj=float(np.ptp(widths,axis=0).max())
    ce=float(np.stack([t['coupling'] for t in records]).max())
    slip=float(np.linalg.norm((os_-ts)-(os_[0]-ts[0]),axis=-1).max())
    pairs=None if device!='cpu' else bool(np.array([t['pairs'] for t in records]).all(axis=-1).mean(axis=0).min()>=.9)
    checks={'reach':reach_error<=.015,'bilateral_target_contact':pairs,'penetration':contact_overlap_ok(all_records,device),
            'tcp_jitter':jitter<=.005,'opening_jitter':wj<=.002,'coupling':ce<=.02}
    if task=='grasp_cube':checks.update(height=bool(os_[:,:,2].min()>=.18),drop=bool((os_[0,:,2]-os_[:,:,2]).max()<=.01),slip=slip<=.01)
    else:checks.update(stroke=bool(ds.min()>=.08),retreat=bool((ds[0]-ds).max()<=.005))
    metrics=dict(reach_error=reach_error,tcp_jitter=jitter,opening_jitter=wj,coupling=ce,slip=slip,min_height=float(os_[:,:,2].min()),min_stroke=float(ds.min()))
    return checks,metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=['grasp_cube','open_drawer'],required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--num-envs',type=int,default=1)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--trials',type=int,default=10)
    p.add_argument('--suite',choices=['all','stability','grasp','reset'],default='all')
    p.add_argument('--baseline',action='store_true',help='Reproduce legacy collision masks and measured-position gripper targets; effort limits stay explicit')
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.trials<1 or a.num_envs<1:p.error('counts must be positive')
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    import isaacgym
    from isaacgym import gymapi,gymtorch
    import torch,numpy as np,yaml,os
    from config_ur5_inspire import process_cfgs
    from load_robot_ur5_inspire import quat_matrix
    from grasp_cube_ur5_inspire import grasp_cube_ur5_inspire
    from open_drawer_ur5_inspire import open_drawer_ur5_inspire
    os.chdir(ROOT)
    sys.argv=[sys.argv[0],'--taskcfg',a.task+'_ur5_inspire','--algocfg','ppo_ur5_inspire',
              '--device_type',a.device,'--algo.num_envs',str(a.num_envs)]
    if a.viewer:sys.argv.append('--headless')
    cfg,sim=process_cfgs()
    sim.physx.contact_collection=gymapi.ContactCollection.CC_ALL_SUBSTEPS
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    if a.task=='open_drawer' and not cfg['task']['asset']['use_test_object']:
        raise ValueError('Contact validator requires the bundled test drawer')
    cfg['task']['random_reset']=False
    if a.baseline:
        cfg['task']['robot']['selective_self_collision']=False
        cfg['task']['robot']['persistent_gripper_target']=False
    report={'arguments':vars(a),'config':yaml.safe_load((ROOT/'cfg/tasks'/f'{a.task}_ur5_inspire.yaml').read_text()),
            'robot_config':cfg['task']['robot'],'tests':[],'contact_pairs_available':a.device=='cpu',
            'note':'Coupled-position approximation; no physical closed-loop linkage. GPU net forces do not identify contact pairs.'}
    env=None
    f=(out/'trajectory.csv').open('w',newline='');writer=csv.DictWriter(f,fieldnames=['trial','stage','step','env','q','qd','targets','tcp','object','opening','coupling','pad_forces','drawer','contacts','overlap','tracking_error','tcp_quaternion','object_quaternion'])
    writer.writeheader();step=0;trial_name='init';stage='init'
    try:
        env={'grasp_cube':grasp_cube_ur5_inspire,'open_drawer':open_drawer_ur5_inspire}[a.task](cfg['task'],sim)
        r=env.robot;dt=env.dt*env.control_freq_inv
        report['effort_limits']=r.effort.cpu().tolist()
        report['collision_masks']=r.collision_masks if not a.baseline else {name:2 for name in r.collision_masks}
        report['collision_exclusions']=r.collision_exclusions if not a.baseline else 'all robot internal pairs'
        report['sim_dt']=env.dt
        report['sim_substeps']=sim.substeps
        report['control_frequency_inv']=env.control_freq_inv
        net=gymtorch.wrap_tensor(env.gym.acquire_net_contact_force_tensor(env.sim)).view(a.num_envs,-1,3)
        pads=[env.gym.find_actor_rigid_body_index(env.env_ptr_list[0],0,n,gymapi.DOMAIN_ENV) for n in ('inspire_left_pad','inspire_right_pad')]
        target_name='cube' if a.task=='grasp_cube' else 'handle'
        obj_id=env.gym.find_actor_rigid_body_index(env.env_ptr_list[0],1,target_name,gymapi.DOMAIN_ENV)
        assert min(pads+[obj_id])>=0
        zeros=torch.zeros((a.num_envs,7),device=env.device)
        def sample():
            nonlocal step
            q=r.dof_qpos_raw.detach();tcp=r.tip_pos.detach()
            expected=q[:,r.master_id,None]*r.multipliers+r.offsets
            coupling=(q[:,r.hand_ids]-expected).abs().max(-1).values
            if a.task=='grasp_cube':obj=env.root_tensor[:,1,:3]
            else:obj=(env.part_bbox[:,0]+env.part_bbox[:,6])/2
            drawer=torch.zeros(a.num_envs,device=env.device) if a.task=='grasp_cube' else env.dof_state_tensor[:,-1,0]-env.part_joint_lower_limits
            pair_flags=[];overlaps=[]
            for i in range(a.num_envs):
                flags=[False,False];overlap=0.
                if a.device=='cpu':
                    contacts=env.gym.get_env_rigid_contacts(env.env_ptr_list[i])
                    for c in contacts:
                        b0,b1=int(c['body0']),int(c['body1'])
                        overlap=max(overlap,float(c['initial_overlap']))
                        for k,pad in enumerate(pads):
                            if {b0,b1}=={pad,obj_id}:flags[k]=True
                pair_flags.append(flags if a.device=='cpu' else None);overlaps.append(overlap if a.device=='cpu' else None)
                writer.writerow(dict(trial=trial_name,stage=stage,step=step,env=i,
                    q=json.dumps(q[i].tolist()),qd=json.dumps(r.dof_qvel_raw[i].tolist()),targets=json.dumps(env.pos_act[i].tolist()),
                    tcp=json.dumps(tcp[i].tolist()),object=json.dumps(obj[i].tolist()),opening=float(r.gripper_length[i]),coupling=float(coupling[i]),
                    pad_forces=json.dumps(net[i,pads].tolist()),drawer=float(drawer[i]),contacts=json.dumps(pair_flags[-1]),overlap=overlaps[-1],tracking_error=float((env.pos_act[i]-q[i]).abs().max()),tcp_quaternion=json.dumps(r.tip_rb_tensor[i,3:7].tolist()),object_quaternion=json.dumps(env.root_tensor[i,1,3:7].tolist())))
            step+=1
            if not all(torch.isfinite(x).all() for x in (q,obj,r.dof_qvel_raw,tcp,net)):raise RuntimeError('nonfinite_state')
            if ((q<r.lower-.02)|(q>r.upper+.02)).any():raise RuntimeError('joint_limit_violation')
            return {'q':q.cpu().numpy().copy(),'tcp':tcp.cpu().numpy().copy(),'obj':obj.cpu().numpy().copy(),
                    'width':r.gripper_length.cpu().numpy().copy(),'coupling':coupling.cpu().numpy().copy(),
                    'drawer':drawer.cpu().numpy().copy(),'pairs':pair_flags,'overlap':overlaps}
        def tick(action=None,targets=None):
            env.pos_act=(r.control(zeros if action is None else action) if targets is None else targets.clone())
            env.pos_act_all[env.dof_state_mask[:,:r.num_dofs]]=env.pos_act
            env.gym.set_dof_position_target_tensor(env.sim,gymtorch.unwrap_tensor(env.pos_act_all))
            for _ in range(env.control_freq_inv):
                env.gym.simulate(env.sim);env.gym.fetch_results(env.sim,True);env.render()
            env.refresh_gym_tensor();env.compute_observations()
            return sample()
        def result(name,checks,metrics):
            status='FAIL' if any(v is False for v in checks.values()) else ('UNVERIFIED' if any(v is None for v in checks.values()) else 'PASS')
            report['tests'].append(dict(name=name,status=status,checks=checks,metrics=metrics,failed_checks=[k for k,v in checks.items() if v is False],unverified_checks=[k for k,v in checks.items() if v is None]))
            print('CONTACT_TEST',name,status,json.dumps(metrics),flush=True)
        def reset(random=False):
            env.random_reset=random;env.reset();env.compute_reward(zeros)
        if a.suite in ('all','stability'):
            for mode in ('zero_action','fixed_target'):
                for trial in range(a.trials):
                    trial_name=f'{mode}_{trial}';stage='settle';reset(False)
                    targets=r.default_dof_pos.repeat(a.num_envs,1) if mode=='fixed_target' else None
                    initial=sample()
                    settling=[tick(targets=targets) for _ in range(round(1/dt))]
                    stage='hold';records=[tick(targets=targets) for _ in range(round(3/dt))]
                    checks,metrics=evaluate_stability(records,r.arm_ids.cpu().numpy(),a.device)
                    checks['penetration']=contact_overlap_ok(settling+records,a.device)
                    excursion=float(np.linalg.norm(np.stack([t['obj'] for t in settling+records])-initial['obj'],axis=-1).max())
                    checks['reset_object_excursion']=excursion<=.05
                    metrics['reset_object_excursion']=excursion
                    result(trial_name,checks,metrics)
        if a.suite in ('all','grasp'):
            for trial in range(a.trials+1):
                trial_name='fixed' if trial==0 else f'random_{trial}';stage='settle';reset(trial!=0)
                for _ in range(round(1/dt)):tick(targets=r.default_dof_pos.repeat(a.num_envs,1))
                initial=sample();target=torch.tensor(initial['obj'],device=env.device)
                if a.task=='grasp_cube':
                    yaw=torch.atan2(quat_matrix(env.root_tensor[:,1,3:7])[:,1,0],quat_matrix(env.root_tensor[:,1,3:7])[:,0,0])
                    yaw=(yaw+math.pi/4)%(math.pi/2)-math.pi/4
                    rot=torch.zeros((a.num_envs,3,3),device=env.device)
                    rot[:,0,0]=yaw.cos();rot[:,1,0]=yaw.sin();rot[:,0,1]=yaw.sin();rot[:,1,1]=-yaw.cos();rot[:,2,2]=-1
                    approach=target.clone();approach[:,2]=.15
                else:
                    base=torch.tensor([[0.,0,1],[1,0,0],[0,1,0]],device=env.device)
                    rot=quat_matrix(env.root_tensor[:,1,3:7])@base
                    approach=target-.06*rot[:,:,2]
                def feedback(goal,grip):
                    action=zeros.clone();action[:,:3]=((goal-r.tip_pos)/r.cfg['translation_step']).clamp(-.4,.4)
                    err=.5*sum(torch.cross(r.tip_rot_9d[:,:,i],rot[:,:,i],dim=-1) for i in range(3))
                    action[:,3:6]=(err/r.cfg['rotation_step']).clamp(-.5,.5);action[:,6]=grip
                    return action
                all_records=[]
                for stage,goal,grip,seconds in [('approach',approach,0.,3.),('reach',target,0.,4.),('close',target,-.4,2.)]:
                    for _ in range(round(seconds/dt)):all_records.append(tick(feedback(goal,grip)))
                reach_error=float(torch.linalg.vector_norm(r.tip_pos-target,dim=-1).max())
                lift=target.clone()
                if a.task=='grasp_cube':lift[:,2]=.20
                else:lift=target-.10*rot[:,:,2]
                stage='move'
                for _ in range(round(4/dt)):all_records.append(tick(feedback(lift,0.)))
                stage='hold';records=[tick(feedback(lift,0.)) for _ in range(round(2/dt))];all_records+=records
                checks,metrics=evaluate_grasp(records,all_records,reach_error,a.task,a.device)
                result(trial_name,checks,metrics)
                stage='release'
                for _ in range(round(1/dt)):tick(feedback(lift,.5))
        if a.suite in ('all','reset'):
            trial_name='reset_isolation';stage='reset';reset(False)
            probe=zeros.clone();probe[:,0]=torch.linspace(-.1,.1,a.num_envs,device=env.device);probe[:,6]=-.2
            for _ in range(5):tick(probe)
            master=r.master_target.clone()
            root=env.root_tensor.clone();q=env.dof_state_tensor.clone();targets=env.pos_act.clone()
            mask=torch.zeros(a.num_envs,dtype=torch.bool,device=env.device);mask[0]=True
            env.reset_idx(mask)
            isolated=torch.equal(root[1:],env.root_tensor[1:]) and torch.equal(q[1:],env.dof_state_tensor[1:]) and torch.equal(targets[1:],env.pos_act[1:]) and torch.equal(master[1:],r.master_target[1:])
            env.train_test_flag='train';env.progress_buf.fill_(env.max_episode_length);env.step(zeros)
            reset_ok=bool((env.progress_buf==1).all())
            result(trial_name,{'other_envs_unchanged':bool(isolated) if a.num_envs>1 else None,'timeout_reset':reset_ok,'reset_targets':bool(torch.all(r.master_target==r.default_dof_pos[r.master_id]))},dict(num_envs=a.num_envs))
        report['status']='FAIL' if any(t['status']=='FAIL' for t in report['tests']) else ('UNVERIFIED' if any(t['status']=='UNVERIFIED' for t in report['tests']) else 'PASS')
    except Exception as exc:
        report['status']='ERROR';report['failed_trial']=trial_name;report['failed_stage']=stage;report['error']=str(exc);report['traceback']=traceback.format_exc();traceback.print_exc()
    finally:
        f.close()
        (out/'report.json').write_text(json.dumps(report,indent=2))
        if env is not None:
            if env.viewer:env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)
    print('REPORT',out/'report.json',report['status'],flush=True)
    return 0 if report['status']=='PASS' else 1
if __name__=='__main__':sys.exit(main())
