import mujoco,numpy as np,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];m=mujoco.MjModel.from_xml_path(str(R/'assets/ur5_inspire/ur5_inspire.xml'));d=mujoco.MjData(m)
arm=['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'];ids=[m.joint(n).qposadr[0] for n in arm];v=[m.joint(n).dofadr[0] for n in arm]
l=m.body('left_tip_ur5_inspire').id;r=m.body('right_tip_ur5_inspire').id;t=m.body('tcp_ur5_inspire').id
hand=[i for i in range(m.nq) if i not in ids]
def pose():
 mujoco.mj_forward(m,d)
 return (d.xpos[l]+d.xpos[r])/2,d.xmat[t].reshape(3,3)
def solve(pos,rot):
 rng=np.random.default_rng(7)
 for attempt in range(50):
  d.qpos[:]=0;d.qpos[hand]=.82
  d.qpos[ids]=np.array([0,-1.4,1.8,-1.9,-1.57,0]) if attempt==0 else rng.uniform(-2.5,2.5,6)
  for i in range(350):
   p,Rc=pose(); e=np.r_[pos-p, .5*sum(np.cross(Rc[:,k],rot[:,k]) for k in range(3))]
   if np.linalg.norm(pos-p)<1e-5 and np.linalg.norm(Rc-rot)<1e-4:
    # avoid below-ground arm poses
    if min(d.xpos[m.body(n).id,2] for n in ('shoulder_link','upper_arm_link','forearm_link','wrist_1_link','wrist_2_link','wrist_3_link'))>.03 and all(c.dist > -0.0001 for c in d.contact):
     return d.qpos[ids].copy(),p.copy(),i
   j1=np.zeros((3,m.nv));j2=j1.copy();jr=j1.copy()
   mujoco.mj_jacBody(m,d,j1,None,l);mujoco.mj_jacBody(m,d,j2,None,r);mujoco.mj_jacBody(m,d,None,jr,t)
   J=np.r_[(j1+j2)/2,jr][:,v]
   dq=J.T@np.linalg.solve(J@J.T+.0025*np.eye(6),e)
   d.qpos[ids]=np.clip(d.qpos[ids]+np.clip(dq,-.15,.15),-3.14,3.14)
 raise RuntimeError('IK failed')
if __name__ == '__main__':
 result={}
 for name,pos,rot in [('grasp_cube',[.45,0,.20],np.diag([1,-1,-1])),('open_drawer',[.42,0,.35],np.array([[0,0,1],[1,0,0],[0,1,0]]))]:
  q,p,it=solve(np.array(pos),rot);print(name,q,p,it,'width',np.linalg.norm(d.xpos[l]-d.xpos[r]))
  result[name]={'arm_positions':q.tolist(),'initial_tcp':p.tolist(),'target_rotation':rot.tolist()}
  # Check static holding and mimic dynamics.
  d.qvel[:]=0;d.ctrl[:6]=q;d.ctrl[6]=.82
  for k in range(2000):mujoco.mj_step(m,d)
  print('after hold',np.max(np.abs(d.qpos[ids]-q)),'hand',d.qpos[hand], 'finite',np.isfinite(d.qpos).all(),'contacts',d.ncon)
 (R/'assets/ur5_inspire/calibration_ur5_inspire.json').write_text(json.dumps(result,indent=2))
 # System Python has PyYAML; only update the two newly introduced configs.
 import subprocess
 for name,pose in result.items():
  config=R/f'cfg/tasks/{name}_ur5_inspire.yaml'
  code="import sys,json,yaml; p=sys.argv[1]; c=yaml.safe_load(open(p)); c['robot']['arm_positions']=json.loads(sys.argv[2]); open(p,'w').write(yaml.safe_dump(c,sort_keys=False))"
  subprocess.check_call(['python3','-c',code,str(config),json.dumps(pose['arm_positions'])])
 print('Updated calibration and *_ur5_inspire task reset poses. Re-run build_assets_ur5_inspire.py to refresh MJCF keyframes.')
