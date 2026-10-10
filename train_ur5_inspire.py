#!/usr/bin/env python3
"""State PPO entrypoint; original packages and source files remain untouched."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
# Import sibling modules directly, avoiding original __init__ imports of vision/BC.
for folder in ('utils','tasks','algorithms','algorithms/algo_utils'):
    sys.path.insert(0,str(ROOT/folder))

def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print('UR5 + EG2 PPO: python train_ur5_inspire.py --taskcfg grasp_cube_ur5_inspire|open_drawer_ur5_inspire --algocfg ppo_ur5_inspire [--algo.num_envs 4 --algo.max_iterations 2 --device_id 0 --headless]')
        print('Configuration overrides follow the original --section.key value syntax. --headless toggles the default True to False (viewer).')
        return
    # Isaac Gym must be imported before torch.
    try:
        import isaacgym
    except ImportError as exc:
        raise SystemExit('Isaac Gym is missing. Activate an Isaac Gym-compatible environment. This entrypoint does not use Isaac Lab. See README_ur5_inspire.md.') from exc
    import os,random
    import numpy as np
    import torch
    from config_ur5_inspire import process_cfgs
    from logger_ur5_inspire import Logger
    from grasp_cube_ur5_inspire import grasp_cube_ur5_inspire
    from open_drawer_ur5_inspire import open_drawer_ur5_inspire
    from ppo_ur5_inspire import ppo
    os.chdir(ROOT)
    cfg,sim_params=process_cfgs()
    task=cfg['task'];algo=cfg['algo']
    registry={'grasp_cube_ur5_inspire':grasp_cube_ur5_inspire,'open_drawer_ur5_inspire':open_drawer_ur5_inspire}
    if cfg['task_name'] not in registry or task['robot']['name']!='ur5_inspire':
        raise ValueError('Select a *_ur5_inspire task configuration')
    if cfg['algo_name']!='ppo' or algo['obs_mode']!='normal_state' or algo['add_proprio_obs'] or algo['model']['network']['name']!='MLP':
        raise ValueError('This entrypoint supports state-only PPO/MLP, not BC/DAgger/vision')
    if cfg['pretrain'] or cfg['save_pose'] or cfg['save_video']:
        raise ValueError('pretrain/save_pose/save_video unsupported in state-only adapter')
    if algo['num_envs']<1 or algo['n_minibatches']>algo['num_envs']*algo['n_steps']:
        raise ValueError('Invalid environment/minibatch counts')
    seed=cfg['seed'] if cfg['seed']>=0 else 1234
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    if cfg['torch_deterministic']:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
    cfg['exp_name']+=f'_seed{seed}'
    logger=Logger(cfg,cfg['exp_name'],cfg['task_name'],cfg['algo_name'])
    if cfg['resume']:
        checkpoint=Path(cfg['resume']).expanduser()
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        algo['resume']=str(checkpoint.resolve())
    env=registry[cfg['task_name']](task,sim_params)
    try:
        runner=ppo(env,algo,logger)
        runner.run()
    finally:
        if env.viewer:env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
