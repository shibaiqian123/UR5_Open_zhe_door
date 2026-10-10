#!/usr/bin/env python3
"""View the combined MJCF, or render headlessly with MUJOCO_GL=osmesa."""
import argparse
from pathlib import Path
import time
import mujoco

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pose', choices=['grasp_cube','open_drawer'],default='grasp_cube')
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    model=mujoco.MjModel.from_xml_path(str(ROOT/'assets/ur5_inspire/ur5_inspire.xml'))
    data=mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,model.key(args.pose+'_ur5_inspire').id)
    mujoco.mj_forward(model,data)
    if args.render:
        from PIL import Image
        with mujoco.Renderer(model,height=480,width=640) as renderer:
            renderer.update_scene(data,camera='overview_ur5_inspire')
            path=ROOT/f'assets/ur5_inspire/{args.pose}_preview_ur5_inspire.png'
            Image.fromarray(renderer.render()).save(path)
            print(path)
        return
    from mujoco import viewer as mj_viewer
    with mj_viewer.launch_passive(model,data) as viewer:
        while viewer.is_running():
            start=time.monotonic()
            mujoco.mj_step(model,data)
            viewer.sync()
            time.sleep(max(0,model.opt.timestep-(time.monotonic()-start)))

if __name__=='__main__': main()
