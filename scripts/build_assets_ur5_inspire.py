#!/usr/bin/env python3
"""Reproducibly build the fixed UR5 + EG2 assets. Run with numpy and mujoco.
Xacro expansion uses the system python3 (which has xacro); no ROS install needed.
"""
from pathlib import Path
import copy
import json
import math
import subprocess
import xml.etree.ElementTree as E
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/ur5_inspire'
MOUNT_XYZ = '0 0 0'
MOUNT_RPY = '0 0 0'
ARM = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

def write_xml(root, path):
    for node in root.iter():
        if node.text is not None and not node.text.strip(): node.text=None
        if node.tail is not None and not node.tail.strip(): node.tail=None
    from xml.dom import minidom
    path.write_text(minidom.parseString(E.tostring(root)).toprettyxml(indent='  '))

def fixed(root, name, parent, child, xyz, rpy='0 0 0'):
    E.SubElement(root, 'link', name=child)
    j=E.SubElement(root,'joint',name=name,type='fixed')
    E.SubElement(j,'parent',link=parent); E.SubElement(j,'child',link=child)
    E.SubElement(j,'origin',xyz=xyz,rpy=rpy)

def mirror_obj(source, target, scale):
    lines=[]
    for line in source.read_text().splitlines():
        p=line.split()
        if not p or p[0] in ('mtllib','usemtl'): continue
        if p[0] in ('v','vn'):
            v=np.asarray(p[1:4],float)*scale
            line=p[0]+' '+' '.join(f'{x:.9g}' for x in v)
        elif p[0]=='f' and np.prod(scale)<0:
            line='f '+' '.join(p[:0:-1])
        lines.append(line)
    target.write_text('\n'.join(lines)+'\n')

def build_urdf():
    OUT.mkdir(exist_ok=True)
    meshdir=OUT/'meshes_ur5_inspire'; meshdir.mkdir(exist_ok=True)
    robot=E.parse(ROOT/'assets/ur5/ur5_description/urdf/ur5_joint_limited_robot.urdf').getroot()
    robot.set('name','ur5_inspire')
    for node in list(robot):
        if node.tag not in ('link','joint') or node.get('name') in ('world','world_joint','ee_link','ee_fixed_joint','base','base_link-base_fixed_joint'):
            robot.remove(node)
    macro=ROOT/'assets/inspire_description/xacro/EG2-4C2.xacro'
    wrapper=macro.read_text().replace('$(find inspire_description)', str(ROOT/'assets/inspire_description')).replace('</robot>', '<xacro:Gripper name="inspire"/></robot>')
    code='import xacro,sys; d=xacro.parse(sys.stdin.read()); xacro.process_doc(d); print(d.toxml())'
    hand=E.fromstring(subprocess.check_output(['python3','-c',code], input=wrapper.encode()))
    for node in hand:
        robot.append(copy.deepcopy(node))
    mount=E.SubElement(robot,'joint',name='ur5_inspire_mount',type='fixed')
    E.SubElement(mount,'parent',link='tool0'); E.SubElement(mount,'child',link='inspire_gripper_base')
    E.SubElement(mount,'origin',xyz=MOUNT_XYZ,rpy=MOUNT_RPY)
    # Inner pad plane centres. At q=0 the pads meet; q=0.82 is open.
    fixed(robot,'left_tip_fixed_ur5_inspire','inspire_left_pad','left_tip_ur5_inspire','0.0177 0.006 0.015','0 0 1.5707963267948966')
    fixed(robot,'right_tip_fixed_ur5_inspire','inspire_right_pad','right_tip_ur5_inspire','-0.0177 0.006 0.015','0 0 1.5707963267948966')
    fixed(robot,'tcp_fixed_ur5_inspire','inspire_gripper_base','tcp_ur5_inspire','0 0 0.14','0 0 1.5707963267948966')
    for mesh in robot.findall('.//mesh'):
        source=mesh.get('filename')
        if source.startswith('package://ur5_description/'):
            source=ROOT/'assets/ur5/ur5_description'/source.split('package://ur5_description/')[1]
            # Same collision geometry is also portable visual geometry for both engines.
            if source.suffix=='.dae': source=Path(str(source).replace('/visual/','/collision/')).with_suffix('.stl')
        else:
            source=Path(source[len('file://'):])
        scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
        if source.suffix=='.obj':
            target=meshdir/(source.stem+('_mirror' if np.prod(scale)<0 else '')+'_ur5_inspire.obj')
            mirror_obj(source,target,scale); mesh.attrib.pop('scale',None)
            mesh.set('filename',str(target.relative_to(OUT)))
        else:
            import os
            mesh.set('filename',os.path.relpath(source,OUT))
    robot.insert(0,E.Comment(' Fixed UR5 + Inspire EG2-4C2. SI units. q=0 closed, q=0.82 open. Mount transform is configured in build_assets_ur5_inspire.py. '))
    write_xml(robot,OUT/'robot_ur5_inspire.urdf')
    return robot

def build_mjcf(robot):
    import mujoco
    mj=E.Element('mujoco',model='ur5_inspire')
    mj.append(E.Comment(' Independent MJCF for reading/viewing. Isaac Gym trains using robot_ur5_inspire.urdf. ctrl[0:6]: UR5 joint radians; ctrl[6]: EG2 opening radians (0 closed, 0.82 open). These are NOT the seven policy Cartesian actions. '))
    mj.append(E.Comment(' 阅读顺序：asset 网格 → worldbody 机械臂与夹爪 → equality 五个跟随关节 → actuator 七路驱动 → keyframe 两个任务初始姿态。长度单位米，角度单位弧度；底座固定。机器人内部碰撞关闭，与地面/外部物体碰撞保留。 '))
    E.SubElement(mj,'compiler',angle='radian',autolimits='true',inertiafromgeom='false')
    E.SubElement(mj,'option',timestep='0.001',integrator='implicitfast',gravity='0 0 -9.81',iterations='100')
    default=E.SubElement(mj,'default')
    E.SubElement(default,'joint',damping='0.05',armature='0.001')
    E.SubElement(default,'geom',friction='1 0.01 0.001',contype='1',conaffinity='2',rgba='0.65 0.68 0.72 1')
    assets=E.SubElement(mj,'asset'); world=E.SubElement(mj,'worldbody')
    E.SubElement(world,'light',pos='0 -1 2',dir='0 0 -1')
    E.SubElement(world,'geom',name='ground_ur5_inspire',type='plane',size='2 2 0.1',contype='2',conaffinity='1',rgba='0.2 0.25 0.3 1')
    E.SubElement(world,'camera',name='overview_ur5_inspire',pos='1.4 -1.4 1',xyaxes='0.707 0.707 0 -0.3 0.3 0.905')
    links={l.get('name'):l for l in robot.findall('link')}
    children={}
    for j in robot.findall('joint'): children.setdefault(j.find('parent').get('link'),[]).append(j)
    mesh_names={}
    def body(parent,name,joint=None):
        attrs={'name':name}
        if joint is not None:
            origin=joint.find('origin')
            if origin is not None: attrs.update(pos=origin.get('xyz','0 0 0'), euler=origin.get('rpy','0 0 0'))
        if name=='inspire_gripper_base':
            parent.append(E.Comment(' UR5 tool0 到 EG2 底座的固定安装变换；当前为直接对接，无虚构转接件。修改生成脚本的 MOUNT_XYZ/MOUNT_RPY 后重新生成。 '))
        if name=='tcp_ur5_inspire':
            parent.append(E.Comment(' TCP 提供统一姿态：局部 Y 为两指分离方向，Z 为接近方向。训练中的 TCP 位置使用左右夹持面中心的平均值，会随开合变化。 '))
        b=E.SubElement(parent,'body',**attrs); link=links[name]
        if joint is not None and joint.get('type')!='fixed':
            lim=joint.find('limit')
            E.SubElement(b,'joint',name=joint.get('name'),type='hinge',axis=joint.find('axis').get('xyz'),range=lim.get('lower')+' '+lim.get('upper'))
        inertial=link.find('inertial')
        if inertial is not None:
            origin=inertial.find('origin'); inertia=inertial.find('inertia')
            if origin is not None and origin.get('rpy','0 0 0')!='0 0 0': raise ValueError('Nonzero inertial RPY requires conversion')
            E.SubElement(b,'inertial',pos=origin.get('xyz','0 0 0') if origin is not None else '0 0 0',mass=inertial.find('mass').get('value'),fullinertia=' '.join(inertia.get(k) for k in ('ixx','iyy','izz','ixy','ixz','iyz')))
        for i,col in enumerate(link.findall('collision')):
            mesh=col.find('geometry/mesh'); attrs={'name':name+'_geom'+str(i)}
            if mesh is not None:
                filename=mesh.get('filename')
                if filename not in mesh_names:
                    mesh_names[filename]='mesh_'+str(len(mesh_names))
                    E.SubElement(assets,'mesh',name=mesh_names[filename],file=filename)
                attrs.update(type='mesh',mesh=mesh_names[filename])
            else: continue
            origin=col.find('origin')
            if origin is not None: attrs.update(pos=origin.get('xyz','0 0 0'),euler=origin.get('rpy','0 0 0'))
            E.SubElement(b,'geom',**attrs)
        for child in children.get(name,[]): body(b,child.find('child').get('link'),child)
    body(world,'base_link')
    eq=E.SubElement(mj,'equality')
    eq.append(E.Comment(' EG2 one motor, five followers. Preserve the source mimic coefficients. '))
    for j in robot.findall('joint'):
        mimic=j.find('mimic')
        if mimic is not None:
            E.SubElement(eq,'joint',joint1=j.get('name'),joint2=mimic.get('joint'),polycoef=f"{mimic.get('offset','0')} {mimic.get('multiplier','1')} 0 0 0",solref='0.002 1')
    act=E.SubElement(mj,'actuator')
    act.append(E.Comment(' 前六项为 UR5 关节位置目标；第七项为夹爪主动关节，0 闭合、0.82 张开。训练策略的前六维是笛卡尔增量，必须经过 IK，不能直接写入这里。 '))
    for name in ARM+['inspire_gripper_joint']:
        j=next(j for j in robot.findall('joint') if j.get('name')==name); lim=j.find('limit')
        arm=name in ARM
        E.SubElement(act,'position',name=name+'_drive',joint=name,kp='5000' if arm else '20',kv='150' if arm else '0.5',ctrlrange=lim.get('lower')+' '+lim.get('upper'),forcerange='-'+lim.get('effort')+' '+lim.get('effort'))
    write_xml(mj,OUT/'ur5_inspire.xml')
    model=mujoco.MjModel.from_xml_path(str(OUT/'ur5_inspire.xml'))
    calibration=OUT/'calibration_ur5_inspire.json'
    if calibration.is_file():
        keys=E.SubElement(mj,'keyframe')
        for name,pose in json.loads(calibration.read_text()).items():
            q=np.full(model.nq,0.82)
            for joint,value in zip(ARM,pose['arm_positions']): q[model.joint(joint).qposadr[0]]=value
            E.SubElement(keys,'key',name=name+'_ur5_inspire',qpos=' '.join(map(str,q)),ctrl=' '.join(map(str,pose['arm_positions']+[0.82])))
        write_xml(mj,OUT/'ur5_inspire.xml')
        model=mujoco.MjModel.from_xml_path(str(OUT/'ur5_inspire.xml'))
    return mj,model

def build_objects():
    folder=OUT/'objects_ur5_inspire'; folder.mkdir(exist_ok=True)
    cube=E.Element('robot',name='cube_ur5_inspire')
    def boxlink(robot,name,size,mass,xyz='0 0 0'):
        l=E.SubElement(robot,'link',name=name)
        it=E.SubElement(l,'inertial'); E.SubElement(it,'origin',xyz=xyz)
        E.SubElement(it,'mass',value=str(mass))
        x,y,z=map(float,size.split()); vals=[mass*(y*y+z*z)/12,mass*(x*x+z*z)/12,mass*(x*x+y*y)/12]
        E.SubElement(it,'inertia',ixx=str(vals[0]),iyy=str(vals[1]),izz=str(vals[2]),ixy='0',ixz='0',iyz='0')
        for tag in ('visual','collision'):
            v=E.SubElement(l,tag); E.SubElement(v,'origin',xyz=xyz)
            E.SubElement(E.SubElement(v,'geometry'),'box',size=size)
        return l
    boxlink(cube,'cube','0.05 0.05 0.05',0.125)
    write_xml(cube,folder/'cube_ur5_inspire.urdf')
    drawer=E.Element('robot',name='drawer_ur5_inspire')
    boxlink(drawer,'cabinet','0.20 0.30 0.02',2,'0.12 0 -0.08')
    boxlink(drawer,'drawer','0.02 0.24 0.12',0.4)
    j=E.SubElement(drawer,'joint',name='slide',type='prismatic')
    E.SubElement(j,'parent',link='cabinet'); E.SubElement(j,'child',link='drawer')
    E.SubElement(j,'axis',xyz='-1 0 0'); E.SubElement(j,'limit',lower='0',upper='0.16',effort='30',velocity='0.5')
    boxlink(drawer,'handle','0.025 0.12 0.025',0.03)
    j=E.SubElement(drawer,'joint',name='handle_fixed',type='fixed')
    E.SubElement(j,'parent',link='drawer'); E.SubElement(j,'child',link='handle'); E.SubElement(j,'origin',xyz='-0.04 0 0')
    write_xml(drawer,folder/'drawer_ur5_inspire.urdf')
    # Corner convention matches original rewards: 0-4 outward, 1-0 long, 3-0 short.
    corners=[[-.0525,-.06,-.0125],[-.0525,.06,-.0125],[-.0525,.06,.0125],[-.0525,-.06,.0125],[-.0275,-.06,-.0125],[-.0275,.06,-.0125],[-.0275,.06,.0125],[-.0275,-.06,.0125]]
    (folder/'bbox_info_ur5_inspire.json').write_text(json.dumps({'link_name':['cabinet','drawer','handle'],'bbox_world':[corners]*3,'axis_xyz_world':[[0,0,0]]*3,'axis_dir_world':[[-1,0,0]]*3},indent=2))

if __name__=='__main__':
    robot=build_urdf(); mj,model=build_mjcf(robot); build_objects()
    print('Built',OUT,'MJCF nq=',model.nq,'nu=',model.nu)
