# UR5 + Inspire EG2-4C2 独立适配

本版本使用**固定底座 UR5 + EG2-4C2 两指夹爪**，不是 RH56E2/RH56F2 五指手。实现抓方块和开抽屉的状态观测 PPO；原源码、配置、机器人资产不修改，需要变更的模块都另存为同目录下的 `*_ur5_inspire` 文件。

## 先看这些文件

- `assets/ur5_inspire/ur5_inspire.xml`：可由 MuJoCo 加载的 MJCF，包含中文阅读注释、地面、驱动、夹爪耦合和两个初始姿态。
- `assets/ur5_inspire/robot_ur5_inspire.urdf`：Isaac Gym 实际加载的完整机器人，已展开 Xacro，使用相对网格路径。
- `train_ur5_inspire.py`：独立训练入口，默认抓方块。
- `tasks/load_robot_ur5_inspire.py`：关节名称映射、IK、夹爪控制与末端状态。
- `cfg/tasks/grasp_cube_ur5_inspire.yaml`、`cfg/tasks/open_drawer_ur5_inspire.yaml`：初始姿态、物体位置、速度及控制参数。
- `assets/ur5_inspire/validation_ur5_inspire.json`：实际验证结果。

训练调用链为：新训练入口 → 新配置解析 → 新任务/基类 → 新机器人适配器 → 新 PPO 模块。为避免原包 `__init__.py` 提前加载视觉、BC、DAgger 依赖，新入口通过同目录模块导入；直接运行该入口即可。未修改原包注册文件。新 PPO 复用原来的 `storage.py` 和 `RMS.py`，网络和日志模块另建副本，状态训练不需要 torchvision、TSDF、wandb。

## 查看 XML 和装配

当前机器已有可用的 MuJoCo 3.2.3，使用：

```bash
cd /home/robot/PartManip
/home/robot/miniconda3/envs/unitree/bin/python scripts/view_ur5_inspire.py
/home/robot/miniconda3/envs/unitree/bin/python scripts/view_ur5_inspire.py --pose open_drawer
```

有桌面显示时会打开交互窗口；没有桌面时可使用已验证的软件渲染：

```bash
MUJOCO_GL=osmesa /home/robot/miniconda3/envs/unitree/bin/python scripts/view_ur5_inspire.py --render
```

输出 `assets/ur5_inspire/grasp_cube_preview_ur5_inspire.png`。另附 `preview_ur5_inspire.png` 和 `open_drawer_preview_ur5_inspire.png`。XML 的训练任务名 keyframe 保存初始关节位置与对应控制目标；查看脚本自动应用该姿态。

装配关系是 `UR5 wrist_3_link → tool0 → inspire_gripper_base`。默认安装平移和旋转均为零，表示软件直接对接，不代表已经测量实物法兰或转接板。真实安装偏移集中在 `scripts/build_assets_ur5_inspire.py` 的 `MOUNT_XYZ`、`MOUNT_RPY`，单位为米和弧度。

UR5 的外观使用已有碰撞 STL，因此不保留原 DAE 的材质外观；尺寸、关节坐标与惯性来源不变。EG2 的右侧镜像已烘焙进新 OBJ，同时翻转面绕序，避免运行时负缩放。原网格保持不变。MuJoCo 和 Isaac Gym 的碰撞求解并不等价，不能用一个引擎的测试替代另一个。

## 动作、关节和观测

策略动作是形状 `(环境数, 7)` 的张量，数值范围 `[-1, 1]`：

1. `0:3`：末端沿仿真坐标轴的平移增量，默认每步最大 5 mm。
2. `3:6`：末端旋转误差向量，默认每步最大 0.02 rad，经阻尼最小二乘 IK 转为 UR5 六轴增量。
3. `6`：夹爪开合速度指令，**正值张开、负值闭合**，默认速度上限 1 rad/s。

UR5 关节顺序为 `shoulder_pan_joint`、`shoulder_lift_joint`、`elbow_joint`、`wrist_1_joint`、`wrist_2_joint`、`wrist_3_joint`。初始姿态数组使用这个顺序；仿真张量索引按名称查询，不依赖加载顺序。机械臂增量同时受源 URDF 速度限位、配置速度上限和位置限位约束。

EG2 主动关节为 `inspire_gripper_joint`，其余五个关节跟随它。角度 0 为闭合，0.82 为张开，最大夹持面间距约 **66.2 mm**。因此模型具有 **12 个运动关节、7 路独立控制量**。

Isaac Gym 每步显式设置六个夹爪关节的耦合位置目标；这是同步位置驱动，不是完整闭环连杆约束。MuJoCo 使用五个 joint equality 约束和一个夹爪执行器。机器人内部自碰撞关闭；Isaac Gym 机器人过滤位为 2，抽屉为 1，方块为 0，以保留机器人与任务物体的接触。

左右夹持面参考点挂在 `inspire_left_pad`、`inspire_right_pad` 上；训练 TCP 的位置和线速度取两参考点的平均值，姿态和角速度使用固定夹爪坐标系，不平均四元数。TCP 的局部 Y 轴为两指分离方向，Z 轴为接近方向。平移 Jacobian 取两参考点的平均，旋转 Jacobian 取 TCP，并且只选择机械臂六轴列。

抓方块状态为 43 维：末端姿态 7、方块位置及旋转矩阵 12、关节位置/速度 24。抽屉状态为 53 维：末端状态 13、把手位置/方向/尺寸 15、关节位置/速度 24、抽屉位置 1。启动时按实际加载的自由度重新计算维数，运行时检查形状和有限值。

**MJCF 的 `ctrl` 不是策略动作。** `ctrl[0:6]` 是 UR5 六轴位置目标，`ctrl[6]` 是夹爪开度角；笛卡尔策略动作需经过控制器后才能成为关节目标。

## 训练命令

训练需要原项目所用的 **Isaac Gym**（不是 Isaac Lab）、兼容的 PyTorch、NumPy、PyYAML。当前机器未找到 Isaac Gym，`nvidia-smi` 也无法连接驱动，所以没有声称 PhysX 训练已通过；本次未改系统驱动或安装 Isaac Gym。

在配置好 Isaac Gym 的 Python 环境中，先分别执行短程检查：

```bash
cd /home/robot/PartManip
python train_ur5_inspire.py --taskcfg grasp_cube_ur5_inspire --algocfg ppo_ur5_inspire --algo.num_envs 4 --algo.max_iterations 2 --algo.save_frequence 1 --exp_name cube_smoke
python train_ur5_inspire.py --taskcfg open_drawer_ur5_inspire --algocfg ppo_ur5_inspire --algo.num_envs 4 --algo.max_iterations 2 --algo.save_frequence 1 --exp_name drawer_smoke
```

确认单环境、批量重置、接触稳定后，再增加训练规模：

```bash
python train_ur5_inspire.py --taskcfg grasp_cube_ur5_inspire --algocfg ppo_ur5_inspire --exp_name cube_ur5 --device_id 0
python train_ur5_inspire.py --taskcfg open_drawer_ur5_inspire --algocfg ppo_ur5_inspire --exp_name drawer_ur5 --device_id 0
```

默认 64 个环境、仅终端日志、不录像；训练迭代数继承原 PPO 配置。`--headless` 沿用原项目布尔参数的“翻转默认值”规则：默认 True，传入该参数会变成 False 并打开 Isaac Gym 查看窗口。可用 `--algo.num_envs 1 --headless` 检查单环境。

检查点保存在 `logs/ckpts/<任务名>_ppo/<实验名>_seed1234/`。恢复或评估时传入明确的检查点文件路径：

```bash
python train_ur5_inspire.py --taskcfg grasp_cube_ur5_inspire --resume logs/ckpts/grasp_cube_ur5_inspire_ppo/cube_smoke_seed1234/model_2.pth --algo.num_envs 4 --algo.max_iterations 3 --exp_name cube_resume
python train_ur5_inspire.py --taskcfg grasp_cube_ur5_inspire --resume logs/ckpts/grasp_cube_ur5_inspire_ppo/cube_smoke_seed1234/model_2.pth --algo.num_envs 4 --test_only
```

保存内容包括机器人及任务签名。Franka、不同任务或缺少签名的检查点会被拒绝。BC、DAgger、视觉观测、预训练权重、录像和场景姿态导出不属于本次入口支持范围，选择时会明确报错。

## 测试物体与原数据集

新增 `assets/ur5_inspire/objects_ur5_inspire/`，包含 5 cm 方块、简易抽屉和把手标注。抽屉沿局部 -X 方向移动，行程 0.16 m，拉开一半且满足抓握几何条件为成功。抽屉的关节进度按 `upper - lower` 归一化。

默认 UR5 底座位于原点；方块中心为 `(0.45, 0, 0.025)`，目标高度为 0.2 m；抽屉原点为 `(0.54, 0, 0.35)`，初始把手中心为 `(0.50, 0, 0.35)`。随机平移范围 ±0.015 m；抽屉额外随机偏航 ±0.05 rad。两个初始姿态分别停在方块上方、把手前方。

原抽屉数据接入方式：在新抽屉配置中设 `use_test_object: false`，将 `drawer_dataset` 指向相对 `assetRoot` 的数据根目录，设置 `splits` 和 `object_scale`。也可通过命令行 `--task.asset.use_test_object` 翻转为 False。保留原来的目录命名、`bbox_info.json`、`mobility_new.urdf` 格式；不存在的路径会报错。不同尺寸物体需要重新设置摆放和检查固定 UR5 的可达性，当前验证仅覆盖新增测试物体。

## 重建和验证

生成依赖：运行解释器需要 NumPy、MuJoCo；系统 `python3` 需要 Xacro、PyYAML。本机已有对应组合环境。资产已经生成，正常训练不需要执行生成脚本或安装 ROS。

修改安装变换后可依次执行：

```bash
/home/robot/miniconda3/envs/unitree/bin/python scripts/build_assets_ur5_inspire.py
/home/robot/miniconda3/envs/unitree/bin/python scripts/calibrate_ur5_inspire.py
/home/robot/miniconda3/envs/unitree/bin/python scripts/build_assets_ur5_inspire.py
/home/robot/miniconda3/envs/unitree/bin/python scripts/validate_ur5_inspire.py
```

校准脚本更新新配置中的机械臂初始关节位置；第二次生成将新姿态写入 XML keyframe。生成器仅写新增资产目录。

本次实际通过的验证：

- 20 组随机关节姿态下，URDF FK 与 MJCF 位姿最大误差约 `8.9e-16`。
- 两个初始姿态没有地面穿透；重力下保持 2 秒，机械臂最大关节偏差约 `0.0067 rad`。
- 两个姿态的闭合/张开测试通过，最大耦合误差约 `2.9e-6 rad`。
- 24 个随机任务端点有可达且不穿透地面的 IK 解；这不是整条轨迹、自碰撞或物体接触验证。
- 已实际渲染并检查机器人装配。
- CPU 控制测试覆盖乱序 DOF 映射、IK 方向、限位、夹爪同步及非法动作；任务测试覆盖 43/53 维观测、有限奖励和局部/全量重置隔离。
- 两种观测维数分别完成两轮真实 PPO 更新及检查点保存/恢复，使用的是明确标识的**张量测试环境**，不是 Isaac Gym 物理环境。

仍待可用 Isaac Gym 环境验证：资产导入、PhysX 接触与 PD 参数、单/多环境物理运行、真实任务 PPO 更新及训练收敛。新测试物体也不等于原 PartManip 数据集上的性能结果。
