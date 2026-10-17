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

Isaac Gym 每步显式设置六个夹爪关节的耦合位置目标；这是同步位置驱动，不是完整闭环连杆约束。MuJoCo 使用五个 joint equality 约束和一个夹爪执行器。Isaac Gym 默认启用选择性自碰撞：仅排除固定组件、相邻连杆和夹爪内部机械重叠对，两侧夹持面之间及机器人与物体的接触保留。机器人使用非零号过滤位，抽屉保留位 0（掩码 1），方块为 0。MuJoCo 查看资产仍采用原自碰撞设置，不能替代 PhysX 验收。

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


## PhysX 接触验证（新增，等待服务器实测）

此改动只修改 UR5 专用代码。控制器保存单一夹爪目标开度，零夹爪动作保持目标；
接触阻力不再把下一步目标拉回实测角度。局部重置只恢复对应环境的目标。
各关节 effort 从组合 URDF 显式设置，这些数值并非经过实物标定的力矩。
`robot.friction` 默认 1.0；原刚度、阻尼、速度及仿真接触参数未调大。
`robot.selective_self_collision` 和 `robot.persistent_gripper_target` 默认 true。
模型仍是同步位置驱动近似，不是完整闭环连杆。

新增日志：`gripper_coupling_error`、`joint_target_error`、`gripper_opening`。
已有模型权重结构没有变化，但接触行为已改变，正式训练使用新实验名、从头训练。

### 先运行基线与稳定性检查

在服务器拉取提交后，保持 `partmanip` 环境，在项目根目录执行：

```bash
python scripts/validate_contact_ur5_inspire.py --task open_drawer --device cpu --suite stability --baseline --output logs/contact_validation/drawer_baseline_cpu
python scripts/validate_contact_ur5_inspire.py --task open_drawer --device cpu --suite stability --output logs/contact_validation/drawer_updated_cpu
```

每次使用新的输出目录，避免覆盖旧证据。`--baseline` 恢复旧碰撞过滤及基于实测开度生成目标的行为，
用于同版本测试脚本对照；它仍显式设置 URDF effort，不是旧提交的逐字节回放。
不要将失败归因于单一参数；先查看 `failed_checks` 和轨迹，再一次调整一类参数并重跑。

### 完整验收命令

以下命令分别记录两个任务、CPU/GPU、旧/新行为；单次失败后仍保存 JSON/CSV。
这是验证矩阵，不是训练启动脚本。每个任务默认 10 次保持重置及 1 次固定位置 + 10 次随机操作。

```bash
for task in grasp_cube open_drawer; do
  for device in cpu cuda; do
    python scripts/validate_contact_ur5_inspire.py --task "$task" --device "$device" --suite all --baseline --output "logs/contact_validation/${task}_${device}_baseline"
    python scripts/validate_contact_ur5_inspire.py --task "$task" --device "$device" --suite all --output "logs/contact_validation/${task}_${device}_updated"
  done
  python scripts/validate_contact_ur5_inspire.py --task "$task" --device cuda --suite reset --num-envs 4 --output "logs/contact_validation/${task}_reset4"
done
```

脚本支持 `--viewer` 打开窗口、`--seed 1234`、`--trials 10`；不需要 MuJoCo、TensorBoard 或 PPO 检查点。
单环境测试不验证跨环境重置隔离，该项标记 UNVERIFIED；4 环境 reset 测试专门验证隔离。
当前只支持随仓库提供的方块与抽屉，不能把结果推广到原数据集所有物体。

- 原始操作测试直接调用训练控制器和 PhysX stepping，绕开成功/超时自动重置。
- 测试不会附着物体、运行中改写物体状态或驱动物体关节；只有测试之间正常重置。
- 双指接触取自实际 pad 刚体，而非无碰撞几何的 tip 标记。CPU 要求保持阶段至少 90% 帧双指同时接触目标。
- CPU 接触重叠阈值 2 mm；GPU 净接触力不能确认接触对或穿透，相关项目标记 UNVERIFIED，必须结合 CPU 结果。
- 静置 1 秒后保持 3 秒：关节漂移 <=0.02 rad，TCP 漂移 <=5 mm，物体漂移 <=1 cm；重置及保持期间物体位移 <=5 cm。
- 方块中心高度 >=0.18 m，保持 2 秒下降 <=1 cm、相对 TCP 滑移 <=1 cm。
- 抽屉拉出 >=0.08 m，保持 2 秒回退 <=5 mm，并有双指把手接触证据。
- 保持阶段 TCP 峰峰位移 <=5 mm，开度峰峰变化 <=2 mm，关节耦合误差 <=0.02 rad。

结果文件为 `report.json`、`trajectory.csv`，包含逐次失败/未验证指标、有效机器人配置、每步关节位置速度及目标、TCP、物体位置、接触力、抽屉位移和阶段。
PASS 返回 0，FAIL/UNVERIFIED/ERROR 返回 1；GPU 报告 UNVERIFIED 不等于程序崩溃，详细看失败项与缺失证据。
达到目标失败可能是脚本轨迹或控制器问题，不能直接当作物体不可抓；CSV 用于区分到达失败、滑移、耦合失真和振荡。

### 接触检查后运行 PPO 回归

确认结果后分别跑两个任务的新实验，50 轮含保存与第 50 轮评估：

```bash
for task in grasp_cube open_drawer; do
  python train_ur5_inspire.py --taskcfg "${task}_ur5_inspire" --algocfg ppo_ur5_inspire --algo.num_envs 4 --algo.max_iterations 50 --algo.save_frequence 25 --algo.eval_frequence 50 --exp_name "${task}_contact_v2" --log.mode screen
done
```

这些检查只能验证有限场景下的模型和代码行为，不能证明收敛或实物表现。
本次本地回归覆盖持久开度、局部目标重置、碰撞掩码、误判防护及 CPU 张量 PPO 保存恢复。
真实 PhysX 基线、接触和训练回归由服务器运行后再确认，不能把本地张量测试当作物理验收。
