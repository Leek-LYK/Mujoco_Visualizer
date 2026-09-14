# project_mujoco_visualizer

支持 `.pkl` 动作字典（GMR/LAFAN）：`fps`、`root_pos (T,3)`、
`root_rot (T,4)`、`dof_pos (T,J)`、`joint_names (J,)`。
`root_rot` 按 `xyzw` 读取并转为 MuJoCo 的 `wxyz`，关节按名称映射。
可选的 `link_body_list` 用作 `local_body_pos` 的身体名称列表。
PKL 与 CSV/NPZ 一样支持上下键切换；仅加载可信来源的 PKL，
因为 Python pickle 反序列化可执行代码。

```powershell
uv run python -m project_mujoco_visualizer --model robot_asset/roban_s22_handball/xml/scene.xml --motion ".\data\dance1_subject1.pkl"
```

一个使用官方 `mujoco` Python API 的 CSV/NPZ 机器人运动可视化工具。它不依赖大型 GUI 框架，使用 MuJoCo 自带的 passive viewer 显示模型。

## 运行

在项目根目录执行：

```powershell
uv run python -m project_mujoco_visualizer `
  --model robot_asset/roban_s22_handball/xml/scene.xml `
  --motion "data/666_Dance.csv"
```

也可以省略 `--model` 和 `--motion`。程序会在默认的
`robot_asset/roban_s22_handball` 下自动检查 XML/MJCF，优先选择
`scene.xml` 这类包含机器人模型的场景入口，并从 `data` 中选择第一个 CSV：

```powershell
uv run python -m project_mujoco_visualizer
```

不打开窗口但完整执行 CSV 分析、模型映射和所有帧的 qpos 构造：

```powershell
uv run python -m project_mujoco_visualizer --analyze-only
```

### LAFAN/Roban NPZ

The retargeted LAFAN batch uses the GMR-style NPZ format:

```text
fps: scalar or one-element array (50 Hz)
root_pos: (T, 3)
root_rot: (T, 4), input order xyzw
dof_pos: (T, 21)
joint_names: (21,)
body_names: (28,)
```

Play one NPZ file directly:

```powershell
uv run python -m project_mujoco_visualizer `
  --model robot_asset/roban_s22_handball/xml/scene.xml `
  --motion "data/LAFAN/lafan_s22_npz/walk1_subject1.npz"
```

The loader converts NPZ `root_rot` from `xyzw` to MuJoCo `wxyz`, generates timestamps from `fps`, and maps `dof_pos` by `joint_names`. The `local_body_pos` and `local_body_rot` arrays are validated when present but are not needed to construct MuJoCo qpos.

Whole-body-tracking NPZ files are also accepted when they contain `joint_pos`,
`body_pos_w`, and `body_quat_w`. The root is read from body index `0`, and
`body_quat_w` is interpreted as `wxyz`. Roban S22 OminiSoma files without a
`joint_names` field use the canonical 21-joint order for this asset.

如果 CSV 没有时间列，必须显式提供采样率，例如
`--sample-rate 25`。编号四元数列（`root_quat_0..3`）必须显式指定
`--quat-order wxyz` 或 `--quat-order xyzw`。

## 当前数据格式分析

仓库中的 6 个 CSV 实际上都是 100 帧、29 列、25 Hz（时间间隔 0.04 s）：

```text
timestamp
root_translateX, root_translateY, root_translateZ
root_quat_w, root_quat_x, root_quat_y, root_quat_z
waist_yaw_joint_dof
zarm_l1_joint_dof, zarm_r1_joint_dof
leg_l1_joint_dof,  leg_r1_joint_dof
zarm_l2_joint_dof, zarm_r2_joint_dof
leg_l2_joint_dof,  leg_r2_joint_dof
zarm_l3_joint_dof, zarm_r3_joint_dof
leg_l3_joint_dof,  leg_r3_joint_dof
zarm_l4_joint_dof, zarm_r4_joint_dof
leg_l4_joint_dof,  leg_r4_joint_dof
leg_l5_joint_dof,  leg_r5_joint_dof
leg_l6_joint_dof,  leg_r6_joint_dof
```

`timestamp` 按秒解释，root 平移按米解释，root 四元数为单位四元数，关节角按弧度解释。程序运行时仍会从 CSV 表头和内容重新识别这些字段，不依赖上面的列顺序。

## 四元数和 qpos 规则

- MuJoCo free joint 的 qpos 是 `[x, y, z, qw, qx, qy, qz]`，内部四元数顺序为 `wxyz`。
- `root_quat_w/x/y/z` 按组件名识别，并组装为 MuJoCo 的 `wxyz`；不会按 CSV 的物理列位置猜测，也不会静默归一化原始值。
- 如果输入是 `root_quat_0..3`，`--quat-order xyzw` 会显式转换为 `[w, x, y, z]`；`wxyz` 则直接使用。
- root free joint 的位置和四元数通过 MuJoCo 的 `jnt_qposadr` 定位；每个 hinge/slide 关节通过 joint name 找到对应 CSV 字段和 qpos 地址。
- 不允许用 CSV 列序或固定 `qpos[7 + i]` 作为关节映射依据。

启动前会输出 CSV 字段、帧数、采样率、root 字段、关节数量和每一条
`CSV column -> qpos[index]` 映射。缺失关节、多余关节、重复语义字段、非数值、时间戳不递增、四元数非法和不支持的多自由度关节都会停止并给出具体名称。

## 播放控制

在 MuJoCo 窗口获得焦点后：

| 按键 | 操作 |
|---|---|
| `Space` | 播放 / 暂停 |
| `Up` / `Down` | 上一个 / 下一个动作文件 |
| `Left` / `Right` | 后退 / 前进一帧，并暂停 |
| `+` / `-` | 播放速度乘以 2 / 除以 2，范围 1/16x 到 16x |
| `L` | 循环播放开关 |
| `R` | 回到第 1 帧 |
| `F10` | 安全退出 viewer |

终端状态行显示当前帧、总帧数、CSV 时间、播放状态、倍速和循环状态。
播放时间使用 CSV 时间戳的实际帧间隔；若间隔不均匀，会按每一帧的间隔推进。
启动时默认开启循环播放；使用 `L` 可切换，命令行可用 `--no-loop` 显式关闭默认循环。

viewer 启动时会将相机设置为 tracking 模式并锁定 `base_link`，使用适度距离和俯视角；终端会打印实际使用的相机目标位置。

MuJoCo 原生 `Ctrl+Q` 也可以退出。终端中按 `Ctrl+C` 时程序会捕获中断并正常关闭 viewer，不再打印 traceback。

## 开发和测试

```powershell
uv run python -m unittest discover -s tests -v
```

测试覆盖 CSV 读取与采样率识别、重复/缺失字段错误、基于 joint name 的映射和自由根 qpos 构造。原始 `data/*.csv` 与 `robot_asset` 文件不会被程序改写。

如果本机的 uv 缓存目录权限异常，可临时使用 `uv run --no-cache ...`；这只绕过 uv 缓存，不改变项目环境或输入文件。

上下方向键在当前 `--motion` 文件所在目录内切换动作（省略 `--motion` 时使用自动选中文件的目录）。启动时扫描同层 CSV/NPZ 文件，按文件名排序（忽略大小写），首尾循环，不递归子目录。切换从第一帧开始，保留播放/暂停、倍速和循环设置；终端显示当前文件路径。加载失败会显示错误并保留当前动作。
