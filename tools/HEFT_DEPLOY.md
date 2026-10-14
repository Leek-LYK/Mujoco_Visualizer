# 大脑 Motion → HEFT Deploy Motion

在项目根目录运行（使用现有 `.venv`，无需新增依赖）：

```powershell
# 单文件，默认写入源目录的 deploy 子目录
.\.venv\Scripts\python.exe tools\convert_heft_deploy.py "data\roban\大脑生成\roban_key_out\001_a person crosses arms and stands still.csv"

# 批量转换同层 CSV；失败文件会报错，继续处理其他文件，最终返回非零退出码
.\.venv\Scripts\python.exe tools\convert_heft_deploy.py "data\roban\大脑生成\roban_key_out" --output-dir "data\roban\heft_deploy"

# 递归转换，保留相对目录结构
.\.venv\Scripts\python.exe tools\convert_heft_deploy.py "data\roban\大脑生成" --recursive --output-dir "data\roban\heft_deploy"

# 对照真实 HEFT 示例检查表头及分隔符
.\.venv\Scripts\python.exe tools\convert_heft_deploy.py "data\roban\大脑生成\roban_key_out\001_a person crosses arms and stands still.csv" --output "data\roban\heft_deploy\example_deploy.csv" --reference "tests\fixtures\heft_deploy_reference.csv"

# 不依赖外部数据集的转换测试
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_heft_deploy.py
```

输出文件名默认加 `_deploy.csv`。已有文件不会被覆盖；批处理排除输出目录以及
`*_deploy.csv` 文件。`--output` 仅用于单文件，`--output-dir` 可用于单文件或目录。

## 数据约定

- 已核对真实源文件：29 列，含 timestamp、7 个 root 字段和 21 个 `*_joint_dof`。
  复用 `load_motion_csv` / `MotionData` 进行语义解析、四元数检查和重复字段检测。
- 已核对真实 HEFT 示例：虽然扩展名为 CSV，实际是 **Tab 分隔**，无 timestamp，
  共 49 列。输出同样使用 Tab，不添加索引、时间或元数据列。
- 字段顺序：`body_pos_x/y/z`、`body_quat_w/x/y/z`、`joint_pos00` 至
  `joint_pos20`、`joint_vel_00` 至 `joint_vel_20`。
- 关节按名称映射：waist_yaw → leg_l1…l6 → leg_r1…r6 → zarm_l1…l4 → zarm_r1…r4。
  实际名称保留 `_joint` 后缀；与源文件的交错排列无关。缺失、重复、额外关节均报错。
- 输入必须有至少两帧、严格递增且均匀的 25Hz 时间戳（秒，间隔容差 1 微秒）。
  保留首末样本时刻，不外推：N 帧变为 2N−1 帧，每帧间隔 0.02 秒。
- 根位置和关节位置采用线性时间插值；四元数采用 SciPy 最短弧 SLERP，输出为
  归一化且符号连续的 wxyz。关节速度在重采样后以时间为自变量求差分，内部为中心差分，
  两端为单边差分，不强制端点速度为零。
- 不修改坐标系、根节点定义、关节符号或角度单位；速度单位为输入角度单位/秒。
- 写出前校验时间网格，再回读序列化候选文件，校验字段顺序、49 列、NaN/Inf 和
  四元数范数。由于目标文件没有时间列，独立文件本身无法证明帧率；部署端应按 50Hz 消费。

测试基准 `tests/fixtures/heft_deploy_reference.csv` 直接取自用户提供的 HEFT 示例
表头及首帧，测试覆盖字段、名称重排、插值、q/-q 符号切换、速度、错误输入和递归批处理。
