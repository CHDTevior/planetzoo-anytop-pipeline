# AnyTop-13 IR Harness —— 添加一个新数据源

一个 agent 可运行的脚手架,把**任意**新动作源(一套游戏绑定、一个 mocap 集、另一个动物库)转成共享的
**AnyTop-13 中间表示**,以便它加入同一个 Graph-VQVAE → Graph-CodeFlow 码本。**先读**
[`docs/ANYTOP13_INTERMEDIATE_REPRESENTATION.md`](../../docs/ANYTOP13_INTERMEDIATE_REPRESENTATION.md) ——
本 harness 只是机制;那份文档才是契约。

> **谁来运行。** 这是为一个 **agent** 端到端执行而写的:为新源实现一个 `SourceAdapter` 子类,运行 driver,harness
> 就产出 IR 并跑验收 gates。模型、图字段、归一化、FK 约定**都不**归你动 —— harness 会派生/强制它们。

---

## 你产出什么(唯一的输出)

```
<out_root>/
  motions/<motion_id>.npy        # float32 [T, J, 13]   (RAW,ORIGINAL 关节顺序)
  cond.npy                       # dict{object_type -> cond record}   (ORIGINAL 关节顺序)
  motion_texts_by_file.json      # {motion_id -> {primary_caption, captions:[...]}}
  motion_object_types.json       # {motion_id -> object_type}   (显式映射;不做前缀猜测)
  splits/{train,val}.txt         # 每行一个 motion_id
  _gate_report.json              # Gates A/B/C/E + overall_hard_pass(由 harness 写)
  _ACCEPTED                      # 仅当所有硬 gate 通过时才写 —— 消费方必须要求它
```

`build_source.build(...)` **在任一红硬 gate(A/B/C/E)时抛 `AcceptanceError`**(除非你传 `strict=False`);它会先写
`_gate_report.json` 以便失败可检查。它**仅**在 `overall_hard_pass` 为真时才写 `_ACCEPTED` 标记(并在开始时清除任何
陈旧标记)—— **下游消费方必须拒绝没有 `_ACCEPTED` 文件的数据集**(外加做手工 Gate D)。它还会拒绝一个已经持有任何
artifact 的 `<out_root>`,除非 `overwrite=True`(这样陈旧文件不会渗进新的一次构建)。

每个 `object_type` 的 `cond` 记录:`parents [J] int`(单根 `-1`、连通)、`offsets [J,3]`、
`joints_names [J] str`、`tpos_first_frame [J,13]`、`mean [J,13]`、`std [J,13]`(外加 gates 使用的 `foot_joint_idx`、
`gate_c_mode`)。其余一切(adjacency / geodesic / graph_dist / joint_relations / skeleton_features / name_hashes /
`new_to_old_perm`)由**训练 loader 派生** —— **不要**装运它。见 spec §6。

你写出的一切,其关节顺序都是源的 **ORIGINAL** 顺序(loader 自己施加 FK 重排)。见 spec §3。

---

## Agent 工作流

1. **理解源。** 绑定拓扑(关节、parents、rest offsets)、坐标系 / up-axis、单位尺度、fps、旋转如何存储、哪些关节是
   脚、captions 从哪来。把 `SourceAdapter` docstring 里的问题填上。

2. **实现一个 `SourceAdapter` 子类**(`source_adapter.py`)。这里承载**逐源**工作,包括源→13ch **编码** ——
   源→13ch 这一步*正是*因源家族(BVH 绑定 vs. 位姿型 mocap)而根本不同的部分,所以 adapter 拥有它(spec §7)。你恰好
   实现四个方法:
   - `iter_object_types()` → 本源贡献的每一个不同骨骼拓扑。
   - `topology(object_type)` → 一个 `Topology`(`parents`、`offsets`、`joints_names`、`tpos_first_frame`),在源的
     **original** 关节顺序。
   - `iter_clips(object_type)` → `.motion` 已经是 **`[T,J,13]`** 的 `Clip` —— 你施加 Y-up/+Z 变换、
     `HML_AVG_BONELEN` 缩放、20fps 重采样 + 丢最后一帧(`T=F−1`)、经 Kabsch 重编码的 **per-parent rot6d** 使
     FK==RIC、根 facing-6D + XZ 速度进 `ch9/ch11`、contact 进 `ch12`、twist 已固定。设置 `Clip.source_frames`(供
     Gate-A 的 `T=F−1` 检查),以及当源有官方世界恢复时设 `Clip.source_world [T,J,3]`(供独立 Gate B)。两个
     **模板桩 (template stubs)** —— `BvhPipelineAdapter`(委托给现有 BVH 导出器 + `motion_process.process_object`,
     后者已经产出 `[T,J,13]`)和 `PoseSourceAdapter`(实现 spec §2/§7 的 Kabsch 重编码)—— 展示这两个家族;它们在你
     填好之前会抛 `NotImplementedError`。
   - `topology()` 也声明逐拓扑的 gate 策略:`foot_joint_idx`(contact 在这些之外必须为 0)和 `gate_c_mode`
     (`"clean"` 或 `"inferred_twist"`)。

3. **运行 driver**(`build_source.build(adapter, out_root, max_joints=…)`)。它**只**做源无关的后端 —— 从不做任何
   编码:它收集你的 `Clip`、在 **train** split 上计算 per-`object_type` 的 `mean/std`、写 `motions/` + `cond.npy` +
   `motion_texts_by_file.json` + `motion_object_types.json` + `splits/`、跑 gates、并**在红硬 gate 时抛异常**。
   `mean/std` 精确镜像 `motion_process.get_mean_std`:根块塌缩为各自标量、**所有非根关节共享每个 pos/rot/vel 块一个
   标量**、contact 非零→其均值 / 零→1.0、存储 std 中不烘焙 `1e-6`(loader 在归一化时加 `+1e-6`)。任何必须与既有
   数据匹配的约定,都已由**你的 adapter**施加;driver 只做打包 + 归一化 + 验证。

4. **Gates A/B/C/E 自动运行**、写 `_gate_report.json`,红硬 gate **抛 `AcceptanceError`** —— 数据集不被接受。
   Gate D(视觉)**不**自动化:用 repo 的渲染工具渲出 GT-vs-转换的 GIF 并亲眼看(视觉 QA 是权威;腿翻转能通过每一个
   标量 gate)。报告携带一条 `gate_D` 提醒,而非渲染结果 —— 训练前你仍须亲手做 Gate D。

5. **合并(可选)。** 因为归一化是每骨骼的,并集就是拼接:`cond.npy` 记录合并、`motions/` 汇集、split 拼接。无需重新
   归一化。要把该源加进 **CodeFlow backbone**,你随后必须重跑 VQVAE token 导出(spec §6 backbone 注)—— harness
   止步于"VQVAE-ready"的数据集。

---

## 验收 gates(spec §8)

| gate | 范围 | 检查 | 通过 |
|---|---|---|---|
| **A** 逐 clip 完整性 | **每个 clip** | `[T,J,13]`、有限、`T=F−1`(设了 `source_frames` 时为硬)& `T>0`、绝对值合理、contact 二值 + 在 `foot_joint_idx` 外为 0、rot6d 非退化、**multi-child rot6d 复制**一致 | 硬 |
| **B** RIC 等价 | 每个带 `source_world` 的 clip | 源自己的世界恢复(`Clip.source_world`)`==` `recover_from_bvh_ric_np(ch0:3)`;若无源恢复则**豁免**(标注) | `mean_l2≤1e-4` |
| **C** FK 自洽 | **每个 clip** | 内部 BFS 重排,然后 `recover_from_bvh_rot_np`(FK)对 RIC 一致 | 干净 `L2<1e-4` / `inferred_twist` ≤0.5% bbox |
| **D** 视觉 QA | 手工 | GT-vs-转换的 GIF(骨架 + 可蒙皮时的网格),侧视角 —— **手工**渲染,不由 `build_source.py` 渲 | **人/agent 肉眼 —— 权威** |
| **E** 布局契约 | 数据集 | `cond.npy` 可加载且非空;每个 `object_type` 都有必需键 + 有限 `mean/std`/`offsets`/`tpos` + 正确 shape + 单根在 0 的 `parents` + 粗略骨长尺度;每个 motion 映射到一个拓扑;每个 motion 在**恰好一个** split(不相交 + 全覆盖) | 硬 |

`build_source` 在**每个** clip 上跑 Gate A **和** C(逐 clip 覆盖;报告在 `gate_C_sample_per_type` 里为每类型保留
一个代表),并在每个携带 `source_world` 的 clip 上跑 Gate B。独立的 `gates.py` CLI 只从落盘重查 A/C/E —— 它无法重跑
Gate B(盘上没有 `source_world`),所以它读 build 的 `_ACCEPTED` 标记;没有它时,除非你传 `--allow_unmarked`(用于
遗留数据集),否则拒绝报告通过。完整的训练 loader ingest(实例化 `AnyTopDataset`、派生图字段)发生在**训练 repo** ——
Gate E 只检查该 loader 依赖的落盘契约。

`gates.py` 把 A/B/C/E 实现为函数,可对既有数据集作为回归检查独立运行:

```bash
python -m tools.ir_harness.gates --data_root <out_root> --object_type <OT> --large_rotation
```

---

## 不可让步项(为什么存在 —— 见 spec)

- **per-parent rot6d**(spec §2):子槽 `j` 存 `parent[j]` 的旋转。你的 adapter 必须产出这套布局;绝不存每关节自身
  旋转。Gate C 会抓到错误约定。
- **关节顺序**(spec §3):写 ORIGINAL 顺序;绝不预先施加 FK 置换。
- **不用 `apply_root_cancel`**(spec §5):double-root-rotation 修正已移除;保持关闭。
- **经 Kabsch 重编码旋转**到 AnyTop 的 rest 基(spec §7):直接 gather 源的原生旋转是错的(高达约 89° 骨基误差)。
  你的 adapter 做这件事(`PoseSourceAdapter` 桩指向 spec §2/§7 的算法;它是模板,不是实现);driver 只经 Gate C
  验证结果。
- **重算 `mean/std`**,在每个 object_type 的 train split 上(spec §4/§8):driver 做这件事 —— 绝不在 `cond` 里装运
  源自己的 flat 统计量。
- **固定 twist DOF**,在从位置推断旋转时于编码处固定(spec §10):一个无约束的轴向 DOF 是好的 gauge,但是一个
  不可学习的目标。

---

## 文件

- `source_adapter.py` —— `SourceAdapter` 抽象基类(你逐源实现的四个方法,含源→13ch 编码)+ `Topology`/`Clip`
  dataclass + 指向两条参考路径的 `BvhPipelineAdapter` / `PoseSourceAdapter` 模板桩。
- `build_source.py` —— driver:收集 adapter 的 `Clip` → per-object `mean/std` →
  `motions/`+`cond.npy`+captions+splits → Gates A/B/C/E + `_gate_report.json`。**不做任何编码。**
- `gates.py` —— A/B/C/E 作为可运行函数(+ 一个 CLI 回归模式)。
- `_ref_recover.py` —— 内置的纯 numpy FK/RIC 恢复(`recover_from_bvh_rot_np` / `recover_from_bvh_ric_np`),供
  Gates B/C 使用,使 gate 运行器只需 numpy。

*可镜像的参考实现:* Planet-Zoo 路径
(`tools/planetzoo/planetzoo_fulltopo_bvh_export.py` → `utils.process_new_skeleton` →
`data_loaders/truebones/truebones_utils/motion_process.py`,全部**在本 repo**)已经从 BVH 产出这套 IR。
HumanML3D/SMPL 的 Kabsch 转换器位于**训练 repo**,为 `scripts/convert_humanml3d_to_anytop13.py`,且**未在此内置** ——
对位姿型源,实现 spec §2/§7 描述的 Kabsch 重编码,而不是导入那个文件。
