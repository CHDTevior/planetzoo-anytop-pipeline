# AnyTop-13 中间表示 (Intermediate Representation, IR)

对 Graph-VQVAE → Graph-CodeFlow 流水线所用动作表示的**源无关、通道级**规格说明,以及一个**新数据源**要进入该流水线必须满足的确切契约。这是每一个数据准备 adapter(以及每一个运行
[`tools/ir_harness/`](../tools/ir_harness/README.md) 中 harness 的 agent)都必须遵循的参考。

> **为什么要一套 IR。** 当前流水线已经把两个结构上迥异的源合并进**同一个**共享码本:Planet Zoo 动物
> (311 个拓扑,原生 BVH 旋转、带真实轴向 twist)和 HumanML3D/AMASS 人体(22 关节 SMPL)。它们能共存,是因为
> 共享这套确切的 13 通道布局、per-parent rot6d 约定、以及 per-skeleton 图字段——**只有前端转换是逐源的**。
> 本文把这条边界钉死,使得未来的源(别的游戏、别的 mocap、别的骨骼绑定)只需写一个 adapter 即可加入,而无需
> 改动模型。

范围:本 repo(`planetzoo-anytop-pipeline`)负责**数据获取与准备**——把原始源资产变成 IR。IR 必须遵守的
FK/图**消费**约定在此以契约形式给出;本 repo 中的参考编码器/解码器是
`data_loaders/truebones/truebones_utils/motion_process.py`。

---

## 1. 张量:`motion` = `float32 [T, J, 13]`

一个 clip 是 `[T, J, 13]`:`T` 帧,`J` 关节,每关节 13 通道。**`T = F − 1`**,其中 `F` 是源帧数——最后一帧
被丢弃,因为速度和 contact 是帧间差分(`motion_process.py get_motion`)。以**未归一化**形式存储(原始的近物理
量纲,见 §4)。`motions/*.npy` 中的关节顺序是源的**原始 (original)** 顺序——见 §3 的顺序陷阱。

### 1.1 通道映射

| ch | 非根关节 `j≥1` | 根关节 `j=0` |
|----|----------------------|------------------|
| `0:3` | **RIC position** —— 关节 xyz,相对根原点,表达在逐帧的根朝向(yaw 已消除)坐标系中 | 根 RIFKE 状态;**只有 `ch1` 有意义 = 根高度 (Y)**。`ch0`、`ch2` 不被任何解码器使用 |
| `3:9` | 本关节**父节点**的 **6D rotation**(Zhou et al. 2019)—— 见 §2 | 逐帧**根朝向 (facing)** 旋转 (6D),用于积分根位移 |
| `9:12` | 逐关节局部速度(辅助;见 §1.2) | `ch9`、`ch11` = 根 **X, Z 线速度**(逐帧位移,根局部);`ch10` 未用 |
| `12` | 二值**脚部 contact** 标志 `∈{0,1}`(在脚上有意义,其余处为 `0`) | 通常为 `0`(非硬性写死:若某源的 contact 关节选择包含根——如无腿的蛇——可以设置它;IR 契约要求根上保持 `0`,除非源刻意选中根) |

分块边界 `0:3 | 3:9 | 9:12 | 12` 在 `docs/PLANETZOO_ANYTOP_USAGE.md:292-301` 中逐字列出。

### 1.2 每个通道实际驱动什么(已从代码核实)

存在两个解码器:**FK/skinning 由旋转驱动;RIC 路径由位置驱动**(它读 `ch0:3` 位置)。承载**生成**信号的是旋转,
但这两者并非都由旋转驱动:

- **FK / skinning**(`recover_from_bvh_rot_np`;skinning 的 `decode_feature_rotations`)**只**读:
  非根 `ch3:9`(per-parent 旋转)+ 根 `ch1`、`ch3:9`、`ch9`、`ch11` + 骨骼的
  `offsets`/`parents`。它**忽略**所有 `ch0:3`(非根 RIC)、`ch10`、非根 `ch9:12` 和 `ch12`。
- **RIC 路径**(本 repo `motion_process.py` 中的 `recover_from_bvh_ric_np`;训练 loader 中的
  等价方法是 `_recover_world_positions`)—— 用于构建世界位置训练目标并作为交叉校验 —— 读非根 `ch0:3` +
  同一套根状态。

因此**承载信号**是:6D 旋转 `ch3:9` **加上根状态——三个标量(`ch1` 高度、`ch9`/`ch11` XZ 速度)和根
facing-6D(关节 0 的 `ch3:9`)**。`ch0:3`(非根)只对 RIC 路径承载。在每个解码器里都真正惰性的:**根 `ch0`/`ch2`,
以及所有关节的 `ch10`(velocity-Y)**。`ch12` 不被 FK/RIC/skinning 使用,但被 remove-joints 数据增强消费,并以
`foot_contact_per_joint` 暴露。

> **对新源的设计含义。** 你必须把 `ch3:9`(旋转,per-parent)和根状态(`ch1`、`ch9`、`ch11`、根 `ch3:9`)
> 弄得完全正确;`ch0:3` 必须与它们自洽(§5 不变量)。`ch10`、根 `ch0`/`ch2` 可以为零。`ch12` 必须是脚上的真实
> contact 信号。

---

## 2. per-parent rot6d 约定(头号语义微妙点)

**对一个非根子槽 `j`,`motion[:, j, 3:9]` 存储的是 `j` 的**父节点**(`parent[j]`)的局部旋转,而**不是**关节
`j` 自身的旋转。** 根槽存储逐帧的 facing 旋转。

- **编码**(`motion_process.py get_bvh_cont6d_params`):`for j,p in enumerate(parents[1:],1):
  cont_6d_reordered[:, j] = cont_6d[:, p]`;槽 0 = 帧 facing 四元数的 6D。
- **解码**(`recover_from_bvh_rot_np`):`for j,p in enumerate(parents[1:],1):
  rot_q[:, p] = all_q_hml[:, j]` —— 子槽 `j` 处的 token 被写回父槽 `p`。

后果:
- **叶子 / 末端执行器**关节(不是任何人的父节点)**没有存自身旋转的槽** → FK/skinning 让叶子停在
  rest/identity。这是正确的:叶子自身的朝向在这里不可观测。
- 一个有**多个孩子**的父节点,其单一旋转被复制到那些子槽里。在干净数据上它们全部一致;在带噪/生成数据上,两个
  解码器在分叉点会不同(FK 是 last-child-wins;skinning POC 通过 `child_for_parent` 字典确定性地 first-child-wins)。

> **陷阱:** 天真地把 `motion[:, j, 3:9]` 当成"关节 `j` 的旋转",会把每个旋转施加到错误的骨头上(沿链条向上错一位)。
> 永远使用本 repo 的 `recover_from_bvh_rot_np`。

---

## 3. 关节顺序 —— `new_to_old_perm` 陷阱

存在**两套关节顺序**,把它们混用会静默产生**假的腿翻转 (fake leg-flips)**(看起来像模型 bug,实则是数据管道
bug——曾让我们花掉一整个调试 session,见文末注)。

- **ORIGINAL 顺序** —— `motions/*.npy`、`cond.npy`(所有数组)、以及导出的 minipack `skeleton.json` 都在源的
  原始关节顺序。
- **FK/BFS 顺序** —— 训练 loader 对每个拓扑做 BFS 重排,使 `parents[0]==-1` 且 `parents[j] < j`,并存下
  `new_to_old_perm`(`new_to_old_perm[new_idx] = old_idx`)。在 `__getitem__` 里它对原始 clip 重新索引:
  `raw_motion = raw_motion[:, cond["new_to_old_perm"], :]`。**模型看到或导出的一切都在这套 NEW 顺序**;归一化
  cond 缓存(`_cond_normalized_J*.pkl`)也在 NEW 顺序。

**转换:** `inv = np.argsort(new_to_old_perm); motion_old = motion_new[:, inv, :]`(NEW→ORIGINAL),
或 `motion_new = motion_old[:, new_to_old_perm, :]`(ORIGINAL→NEW)。任何 FK / skinning / 分析都必须对 motion
**和** `parents`/`offsets`/骨骼一致地使用**同一套**顺序。

> 经验法则:**模型输出 / 生成导出是 NEW 顺序;skinning 流水线 + `cond.npy` 是 ORIGINAL 顺序。** skinning 之前
> 用 `argsort(new_to_old_perm)` 转换导出(参考导出器现在已经这么做)。

FK 重排是源无关的,但**要求恰好一个根(`parent==-1`)且是连通、单根的树**,否则 loader 会报错。

---

## 4. 归一化

per-**object-type**、per-**joint**、per-**channel** 的 mean/std,以 `[J,13]` 存于 cond 记录中:

```
std_safe = std + 1e-6          # _STD_FLOOR
normed   = (raw - mean) / std_safe
```

- 模型消费**归一化**视图;FK/RIC/skinning 消费**原始 (raw)** —— 先反归一化:
  `raw = normed * (std + 1e-6) + mean`(用 FK 顺序的 `mean`/`std`),**然后**若需要 ORIGINAL 顺序再施加关节
  置换(§3)。只做二者之一会留下不匹配。
- std 在构建时做**按通道组均衡化**(`motion_process.py get_mean_std`):**根**关节的 pos(`0:3`)/rot(`3:9`)/vel(`9:12`)
  各块分别塌缩为该块自身的标量均值,而**所有非根关节共享每块一个标量**(在该块内、对每个非根关节*且*每个通道
  取均值)—— 非根关节**不**各自独立均衡化。Contact(`12`):非零 std → 它们的均值;零 → `1.0`。存储的 std 中
  **不**烘焙 `1e-6` —— 它在归一化时加上(上面的 `std + 1e-6`)。
- 因为归一化是**每骨骼自包含**的,**合并源无需重新归一化** —— 并集 `cond` 只是记录的拼接。绝不复用另一个骨骼的
  统计量。

---

## 5. 世界恢复 (FK) 与自洽不变量

从**原始 (raw)** 13ch 恢复世界关节位置 `[T, J, 3]` 的两条独立路径:

- **FK 路径** `recover_from_bvh_rot_np(data, parents, offsets)` —— 6D 旋转 `ch3:9` → 旋转矩阵 →
  在 `offsets` 上做 local→global 骨链 matmul;根平移来自根状态(`ch1`、`ch9`、`ch11`、facing)。
- **RIC 路径** `recover_from_bvh_ric_np(data)`(训练 loader 等价:`_recover_world_positions`)
  —— 非根 `ch0:3` 被逐帧根 facing 反向旋转 + 积分根 XZ。

**根恢复(两条路径共享,skinning 的 `recover_root_positions` 也是):** facing = 关节 0 的 `ch3:9`;累积 XZ =
被反 facing 旋转后的 `(ch9, ch11)` 逐帧速度的 `cumsum`;高度 = `ch1`(直接取,不积分)。无 fps 除法 ——
`ch9/ch11` 是逐帧增量。

**核心不变量(下文 Gate C):** 在**一致关节顺序**下的 GT 数据上,FK 与 RIC 一致到 `L2 ≈ 0`(干净数据
`selfcheck < 1e-4`)。这是最强的单一数据完整性检查:它证明旋转通道能驱动 skinning。大 selfcheck ⇒ 要么关节顺序
错(§3),要么 rot6d 约定被破坏(§2)。

> **double-root-rotation 陷阱(已移除)。** 原版 AnyTop/SALAD 的 `recover_from_bvh_rot_np` 在重新索引后再施加
> `rot_q[:,0] = -r_rot_quat * rot_q[:,0]`,这会**二次施加**根 yaw(重排后的 root-child token 已携带 facing)。
> 本 repo 移除了它(`apply_root_cancel` 默认 **False**;`True` 仅为调试复现官方的 buggy 行为)。证据:带该修正时
> FK-vs-RIC `absL1 = 0.65`(转身 clip 上约 2× 旋转);不带时 `= 0.0000`。**该 bug 在近静止 clip 上不可见** ——
> 任何回归检查必须用大根旋转 clip(转身、盘旋飞),绝不用静止位姿。

---

## 6. per-skeleton 图 / `cond` 字段(图模型消费的东西)

Motion 是逐 clip 的;**以下一切都是逐拓扑静态的**,存于 `cond` 记录。新源必须提供**primary(主要)**集;
**derived(派生)**集由 loader 为你计算。

> **是哪个 loader。** 这里的"loader"指 **Graph-VQVAE 训练 repo 的 `AnyTopDataset`**(本 IR 的消费者),它做
> FK-BFS 重排(§3)并派生下面的图字段。该模块位于训练 repo,**不**在本 `planetzoo-anytop-pipeline` repo ——
> 本 repo 那个遗留的 `data_loaders/truebones` loader 是*另一个*、更旧的 loader,**不是** IR 的消费者。因此
> harness 的 Gate E 只检查落盘契约;完整 ingest 在训练 repo 中验证。

**Primary(源必须提供,per object_type):**

| 字段 | shape | 作用 |
|-------|-------|------|
| `parents` | `[J]` int | 拓扑;**必须单根 & (FK 重排后) `parents[j]<j`**。一切图结构量都由它派生。 |
| `offsets` | `[J,3]` | rest-pose 骨向量;喂 `skeleton_features` **以及** FK 解码器(`rest_offsets`)。 |
| `joints_names` | `[J]` str | 驱动 `name_hashes` + `skeleton_features` 中的 left/right/center 侧别启发式。 |
| `tpos_first_frame` | `[J,13]` | rest/T-pose 作为一行 13ch(用于一致性核对/渲染的直通;对图模型不承载)。 |
| `mean`,`std` | `[J,13]` | 每骨骼归一化(§4);**必需**。 |

**由 loader 从 `parents`/`offsets`/`joints_names` 派生(源**无需**提供;它们会被重算):** `skeleton_features [J,9]`
(`norm_offsets(3)+bone(1)+depth(1)+degree(1)+side_onehot(3)` —— **唯一**被投影进初始关节 token 的静态字段)、
`adjacency [J,J]`、`geodesic_dist [J,J]`(真实 Floyd hop,未截断)、`joints_graph_dist [J,J]`(hop 距离**截断在 5** ——
Graphormer 的 hop-bucket 偏置,与 `geodesic_dist` 不同)、`joint_relations [J,J]`(6 类边类型:
self/parent/child/sibling/no_relation/end_effector)、`name_hashes [J]`(`md5(name)%1024`)。*`cond.npy` 中存在的
`joint_relations`/`joints_graph_dist` 只做 shape 检查然后丢弃并重新派生。*

**哪个字段驱动哪个机制(现役 v4b272 graphormer VQVAE):** 节点特征 = `skeleton_features`;关节级注意力偏置 =
`joints_graph_dist`(hop-bucket)+ `joint_relations`(边类型);图池化/粗化 = `adjacency` + `geodesic_dist`;
FK 解码 = `offsets`+`parents`;反归一化 = `mean`/`std`。`name_hashes` 是**存在但关闭**(v4b272 VQVAE 训练器中
`use_name_embed` 默认 False)。`tpos_first_frame`/`kinematic_chains` 是直通。

> **Backbone 注。** CodeFlow backbone 在**冻结的 VQVAE `z_q` token** 上训练,训练时**不**读 `cond` —— per-skeleton
> 图状态已被烘焙进导出的 token 缓存。要给 backbone 加一个新骨骼:(1) 把它加进 `cond` 以便 VQVAE 能编码,
> (2) **重跑 token 导出**。你无法在 backbone 推理时不重导出就换骨骼。

> `d_model % n_heads == 0`;`skeleton_features` 固定为 **9** 维、motion 为 **13** ch —— 新源必须精确匹配这些。
> `max_joints` padding 目标必须 `≥` 源的最大骨骼(更大的骨骼在建数据集时被静默跳过)。

---

## 7. 通用 vs 逐源(IR 边界)

**通用核心(对每个源都相同 —— 不要逐源重新实现):**
`[T,J,13]` 布局 + 根特判;**per-parent rot6d** 约定;**全部**图字段(`parents`/`offsets`/`names` 的纯函数);
FK-BFS 排序 + `new_to_old_perm`;per-object `mean/std` 归一化;padding/masking;caption/T5 流水线(以 `motion_id`
为键)。

**逐源 adapter(新源必须实现的):**

| adapter 关注点 | 做什么 |
|---|---|
| 拓扑 | 为源绑定定义 `parents`/`offsets`/`joints_names`(单根、连通) |
| 坐标系 / up-axis | 变换到 **Y-up、+Z-facing**。(PZ 需要 Z=−90° roll + hips→chest yaw + 一次朝向选择测试;HumanML3D 已经是 Y-up/+Z。) |
| 单位尺度 | 缩放使 **平均骨长 = `HML_AVG_BONELEN = 0.2092142857142857`**(共享的度量锚) |
| 根原点 + 地面 | 减去 T-pose 初始根 XZ;平移使脚触 `Y=0` |
| fps / 时序 | 重采样到 **20 fps**;输出 `T = F−1`(丢最后一帧) |
| rot6d 重编码 | 通过对世界位置做逐关节 Kabsch,把旋转重新表达到 AnyTop 的 rest 基,使 FK(rot) 复现 RIC(**直接 gather 原生旋转是错的** —— 高达约 89° 的骨基误差);确定性地固定单孩子 twist DOF(或一致地携带真实 twist) |
| 根约定 | 重建根 facing-6D + 把根 XZ 速度重打包进 `ch9`/`ch11`;把任何角-Y-速度源积分成 yaw |
| contact | 选择源的脚部关节;在它们上产生/阈值化 `ch12`,其余处为 `0` |
| helper/控制关节剪枝 | 转换前丢弃非解剖关节(IK/twist/control) |
| captions | 产出以 `motion_id` 为键的 `motion_texts_by_file.json` |

---

## 8. 新源**契约**(adapter 必须满足的检查清单)

1. 产出 `motions/*.npy` `[T,J,13]`,带精确通道映射(§1)和根特判。根在 original 顺序里必须是**关节 0**
   (`parents[0]==-1`)—— 根特判和归一化都以索引 0 为键(Gate C / harness 强制)。
2. 产出 `cond` 记录:`parents`(单根 `-1` 在索引 0、连通)、`offsets [J,3]`、`joints_names [J]`、
   `tpos_first_frame [J,13]`,以及**在 train split 上新鲜重算的 `mean`/`std [J,13]`**(绝不复用源自己的
   flat Mean/Std)。
3. Y-up / +Z-facing(§7)。
4. 缩放使**参考骨子集的均值** = `HML_AVG_BONELEN`(参考实现 `motion_process.scale`;该子集是
   `scale_joint_indices`,逐源)。把该子集声明为 `Topology.scale_ref_joint_idx`,以便 Gate E 能**精确**验证尺度;
   若你留 `None`,Gate E 只能施加**粗略**带(全骨均值 ~0.3×–3×),因为该参考子集无法从 `cond` 另行恢复(在现有
   382 个 object_types 上,全骨均值正因此横跨 ~0.39×–1.42×)。
5. 20 fps,`T = F−1`。
6. `ch3:9` 用 **per-parent** 约定,使 FK(rot) 复现 RIC(§5),twist DOF 已固定。
7. `ch12` contact 在所选脚部关节上,其余处为 `0`。
8. 根 facing-6D(`ch3:9`,关节 0)+ 根 XZ 速度重打包进 `ch9`/`ch11`。
9. Captions JSON + train/val split。
10. 训练前**通过 Gates A/B/C/E**(driver,调用为 `build(adapter, out_root, max_joints=…)`,任一红硬 gate 会抛
    `AcceptanceError`)**以及**强制的手工 Gate D(视觉 QA)。

### 验收 gates(视觉 QA 是强制的,且高于 metrics)

- **Gate A —— 逐 clip 完整性(每个 clip)。** shape `[T,J,13]`;有限;`T=F−1`(提供 `source_frames` 时为硬约束)
  且 `T>0`;绝对值合理;contact 二值 `∈{0,1}` 且在声明的 `foot_joint_idx` 之外为 `0`;rot6d 行非退化;以及
  **multi-child rot6d 复制**不变量(一个父节点的所有子槽携带*相同*的 rot6d,否则 FK/skinning 解码器发散)。
- **Gate B —— RIC 恢复等价(独立)。** 源自己的官方世界恢复,以 `Clip.source_world [T,J,3]` 提供,对
  `recover_from_bvh_ric_np(ch0:3)` 必须匹配到 gate 容差(`mean_l2 ≤ 1e-4`;HumanML3D 达到 ~0)。在**每个**携带
  `source_world` 的 clip 上运行。若源没有官方恢复,Gate B 被**豁免 (waived)**(在报告中显式标注),不是静默通过。
  验证 RIC 打包。
- **Gate C —— FK 路自洽(核心不变量)。** 对**每个 clip**(`build_source` 逐 clip 运行;报告为每个类型保留一个代表),
  `recover_from_bvh_rot_np`(FK)对 RIC 路径一致到 `L2 ≈ 0`(干净 `< 1e-4`;`gate_c_mode="inferred_twist"` 时
  ≤0.5% bbox)。该 gate 内部做 BFS 重排,因此对任意输入关节顺序都正确。验证 `offsets` + per-parent rot6d 约定。
  double-root 陷阱在静止 clip 上不可见,所以每个拓扑必须有一个**大根旋转** clip(§5);容差档位逐拓扑声明,绝不从
  object_type 名字推断。
- **Gate D —— 视觉 QA(强制、手工)。** GT-vs-转换的并排 **GIF/视频**(骨架,以及对可蒙皮绑定的网格),用 repo 的
  渲染工具手工渲染。视觉正确性**高于**任何 metric —— 腿翻转 / 蜷缩能通过每一项标量检查。从清晰侧视角渲染;确认所有
  肢体着地、无关节内翻。
- **Gate E —— 布局契约(结构性)。** `cond.npy` 可加载且非空;每个 `object_type` 都有必需键、有限的 `mean/std`、
  单根 `parents`、合理的骨长尺度;每个 motion 映射到已知拓扑;split 存在、非空、且 train/val **不相交**。这只是落盘
  契约 —— **完整**的 loader ingest(实例化 `AnyTopDataset`、派生图字段,加上逐通道 + 时序二阶差分分布与已知良好切片
  的匹配)在**训练 repo** 里跑,不在本 harness。

---

## 9. 当前(Planet Zoo)数据是如何获取的

端到端列出,以便新游戏/mocap 源可以镜像这些阶段。完整细节见
[`docs/PLANETZOO_ANYTOP_PIPELINE.md`](PLANETZOO_ANYTOP_PIPELINE.md)、
[`docs/ANIMO4D_ANYTOP_PIPELINE.md`](ANIMO4D_ANYTOP_PIPELINE.md)、
[`docs/ANIMO4D_ANYTOP_DATA_LINEAGE.md`](ANIMO4D_ANYTOP_DATA_LINEAGE.md)。

| 阶段 | 工具 | 输入 → 输出 |
|---|---|---|
| 0. 提取资产 | cobra-tools(外部) | Steam 安装 → `01_ovl_extracted/<Object>.ovl`(`.ms2` 网格 + `.manis` 动画) |
| 1. BVH 导出(在 Blender 中) | `tools/planetzoo/planetzoo_fulltopo_bvh_export.py`(+ `*_batch_/*_parallel_` 包装器,每个对象一个全新 Blender) | `.ms2`/`.manis` → 全拓扑 `raw_bvhs/*.bvh` @20fps + `*__tpos.bvh` + `export_manifest.jsonl`;剥掉一个 wrapper ROOT 使 AnyTop 看到单根树 |
| 2. AnyTop 13ch 转换 | `tools/planetzoo/planetzoo_parallel_anytop_process.py` → `utils.process_new_skeleton` → `motion_process.py` | BVH → `motions/*.npy [T,J,13]` + per-object `cond.npy`。做:剪 `srb`/twist helper → yaw 到 +Z → roll 到 Y-up → 原点/地面/尺度 → **rot6d 编码(per-parent)** + RIC + 局部速度 + contact |
| 3. 对齐文本 | `tools/planetzoo/build_animo4d_anytop_manifest.py` | 文本索引 ↔ 原始 stem ↔ 处理后 npy → matched/missing 状态 |
| 4. 附加 captions | `build_animo4d_anytop_text_manifest.py`(官方文本)/ `build_planetzoo_text_manifest.py`(通用) | per-npy captions;**缺文本保留为空串,motion 绝不丢弃** |
| 5. 打包 | `tools/planetzoo/pack_planetzoo_anytop_dataset.py` | per-object 文件夹 → 一个池化布局 + 合并 `cond.npy`;**硬链接 = 无损** |
| 6. 修复/审计 | `tools/planetzoo/repair_bad_motion_values.py` | 隔离越界 npy + 重算 `mean/std` |

`expand_minipack_motion_to_full_rig.py` 是一个 **skinning/可视化**工具(按关节名把精简 minipack 动作映射到全绑定,
把每个父节点的 rot6d 广播到其子槽,把省略/叶子关节冻结在 rest)—— **不是**有效的全拓扑训练目标。

---

## 10. 已踩过的坑(带进每个新源)

- **关节顺序不匹配 → 假腿翻转。** 模型输出是 NEW 顺序;skinning + `cond.npy` 是 ORIGINAL 顺序。用
  `argsort(new_to_old_perm)` 转换(§3)。诊断法:在 motion 自己的顺序里重跑 Gate C —— GT 在那里是干净的。
- **double-root-rotation。** 绝不重新加回 `apply_root_cancel`;在转身 clip 上测,别用静止(§5)。
- **跨源直接 gather 旋转是错的。** 通过 Kabsch 重编码到 AnyTop 的 rest 基(§7)。
- **从低 DOF 输入推断高 DOF 目标**(如从位置推轴向 twist)会产生一个无约束 DOF,它是一个好的 gauge,但除非在编码时
  固定到一个规范值,否则是一个**不可学习的目标** —— 在数据里修,不要在 loss 里修。
- **水中/行为 clip 在平地上看起来是错的。** 做陆地 skinning **demo** 时,挑移动 clip(walk/run/trot);
  `walktoswim`/`drink` clip 会真实地弓起身体(GT 也如此)—— 那是动作,不是 bug。
- **Metrics 漏掉几何。** R-precision/FID/freeze-detector 都可能通过,而某肢体却内翻。Gate D(视觉)是强制且权威的。

---

*配套:agent 可运行的数据准备 harness —— [`tools/ir_harness/README.md`](../tools/ir_harness/README.md)。*
