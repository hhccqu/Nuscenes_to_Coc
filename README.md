# NuScenes-CoC：基于 nuScenes 复现 Alpamayo-R1 CoC 数据集

本项目在公开的 [nuScenes](https://www.nuscenes.org/) 数据集上，完整复现了 NVIDIA [Alpamayo-R1](https://arxiv.org/abs/2505.03047) 论文中提出的 **Chain-of-Context（CoC）** 标注流程，并输出与官方格式对齐的 `ood_reasoning.parquet`。

---

## 项目背景

Alpamayo-R1 论文提出了一种驾驶数据集标注方法，每条视频片段包含：

- **驾驶决策**：纵向动作 × 横向动作的组合（如"减速跟车 + 保持居中"）
- **思维链（CoT）**：一句英文短语，解释自车为何做出该决策
- **关键组件**：触发该决策的因果场景元素（前车、行人、停止线等）

本项目将这套方法在 nuScenes 上端到端复现：

```
nuScenes 原始数据
      ↓  [候选片段提取 + 规则决策标注]
CoC 候选样本（JSON）
      ↓  [VLM 教师标注，qwen-vl-max]
含英文 CoT 的 CoC 数据集（JSON）
      ↓  [格式导出]
ood_reasoning.parquet（对齐官方 Alpamayo-R1 格式）
```

---

## 已完成功能

| 模块 | 状态 | 说明 |
|------|------|------|
| 片段提取 | ✅ | 滑动窗口从 nuScenes 样本中检测候选片段 |
| 规则决策标注 | ✅ | 基于自车运动学、地图拓扑和周围目标，生成 6 纵向 × 4 横向决策类别 |
| 低信号过滤 | ✅ | 过滤无因果触发的平凡片段（18m 同车道无前车时丢弃 set_speed_tracking） |
| VLM 教师标注 | ✅ | 调用 qwen-vl-max（或任意 OpenAI 兼容接口），输入 CAM_FRONT 图像 + 结构化场景上下文 |
| 英文 CoT 生成 | ✅ | 输出与官方 Alpamayo-R1 风格一致的动作短语 |
| 官方格式导出 | ✅ | 导出 `ood_reasoning.parquet`，schema：`clip_id / feature / event_cluster / events / split` |
| 可视化人工评测 | ✅ | 自包含 HTML 报告，含图像展示、决策徽章、评分按钮 |

---

## 仓库结构

```
Coc/
├── nuscenes_coc/                        # 核心库
│   ├── cli.py                           # 主流程编排（build_dataset）
│   ├── segment_filter.py                # 候选片段检测与过滤
│   ├── decision_rules.py                # 规则决策逻辑（纵向 / 横向）
│   ├── component_extractor.py           # 关键场景组件提取
│   ├── teacher_prompt.py                # VLM system/user prompt 及 JSON schema
│   ├── geometry.py                      # 自车坐标系变换
│   ├── motion.py                        # 速度 / 加速度计算
│   ├── nusc_access.py                   # nuScenes API 封装
│   ├── meta_actions.py                  # 动作分类定义
│   ├── exporter.py                      # JSON 序列化工具
│   ├── quality.py                       # 后处理质量检查
│   └── text_templates.py                # 兜底文本模板
│
├── scripts/
│   ├── generate_nuscenes_coc.py         # 第1步：提取片段 → CoC JSON
│   ├── build_teacher_labeling_assets.py # 第2步：构建 VLM 请求 JSONL
│   ├── run_teacher_llm_labeling.py      # 第3步：调用 VLM API，写入响应
│   ├── export_to_official_format.py     # 第4步：导出为 parquet 格式
│   └── generate_eval_report.py          # 可选：生成 HTML 人工评测报告
│
├── requirements.txt
└── README.md
```

---

## 快速开始

### 第一步：环境配置

```bash
conda create -n coc_nuscenes python=3.10 -y
conda activate coc_nuscenes
pip install -r requirements.txt
pip install pandas pyarrow openai
```

> **Windows 注意事项：** nuScenes 地图加载需要 `maps/expansion/` 路径可访问。若遇到 `FileNotFoundError`，以管理员身份运行 PowerShell 创建目录链接：
> ```powershell
> cmd /c mklink /J "nuScenes数据路径\maps\expansion" "nuScenes数据路径\maps\nuScenes-map-expansion-v1.3\expansion"
> ```

### 第二步：下载 nuScenes 数据

从 [nuScenes 官网](https://www.nuscenes.org/nuscenes#download) 下载 mini 版（或完整 trainval），解压后目录结构如下：

```
nuscenes-mini/
  v1.0-mini/
  samples/
  sweeps/
  maps/
```

### 第三步：运行完整流程

```bash
DATA_ROOT="nuScenes数据路径"
cd Coc

# 第1步 - 提取 CoC 候选片段
python scripts/generate_nuscenes_coc.py \
    --data-root "$DATA_ROOT" \
    --version v1.0-mini \
    --output out/nuscenes_coc.json \
    --stats-output out/stats.json

# 第2步 - 构建 VLM 标注请求
python scripts/build_teacher_labeling_assets.py \
    --input out/nuscenes_coc.json \
    --output-dir out

# 第3步 - 调用 VLM（需 OpenAI 兼容接口的 API Key）
python scripts/run_teacher_llm_labeling.py \
    --requests-input out/teacher_requests.jsonl \
    --responses-output out/teacher_responses.jsonl \
    --model qwen-vl-max \
    --base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
    --api-key "YOUR_API_KEY" \
    --with-images \
    --cameras CAM_FRONT \
    --image-frames 0

# 第4步 - 合并响应，生成最终数据集
python scripts/build_teacher_labeling_assets.py \
    --input out/nuscenes_coc.json \
    --output-dir out \
    --responses out/teacher_responses.jsonl \
    --final-output out/nuscenes_coc_final.json

# 第5步 - 导出为官方 parquet 格式
python scripts/export_to_official_format.py \
    --input out/nuscenes_coc_final.json \
    --output-dir out/official \
    --auto-split
```

最终输出 `out/official/reasoning/ood_reasoning.parquet`，与 Alpamayo-R1 官方格式完全对齐。

---

## 输出格式说明

导出的 parquet 文件遵循 Alpamayo-R1 官方 schema：

| 字段 | 类型 | 说明 |
|------|------|------|
| `clip_id` | string（索引） | 唯一片段标识符 |
| `feature` | string | 摄像头模态（`camera_front_wide_120fov`） |
| `event_cluster` | string | 高层事件类别（如 `PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY`） |
| `events` | dict | `{event_start_frame, event_start_timestamp, cot}`，即 CoT 标注 |
| `split` | string | `train` 或 `val` |

### 决策到事件类别的映射

| 纵向决策 | 事件类别 |
|----------|---------|
| `lead_obstacle_following` | `SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR` |
| `speed_adaptation_road` | `PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY` 或 `WORK_ZONES_TEMP_TRAFFIC_CONTROL` |
| `stop_static_constraint` | `WORK_ZONES_TEMP_TRAFFIC_CONTROL` |
| `yield_agent_right_of_way` | `PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY` |
| `maintain_speed`（转弯车道） | `COMPLEX_INTERSECTION_INTERACTION` |
| 其他 | `SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR` |

---

## 人工评测

生成自包含 HTML 报告，在浏览器中对标注结果进行人工打分：

```bash
python scripts/generate_eval_report.py \
    --input out/nuscenes_coc_final.json \
    --output out/eval_report.html
```

用任意浏览器打开 `eval_report.html`，每条样本卡片包含：

- CAM_FRONT 图像（历史帧 → 关键帧 ★）
- 彩色决策徽章、VLM CoT 标注、规则系统原始判断（对比参考）
- 关键组件结构化表格
- **✓ 正确 / ~ 部分正确 / ✗ 错误** 评分按钮（自动保存到浏览器 localStorage）
- 一键导出评分结果为 JSON

---

## VLM 兼容性

`run_teacher_llm_labeling.py` 支持任意 **OpenAI 兼容** 接口：

| 服务商 | 推荐模型 | `--base-url` |
|--------|---------|-------------|
| 阿里云百炼（DashScope） | `qwen-vl-max` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| OpenAI | `gpt-4o` | 默认，无需指定 |
| 本地（Ollama） | `llava` 等 | `http://localhost:11434/v1` |

---

## 驾驶决策分类

### 纵向（6类）

| 类别 | 含义 |
|------|------|
| `lead_obstacle_following` | 跟随前车，保持安全距离 |
| `speed_adaptation_road` | 根据道路/环境条件调整速度 |
| `stop_static_constraint` | 在停止线/交通管控处停车 |
| `yield_agent_right_of_way` | 礼让行人或穿越目标 |
| `maintain_speed` | 保持当前速度巡航 |
| `set_speed_tracking` | 跟踪设定速度（无因果触发时过滤） |

### 横向（4类）

| 类别 | 含义 |
|------|------|
| `lane_keeping_centering` | 保持车道居中 |
| `in_lane_nudge_left` | 车道内小幅左偏 |
| `in_lane_nudge_right` | 车道内小幅右偏 |
| `lane_change_left/right` | 完整换道 |

---

## 与官方 Alpamayo-R1 的差距

| 方面 | 官方数据集 | 本项目 |
|------|-----------|--------|
| 数据规模 | ~1,740 条 | 14 条（mini）/ ~1,000 条（trainval，待测试）|
| CoT 质量 | 人工审核 | VLM 生成（qwen-vl-max）|
| `event_start_timestamp` | 相对毫秒时间戳 | nuScenes 绝对微秒时间戳 |
| 数据来源 | NVIDIA 内部数据 | 公开 nuScenes |

---

## 参考资料

- [Alpamayo-R1 论文（arXiv）](https://arxiv.org/abs/2505.03047)
- [nuScenes 数据集](https://www.nuscenes.org/)
- [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit)
