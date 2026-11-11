# 复现数据对照清单（reproduction-data-mapping.md）

- 适用稿件：`llm-logfusion-csf2027-v2.tex`（《软件学报》投稿版）
- 原则：**一切数值以本次复现产物为准**。系统重装前的原始数据、LLM 调用日志与多次运行产物一律不再作为依据。
- 复现日期：2026-11-11
- 结果根目录：`results/tables/`（CSV/JSON，由脚本直接写出，未手工编辑）

---

## 0. 复现环境与运行方式

| 项目 | 值 | 证据 |
|---|---|---|
| 机器 | Intel Core i9-14900HX（24 核 32 线程）、32 GB、Windows 11 10.0.26100、无 GPU | 本机 `Win32_Processor` / `Win32_ComputerSystem` |
| Python | 3.12.10 | `python --version` |
| 依赖 | numpy 2.5.4、pandas 3.1.1、scikit-learn 1.9.1、scipy 1.18.1、matplotlib 3.11.2、torch 2.15.0+cpu | `pip list` |
| 语义后端 | **离线替代后端（LSA）**：TF-IDF + 截断 SVD → 384 维，经 Nyströmformer 线性注意力（k=64，eps=1e-6）+ 均值池化 | `llm_logfusion/embedding.py`；`results/metrics.json:backend = offline-surrogate` |
| 数据 | `data/extended_logs.csv` 42 940 行、异常 2 395、1 010 条 campaign；`data/darpa_tc_stage_pairs.csv` 2 500 条阶段对 | 数据文件本身 |
| 全量复现 | `python scripts/run_reproduction.py`，350 s 完成 | `.tmp/repro_after_fix2.log` |
| 扩展实验 | `python scripts/run_extended_checks.py`（lowrate / threshold / leakage / chain / repeats / reliability） | `.tmp/extra_after_fix.log` |
| 基线 5 种子补充 | `python .tmp/run_baselines_5seed.py`（分种子缓存，可续跑） | `.tmp/baselines_after_fix2.log` |
| 融合层受控诊断 | `python .tmp/diagnose_fusion.py` | `.tmp/diagnose_fusion2.log`；`results/tables/diagnosis_fusion.csv` |
| 图 2–图 8 重绘 | `python .tmp/make_figs_repro.py`（读取 `results/tables/*.csv`） | `figures/*_cn.png` |

**重要修订（2026-11-11）**：第一层孤立森林原实现按列均匀随机抽取分裂维，384 维语义块因此获得 96.5% 的分裂机会，14 维结构化块被淹没。修复为块均衡分裂（`IF_BLOCK_BALANCE=True`、`IF_BLOCK_SIZES=(14,384)`）后全套实验已重跑。修复前的全套产物完整备份在 `results/before_fix/`，诊断过程与证据见 [fusion-diagnosis.md](fusion-diagnosis.md)。本文件以下所有数值均为**修复后**结果。

说明：`sentence-transformers` 在本环境无法安装（下载无进展），语义编码自动回退到 `lsa`。正文 6.1 节已如实声明该替换，因此“384 维语义嵌入”的语义类结论只在该替代后端上成立。

---

## 1. 表 1 / 图 2（数据集构成）

| 论文数值 | 复现文件 : 字段 | 状态 |
|---|---|---|
| 各平台日志数 5 720 / 5 100 / 4 950 / 4 780 / 4 680 / 4 520 / 4 450 / 4 370 / 4 370 | `results/tables/table_dataset.csv : logs` | 替换（新语料） |
| 各平台异常数 269 / 194 / 198 / 103 / 398 / 112 / 441 / 353 / 327 | `table_dataset.csv : anomalies` | 替换 |
| 各平台异常率 4.70 / 3.80 / 4.00 / 2.15 / 8.50 / 2.48 / 9.91 / 8.08 / 7.48 % | `table_dataset.csv : ratio` | 替换 |
| 合计 42 940 / 2 395 / 5.58% | `table_dataset.csv` 求和；`results/metrics.json` 同源 | 替换（旧稿 2 177 / 5.07% 已删） |
| 1 010 条攻击链 | `data/extended_logs.csv : campaign` 去重计数 | 替换 |

---

## 2. 6.2 节主实验 / 图 3

| 论文数值 | 复现文件 : 字段 | 状态 |
|---|---|---|
| 精确率 0.963±0.012 | `results/tables/table_main.csv : precision`（5 行均值/样本标准差） | 替换（修复前 0.975） |
| 召回率 0.820±0.031 | `table_main.csv : recall` | 替换（修复前 0.472） |
| F1 0.886±0.021 | `table_main.csv : f1`；`results/metrics.json:f1_mean` | 替换（修复前 0.634） |
| 假正率 0.19%±0.06% | `table_main.csv : fpr` | 替换 |
| LLM 调用比例 10.29%±0.22% | `table_main.csv : call_rate`；`metrics.json:call_rate` | 替换（修复前 9.95%） |
| 实测吞吐 1.66×10⁴ 条/s | `table_baselines_summary.csv : measured_throughput`（默认级联行） | 替换（修复前 1.35×10⁴） |
| 估算吞吐 156.1 条/s | `table_baselines_summary.csv : estimated_throughput` | 替换（按 61.7 ms/次摊销；该常量沿用，见 §10） |
| 各平台 F1 0.984/0.968/0.962/0.962/0.957/0.944/0.879/0.857/0.522 | `table_per_platform.csv` 按 `platform` 求均值 | 替换（修复前 0.909/0.826/…/0.358/0.371） |
| 平台召回 0.360~0.969、精确率≥0.886 | `table_per_platform.csv : recall/precision` | 替换（修复前召回 0.232~0.847） |
| 平台级 F1 标准差 0.006~0.092 | `table_per_platform.csv : f1` 分组标准差 | 替换（修复前 0.049~0.177） |

图 3 直接从 `table_per_platform.csv` 绘制，红色虚线为 `table_main.csv:f1` 的均值 0.89；图注已写明该线含义。

---

## 3. 6.3 节 LOPO / 图 4

| 论文数值 | 复现文件 : 字段 | 状态 |
|---|---|---|
| LOPO 平均 F1 0.569 | `table_lopo.csv : f1` 均值；`results/metrics.json:lopo_f1_mean` | 替换（修复前 0.288） |
| Firewall 0.968、OpenStack 0.911、Apache 0.761、DNS 0.065、BGL 0.153、Windows 0.404 | `table_lopo.csv : f1`（按 `held_out`） | 替换（修复前最好 0.574、最差 0.042） |
| 误报率 BGL 43.4% / DNS 3.0% | `table_lopo.csv : fpr` | 替换（修复前 Firewall 41.8% / DNS 55.8%） |
| 平台内对照 0.886 | `table_main.csv : f1` 均值 | 替换 |

---

## 4. 表 3 / 图 5（第一层过滤器）

| 论文数值 | 复现文件 : 字段 | 状态 |
|---|---|---|
| IF / HBOS / LOF / OCSVM / AE / DBSCAN 的 F1 0.583 / 0.450 / 0.138 / 0.387 / 0.437 / 0.143 | `results/tables/table_filters.csv : f1` | 替换（修复前 IF=0.318，其余不变） |
| 对应精确率、召回率、假正率 | `table_filters.csv : precision/recall/fpr` | 替换 |
| 打分吞吐 1.35e4 / 8.28e4 / 1.91e4 / 1.18e4 / 1.84e5 / 5.67e4 | `table_filters.csv : throughput` | 替换 |
| 口径：固定 10% 预算子集、种子 0 单次运行 | 表注已声明；`scripts/run_reproduction.py : exp_filters` | 新增说明 |
| 结论：IF 由“并非最优”改为“最高 F1，但吞吐低于 HBOS/AE” | 6.4 节第 1 段 | 论断反转 |

---

## 5. 表 4 / 图 8（基线对比）

| 论文数值（5 次均值） | 复现文件 : 字段 | 状态 |
|---|---|---|
| 仅孤立森林 F1 0.578±0.021，吞吐 1.75e4，调用 0 | `table_baselines_summary.csv`（`baseline = Isolation Forest only`） | 替换（修复前 0.339） |
| 纯文本 LLM F1 0.969±0.006，吞吐 6.00e4，估算 16.2，调用 100% | 同上（`Text-only LLM`） | 替换 |
| 文本+结构 LLM F1 0.972±0.006，吞吐 6.11e4，估算 16.2，调用 100% | 同上（`Text+struct LLM`） | 替换 |
| 398 维逻辑回归 F1 0.880±0.004，吞吐 5.58e5 | 同上（`398-d logistic classifier`） | 替换 |
| 默认级联 F1 0.886±0.021，吞吐 1.66e4，估算 156.1，调用 10.29% | 同上（`LLM-LogFusion (cascade)`） | 替换（修复前 0.634） |
| 结构第一层级联 F1 0.942±0.006，吞吐 2.24e4，估算 142.2，调用 11.33% | 同上（`Cascade, struct-only 1st layer`） | 替换 |
| DeepLog 类基线 F1 0.510±0.007，P 0.344，R 0.980，FPR 11.0%，吞吐 8.26e5，调用 0 | `table_deeplog.csv`；`table_baselines_summary.csv`（`DeepLog-style log-key`） | 新增（同协议重实现，仅日志键部分） |
| 外部方法（HOLMES 0.84、FRLog 0.87、LogGPT 0.89、LogLLM 0.90、LogPrompt 0.88、LLM-LADE 0.91） | — | **删除**：不可同协议复现，表注已说明“不可比、不列表”；同协议重实现的 DeepLog 类基线已单独入表 |

原始逐次结果：`table_baselines_5seed.csv`（30 行 = 6 配置 × 5 种子）；DeepLog 类基线为 5 行，见 `table_deeplog.csv`。

### 首层表示消融（表 4 后正文）

| 论文数值 | 复现文件 : 字段 | 状态 |
|---|---|---|
| 结构化 14 维 F1 0.608±0.006、召回 0.921、AUC 0.960±0.002 | `table_first_layer_summary.csv`（`structured (14-d)`） | 保持（修复不作用于 14 维输入） |
| 语义 384 维 F1 0.261±0.032、召回 0.363、AUC 0.796±0.019 | 同上（`semantic (384-d)`） | 保持 |
| 融合 398 维 F1 0.578±0.021、召回 0.823、AUC 0.945±0.003 | 同上（`fused (398-d)`） | 替换（修复前 0.339/0.473/0.827；现已接近结构化） |

---

## 6. 表 5 / 图 6（污染率敏感性）

| 论文数值（种子 0） | 复现文件 : 字段 | 状态 |
|---|---|---|
| 污染率 0.05/0.10/0.15/0.20/0.30 | `results/tables/table_sensitivity.csv : contamination` | 替换（旧网格 0.05~0.30，值全换） |
| 调用比例 4.75/10.40/15.42/20.01/29.69% | `table_sensitivity.csv : call_rate` | 替换 |
| F1 0.672/0.914/0.939/0.945/0.958 | `table_sensitivity.csv : f1` | 替换（修复前 0.549/0.709/0.782/0.825/0.918） |
| 精确率 0.969~0.990、召回率 0.509~0.947 | `table_sensitivity.csv : precision/recall` | 替换 |
| 实测时间 1.72/1.63/1.58/1.67/1.67 s | `table_sensitivity.csv : total_seconds` | 替换 |
| 估算时间 64.7/139.4/205.9/266.9/395.1 s | `table_sensitivity.csv : estimated_seconds` | 替换 |
| 口径：种子 0 单次运行、测试集 21 470 条 | 表注已声明；正文已解释 0.914 与 0.886 的差异来源 | 新增说明 |
| 结论：0.15 后进入平台区（修复前为“无平台期”） | 6.4 节第 3 段 | 论断改写 |

---

## 7. 表 6 / 图 7（跨层信息传递）

| 论文数值（5 次均值±标准差） | 复现文件 : 字段 | 状态 |
|---|---|---|
| A 无传递 0.879±0.023 / 0.946 / 0.821 / 0.28% / 2 209 | `extra_transfer_repeats_summary.csv`（列组 `f1/precision/recall/fpr`，行 `none`） | 替换（修复前 0.634） |
| B 仅异常分数 0.886±0.021 / 0.964 / 0.820 / 0.18% | 同上（`score`） | 替换 |
| C 仅多模态特征 0.881±0.023 / 0.952 / 0.821 / 0.25% | 同上（`features`） | 替换 |
| D 完整传递 0.886±0.021 / 0.963 / 0.820 / 0.19% | 同上（`full`） | 替换 |
| 结论：分数传递带来 +0.007 F1 / +1.8 pp 精确率，追加特征与原始日志无增益；差异约为种子标准差的 1/3，视为弱证据 | 由上述均值/标准差直接比较；正文 6.4 节第 4 段 | 论断改写（修复前“四配置无差异”） |

原始逐次结果：`extra_transfer_repeats.csv`。

---

## 8. 表 8 / 6.6 节（12 阶段归因）

| 论文数值（5 次均值） | 复现文件 : 字段 | 状态 |
|---|---|---|
| 逐阶段 P/R/F1（12 行） | `extra_stage_per_stage_repeats.csv` 按 `stage` 求均值 | 替换（旧值全部作废） |
| 宏平均 0.861/0.863/0.859 | 同上，12 阶段算术平均 | 替换 |
| 加权平均 0.862/0.862/0.860 | `extra_stage_repeats.csv : weighted_precision/weighted_recall/weighted_f1` 均值 | 替换 |
| 加权 F1 标准差 0.013、宏平均 F1 标准差 0.014 | `extra_stage_repeats.csv : weighted_f1/macro_f1` 样本标准差 | 替换 |
| 总体准确率 86.2%±1.3% | `extra_stage_repeats.csv : accuracy` | 替换（旧 85.2%） |
| 每阶段每种子支撑 28~55 条 | `extra_stage_per_stage_repeats.csv : support` 最小/最大 | 替换（删去“固定 43 条”） |
| Exfiltration 0.654、Discovery 0.783、C2 0.802、Impact 0.813 | `extra_stage_per_stage_repeats.csv : f1` | 替换（旧 0.608/0.756/0.836/0.753） |
| Defense Evasion 0.981、Collection 0.926、Execution 0.942 | 同上 | 替换 |

说明：阶段实验的标注对为 2 500 条合成对（2 000 训练 / 500 测试），每个种子重新划分；特征编码器在全部对上拟合，属 12 阶段实验自身的口径，已在 6.6 节说明。

---

## 9. 6.6 节攻击链重建、6.7 节局限、3.6 节阈值

| 论文数值 | 复现文件 : 字段 | 状态 |
|---|---|---|
| 链重建覆盖率 1.000（12/12 阶段） | `results/metrics.json : coverage` | 替换（修复前 0.917） |
| 链完整度 0.910 | `results/metrics.json : chain_completeness` | 替换 |
| 775 条测试链、LCS-F1 0.344、编辑相似度 0.287、边级 F1 0.082 | `extra_chain_summary.json : n_campaigns/lcs_f1/edit_sim/edge_f1`；逐链结果 `extra_chain.csv` | 替换（修复前 0.243/0.197/0.038） |
| 0.5% 异常率 F1 0.659±0.204、P 0.659±0.249、R 0.753±0.255、调用 2.0%~20.3% | `extra_lowrate.csv`（`target_rate=0.005`） | 替换（修复前 0.634±0.173） |
| 0.1% 异常率 F1 0.179±0.150、P 0.150±0.182、R 0.346±0.142、调用 0.4%~1.0% | `extra_lowrate.csv`（`target_rate=0.001`） | 替换（修复前 0.112±0.093） |
| 模板泄漏 75.2% / 异常模板 50.2% / 链重叠 62.3% | `extra_leakage.csv`：`test_sig_overlap`/`test_anomaly_sig_overlap`/`campaign_overlap` 5 次均值 | 不变 |
| 阈值 0.82：保留 92.6%、阶段准确率 0.859 | `extra_threshold.csv`（`threshold=0.82` 行 `coverage/stage_acc`） | 替换（修复前保留 96.1%、准确率 0.841） |
| 阈值 0.50→0.98：保留率 97.9%→82.3%、准确率 0.859~0.861 | `extra_threshold.csv` 全表 | 替换 |
| 解析失败 0 次、阶段幻觉率 11.68% | `extra_reliability.json : parse_failures/stage_hallucination_rate` | 替换（修复前 12.9%；代理后端口径，正文已声明） |

---

## 10. 图 2–图 8 与表/正文一致性核对

| 图 | 数据文件 | 对应表 | 对应正文 | 核对结果 |
|---|---|---|---|---|
| 图 2 数据集构成 | `table_dataset.csv` | 表 1 | 6.1 节 | 42 940 / 2 395 / 5.58% 三处一致 |
| 图 3 各平台 P/R/F1 | `table_per_platform.csv` | — | 6.2 节第 2 段 | 9 平台数值与正文逐一一致；红线 0.89 |
| 图 4 LOPO | `table_lopo.csv` | — | 6.3 节 | 9 平台 F1 与正文一致；图例标注均值 0.57/0.89 |
| 图 5 过滤器 | `table_filters.csv` | 表 3 | 6.4 节第 1 段 | 6 种算法 F1/吞吐一致；对数轴刻度已修正 |
| 图 6 敏感性 | `table_sensitivity.csv` | 表 5 | 6.4 节第 3 段 | 5 个点、估算时间一致 |
| 图 7 跨层传递 | `extra_transfer_repeats_summary.csv` | 表 6 | 6.4 节第 4 段 | 4 配置均值/误差棒一致 |
| 图 8 基线 | `table_baselines_summary.csv` | 表 4 | 6.4 节第 2 段 | 7 配置 F1 与吞吐一致（含 DeepLog 类基线） |

编译核验：`xelatex → bibtex → xelatex → xelatex`，16 页，无 Overfull/Underfull，无未定义引用，`??` 计数为 0。

---

## 11. 删除与不可复现项清单

| 内容 | 旧稿位置 | 处理 | 原因 |
|---|---|---|---|
| 旧语料统计（2 177 条异常、5.07%） | 摘要、表 1、图 2 | 删除，改为 2 395 / 5.58% | 原数据随系统重装丢失 |
| 两名分析师人工标注、Cohen κ=0.94 | 6.1 节 | 删除 | 无标注记录，标签实由生成器给出 |
| 70/15/15 划分 | 6.1 节 | 删除，改为 50/50 分层随机、5 种子 | 复现脚本采用 50/50 |
| LogHub / DARPA TC 公共基准子集评估 | 6.1 节 | 删除 | 本次未在公共基准上运行 |
| 真实 LLM 调用日志（Token 528/542/612/648、输出 142/138/127/108、耗时 307.82/156.78 s） | 6.4 节 | 删除 | 无 API 调用产物 |
| 调用比例 8.90%、F1=0.92、LOPO 0.86、吞吐 139.91 条/s | 摘要、6.2、6.3、6.4 | 删除并替换 | 与复现结果不符 |
| Nyströmformer 消融 0.922 / 0.925 / 0.879 | 6.1 节 | 删除 | 无对应运行产物 |
| 8 种外部方法原论文 F1 并列 | 表 4 | 删除 | 不同数据集、不同协议，不可比 |
| 真实 LLM 的 JSON 解析失败率与阶段幻觉率 | 6.7 节 | 删除真实值，改为代理后端统计并加限定 | 代理后端不产生自由文本 |
| “低异常率下 F1 仍保持 0.78 以上” | 6.2 节 | 删除并改写为否定结论 | 复现中 0.1% 异常率 F1=0.112 |
| SBERT（all-MiniLM-L6-v2）语义编码 | 4.2 节 | 正文保留设计描述，6.1 节声明实际使用 LSA 替代 | 模型权重无法下载 |
| 攻击链覆盖率的“DARPA TC 真值链”表述 | 6.6 节 | 改为合成语料实测 | 无 DARPA 链级产物 |
| 阶段“每阶段固定 43 条测试样本” | 6.6 节 | 删除，改为 28~55 条 | 实际支撑随划分变化 |

---

## 12. 原 1–7 条阻塞项的当前状态

| 序号 | 阻塞项 | 本次能否回答 | 证据 / 仍缺材料 |
|---|---|---|---|
| 1 | 阈值 0.82 的来源与标定 | **部分回答** | `extra_threshold.csv` 给出 0.50~0.98 扫描曲线；但基于代理后端置信度，真实 LLM 置信度分布仍缺，正文已限定 |
| 2 | Prompt 全文、解析失败率、幻觉率 | **部分回答** | 解析失败率 0（代理后端无 JSON 解析环节）；阶段幻觉率 12.9% 按“预测阶段不在该链真实阶段集合内”定义；真实 API 的失败率与幻觉率仍缺 |
| 3 | 攻击链/模板是否跨划分隔离 | **已回答（含隔离重跑）** | `extra_leakage.csv`：75.2% 测试模板、50.2% 异常模板、62.3% 测试链在训练集出现；`review_round1.json` 另给 campaign 隔离划分 F1=0.902±0.040（默认）/0.941±0.018（结构第一层） |
| 4 | 0.5% / 0.1% 低异常率重校准 | **已回答（代理后端）** | `extra_lowrate.csv`；真实在线后端的重校准仍缺 |
| 5 | 6.6 节链重建三类实测指标 | **已回答（合成语料）** | `extra_chain_summary.json`；DARPA TC 真值链仍缺 |
| 6 | 硬件、软件栈、种子、吞吐量范围 | **已回答** | 6.1 节新增完整配置；吞吐口径已写明含/不含项 |
| 7 | 表 6 / 表 8 重复次数与标准差 | **已回答** | 表 6 逐格 ±std；表 8 现给逐阶段 F1±std 与宏/加权标准差（`extra_stage_per_stage_repeats.csv`） |

---

## 13. 仍需作者/外部条件确认的事项

1. 真实 LLM（API）后端上的 Prompt、JSON 解析失败率、阶段幻觉率、Token 与延迟——当前全部为离线代理后端口径。
2. `sentence-transformers` 权重可用后，384 维语义嵌入应重跑；当前语义块单独 AUC=0.796，融合后 AUC 由 0.960 降至 0.945，相关结论可能变化。
3. DARPA TC 链级真值与预测链，用于替换合成语料上的链重建指标。
4. 收稿日期与四位作者简介等元数据，按作者要求保持占位。

---

## 14. 模拟评审补充实验（2026-11-11）

| 实验 | 脚本 | 产物 | 关键数值 | 稿件位置 |
|---|---|---|---|---|
| bootstrap 区间 + McNemar | `scripts/run_review_checks.py` | `results/tables/review_round1.json` | F1 区间半宽 0.012~0.015；结构第一层多判对 64~168 条、反向 4~10 条，$p<10^{-10}$ | §6.2 |
| 攻击链隔离划分 | 同上 | 同上 | 0.902±0.040（默认）/ 0.941±0.018（结构第一层），调用 9.9%~10.4% | §6.2、§6.6(6) |
| 第二层错误注入 | 同上 | 同上 | 5%/10%/20% 错误率 → F1 0.839 / 0.792 / 0.699 | §6.6(5) |
| Nyström 延迟 | 同上 | 同上 | 32/64/128/256 token 加速比 0.49/1.43/1.01/2.77；2 000 条编码 0.27→0.24 s | §3.3 |
| 全量第一层过滤器复核 | `scripts/recheck_filters_full.py` | `results/tables/review_filters_full.csv` | IF 0.578±0.021、AE 0.490±0.033、HBOS 0.440±0.007 | §6.4 |
| LOPO 阈值重标定 | `scripts/review_lopo_recal.py` | `results/tables/review_lopo_recal.csv`、`review_lopo_recal_prior.csv` | BGL 0.153→0.817（FPR 43.4%→0.7%）；DNS 0.065→0.268；90% 分位时 BGL 0.607 | §6.3、§6.6 |

红线数据未被替换或隐藏：0.886 与 0.942 的对比、BGL 0.153 / DNS 0.065、0.1% 异常率 F1 0.179 均在正文原样保留；重标定结果只作为补充证据并列呈现。

---

## 15. 第二轮九条补充实验（2026-11-11）

| 实验 | 脚本 | 产物 | 关键数值 | 稿件位置 |
|---|---|---|---|---|
| 链隔离 vs 随机划分的构成分析 | `scripts/analyze_chain_split.py` | `chain_split_analysis.csv`、`chain_split_composition.csv`、`chain_split_summary.json` | 测试异常 1197~1198 vs 1198~1211；模板重叠 75.2% vs 75.3%；反事实重加权 0.889 vs 0.871（原始 0.902 vs 0.886） | §6.2 第 3 段、摘要 |
| 固定阈值 vs 先验重标定（表 2） | `scripts/review_lopo_recal.py`（复用） | `review_lopo_recalibration.csv`、`review_lopo_recal_prior.csv` | BGL 0.153/0.607/0.746/0.817（FPR 43.4%/4.6%/2.0%/0.7%）；DNS 0.065/0.268/0.293/0.268 | §6.3、表 2 |
| DeepLog 类基线 | `scripts/run_deeplog_baseline.py` | `table_deeplog.csv`、`deeplog_summary.json` | F1 0.510±0.007、P 0.344、R 0.980、FPR 11.0%、吞吐 8.26e5 | §6.4、表 4、图 8 |
| BGL 真实日志试点（外部效度） | `scripts/run_loghub_pilot.py` | `table_loghub_pilot.csv`、`loghub_pilot.json` | 级联 0.493±0.017；结构第一层 0.444±0.034；仅 IF 0.290±0.024；全量调用 0.961±0.021 | §6.5、表 7、摘要、§1 贡献（4） |
| 摘要压缩与代理限定 | — | 稿件正文 | 中文摘要 420→344 字；英文 275→224 词；题名加“离线代理评估” | 题名、摘要、§1 |

数据来源与可用性：BGL_2k 取自 LogHub（`BGL/BGL_2k.log` 与 `BGL_2k.log_structured.csv`，143/2000 异常，7.15%）；HDFS_2k 无标签，未纳入。上述文件与全部结果已收入 `artifact/llm-logfusion-anonymous.zip`。完整 BGL 语料（约 470 万条）需从 Zenodo 下载，本次网络条件未完成，故只报告 2 000 条试点。

红线数据仍原样在场：0.886 / 0.942 / LOPO 0.153、0.065 / 0.1% 异常率 0.179 / 全量 0.972 / 语义块 0.796 / 融合 0.945 / 结构化 0.960。
