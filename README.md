# DASE7506 MP3 — SQL Repair Agent

最终选用方案：`q025_v075_compilerfirst_B`，官方完整测试 **197/240（82.083333%）**，每题必须通过全部三个评分实例。

本仓库只保存该方案的代码依赖、配置、报告、训练来源、推理审计、原始预测和完整评分。源码中的历史 step 文件名是最终控制器的依赖；不包含实验轮次输出、其他候选权重、数据集、基座权重、虚拟环境或凭据。

当前最终提交为 `mp3-197-v2`：补齐课程强制报告内容；算法、模型、原始预测和197分评分均保持不变。

## 提交附件

- [源码及完整附件 ZIP](https://github.com/JINWEI-1031/mp3-dase7506/releases/download/mp3-197-v2/mp3-197-source.zip)
- [仅包含两个文件的适配器 ZIP](https://github.com/JINWEI-1031/mp3-dase7506/releases/download/mp3-197-v2/mp3-197-adapter.zip)
- [完整评分 JSON](outputs/test-score.json)，保留官方输出原文，不裁剪逐题结果。
- [原始预测与工具轨迹](outputs/test.json)
- [报告](report.pdf) · [复现命令](REPRODUCE.md)

## 模型与运行

使用原始 `Qwen/Qwen3-0.6B`，固定 revision `c1899de289a04d12100db370d81485cdf75e47ca`。最终适配器将 C4/C17 的 LoRA 增量按固定模块系数组合：q_proj 的 C4 系数为 0.25，v_proj 为 0.75；所有层、题目和数据库使用相同系数。部署只加载一份 rank-32 LoRA A/B 适配器，不加载两个来源适配器或合并基座。实际参数总数为 600,637,440。

原始结果在 Windows 上以 4 CPU、float32、单进程推理得到。Linux 独立完整复现尚未完成；这与已有的完整官方评分是两件事。原始推理与评分证据见 `evidence/`，新打包入口的检查记录见 `evidence/export-verification/`。

推理离线运行，通过官方工具感知反馈并修复 SQL，每题最多 8 次付费工具调用、6 次模型生成。本次原始运行实际生成 19 次、582 个新 token。详见报告中的复现状态、开发曝光与局限披露。

## 作业提交

到课程项目页选择 `outputs/test-score.json`，填写学号和上述两个 ZIP 链接，准备提交后实际创建生成的 GitHub issue。上传本仓库本身不等于提交课程成绩。本次先供本人审核，尚未创建课程提交 issue。

采用 Codex 辅助开发、调试、核验和整理材料；评分时不调用外部在线模型。课程与 Spider 衍生材料归属见 NOTICE。
