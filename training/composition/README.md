R67仅构造根在quarter评分前预注册的两份固定q/v模块交叉纯LoRA A/B资产：q_C4=.25/v_C4=.75，及q_C4=.75/v_C4=.25，C17系数均为1−C4。系数仅由模块类型q_proj/v_proj决定，跨全部28层、全部题目和DB为常数；无拟合、搜索、per-layer/per-task调整或test ID路由。根预注册元数据、original C4/C17权重/config哈希、配方及脚本在构造前封存。

只读取原C4/C17的A/B矩阵作为权重来源，不使用quarter或test预测/标签生成权重。两source均rank16/alpha32、scale2；新rank32/alpha32、scale1，A=concatRows(A4,A17)，按模块类型令B=concatCols(2λ_module B4,2(1−λ_module)B17)。因此每模块有效ΔW为λ_module ΔW_C4+(1−λ_module)ΔW_C17，无因子平均交叉项。A保留原始FP32位；B的1.5乘法允许正常FP32舍入，分别审计公式存储误差与已存储block代数误差。

每份112个纯A/B tensor，包含28个q模块与28个v模块，4,587,520 adapter参数、部署总600,637,440参数。权重文件18,365,216字节（约17.51MiB），低于512MiB；部署只加载其中一份输出adapter和原始base，不同时加载原C4/C17或两个cross适配器。无embedding、lm_head、bias、modules_to_save、DoRA、SVD、量化、full/base-merged checkpoint或梯度更新。

基座/tokenizer仍为Qwen/Qwen3-0.6B immutable revision c1899de289a04d12100db370d81485cdf75e47ca，base SHAf47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b。当前tokenizer哈希再次核验，源config/AB和既有完整samebase/shape审计绑定，完成后原C4/C17再次复核未改。原base整权重未打开，完整Qwen网络未加载或构造。

权重SHA256：q025_v075=09e900d89768709e35e2060f8706eb42ed7d60c7f8a05515e26602801342256e；q075_v025=53a77080b1609d66e2c1c1c43e3326c5c04bf33722c066f9f424f4a5c9377a02。两config SHA均fc1aac31b4139b53d3f40af9c8172133dc1b4b514c6dd2c647e30731f377c3ad。

每份56实际模块×5向量探针和q、v各一个17→13真实PEFT tiny层验证通过。公式参考FP64容差atol1e-8/rtol1e-6，存储block拼接容差1e-10；各模块lambda全部记录并验证常数作用域。A切片、save/reload均bit-exact；grad disabled、tiny层requires_grad=False，无optimizer/backward。数值正确不保证SQL或任务正确数。

合成/验证1 CPU、IDLE、线程1、16GiB Job硬限、300秒wall硬限、CUDA_VISIBLE_DEVICES空。实际guard12.39秒、峰值RSS756,207,616字节，exit0/Job关闭，自有进程已退出。无完整LLM、模型预测、paid、official score、训练、GPU或test逐题读取。

每份*-adapter-only.zip仅两adapter文件，source-review.zip单独包含来源、配方、代码与审计，低于16MiB。规则边界同本机README94–100及R61补充：纯LoRA A/B无预设rank，base/tokenizer不变，不提交full或base-merged模型。后续R68只静态准备C4当前196控制与两q/v交叉的同R33完整VAL120包，三份360预测全seal/模型退出后需根另批score；当前无实际完整模型/paid/score执行授权。
