# realrobot/archive — NERO 真机线诊断脚本索引

这些是 pi0.5 从 v2 到 v8 再到 b1/b2 的排查过程中写的一次性诊断工具。真机线没有 held-out
验证集,每一个结论都是靠这些脚本 + 真机 A/B 得出的,原样保留以便复核。

「得出的结论」一列只抄脚本 docstring / 顶部注释里**已经写下**的实测结果。绝大多数脚本是在
跑之前写的,只记了假设和判据、没回填结果 —— 这类一律写 `(未记录)`,不做任何推测性补写。

| 脚本 | 干什么 | 得出的结论 |
|---|---|---|
| `analyze_exec_lag.py` | 读 server trace,测"执行层跟不上"假设:(A) 模型 chunk 里到底有没有命令真抓取位姿(下扎 j4=idx3 低 + 腕 j7=idx6 到位),(B) 命令跳变幅度 vs 单次 infer 间隔内臂的实际位移;顺带打印 infer 间隔 median/p90 | (未记录) |
| `b_loss_to_tb.py` | 正则解析 `train_b1.log` 的 `Step N: grad_norm=.., loss=.., param_norm=..` 行,写成 TensorBoard events(tags 与主仓 v8 train.py 一致),让 b1 与 v8 两条 loss 曲线叠在同一张图上比 | b1 训练**并非没记 loss** —— openpi-agilex 用 tqdm_loggable 把 pbar.write 转进 logging,loss 一直在 log 文件里 |
| `brighten_test.py` | 把真机帧的每通道亮度匹配到训练 grasp 帧(ep25 f164)的均值,再喂 v6_gmask/7999,看预测 grip 是否从"开"变"闭" | (未记录;docstring 只给判据 —— 调亮后该闭的帧变闭 = 光照是"病B"主因,加灯即可修不必重训) |
| `chain_tail_residual.py` | 逐关节算链尾残差 `abs(state[infer_k] − chunk[-1] of infer_{k-1})`,即臂执行完上一 chunk 后的实际落点 vs 该 chunk 末 waypoint;并与同段实际位移相除得"残差/实走比"(残差要相对运动幅度看) | (未记录) |
| `check_action_shape.py` | 给 9095 上的 v6 server 发一帧随机 dummy obs,打印返回 actions 的 shape/dtype,钉死 model 原生 horizon 与 action_dim;同时固化 client 契约(224x224x3 uint8 双图 + 32 维 state + prompt) | (未记录) |
| `check_frame0_gripper.py` | 统计训练集每条 episode 第 0 帧的 state[7] / action[7] 夹爪态、最后一帧收尾态、以及全程 grip 范围,验部署 SAFE_HOME grip=0(闭) 与训练起步态是否失配 | (未记录;docstring 只给判据 —— 训练第 0 帧 grip>40(开) 则 t=0 失配坐实,v7 采集起步态要对齐) |
| `check_grip_lead.py` | 数据侧判 `action.grip` 到底是 leader 命令还是 follower state 的拷贝:在夹爪过渡帧(相邻 state.grip 变化 >5mm)上比 action 与 state 的差,并做 lag∈[-2,5] 扫描找误差最小的 lag | (未记录;docstring 只给判据 —— lag=0 则 action 就是当前 state(主从同步,copycat 根源,mask 是对的);lag>0 则 action 领先 state 是真命令,改记 leader 命令可根治) |
| `check_sponge_pos.py` | 量 v7 那 23 条 episode 里海绵的位置散布:每 ep 取抓取前早帧(t_g−90,海绵可见且臂未遮挡),HSV 抠 pink/magenta 取最大连通块质心 (x,y) | (未记录;docstring 只给判据 —— 散布小 = 仍是"死一个点",30Hz 也救不了位置泛化,需 v7.1 补数据) |
| `check_v4_plateau.py` | 读 v4 的 TensorBoard events,把最后一步 loss 与 1000 step 前(log_interval=50,即回退 20 个 scalar)比,降幅 <10% 判 `PLATEAU=YES` 并 exit 0,否则 `PLATEAU=NO_keep_training` exit 1 | (未记录;脚本本身就是判据,不产出结论) |
| `coord_check.py` | 真机线核心诊断之一:从 server trace 统计同一帧 j7(腕,idx6)<45 且 j4(下扎,idx3)<71 同时成立的帧数(= 完整 grasp pose),另出 Pearson r(j7,j4)、腕到位时 j4 的中位/最低、手扎深时 j7 的中位/最低,以及最接近 grasp 的单帧 | docstring 记录待复核的 windows 结果:**6k/15k 两个 ckpt 的"同时成立帧数"都 = 0**,两轴反相关,夹爪闭在半空;训练 grasp 值 j7≈37.8 / j4≈70.9,判据 box 为 j7<45 且 j4<71 |
| `counterfactual_grip.py` | causal confusion 的反事实判决:取视觉明确的训练帧(approach 135/140/145、grasp 168/171/174),把输入 state.grip 篡改成 {5,35,65},看 v5/4000 预测的 grip0 跟注入值走还是跟视觉走 | (未记录;docstring 只给判据 —— pred≈注入值 = 复读 state 的 causal confusion 铁证,pred≈视觉该有的值(grasp 帧≈13) = vision driven 要另寻因) |
| `counterfactual_grip2.py` | 同一反事实协议但把 config + ckpt 参数化,用于验 `mask_gripper_state` 有没有真的破掉 chunk[0] 复读 state.grip;运行时会打印所用 config 的 mask_gripper_state 值 | 部分记录:**v5(未 mask)对照输出随注入 1:1 变,spread≈55 = 仍在复读**;v6(masked) 一侧 docstring 写的是期望("应完全不影响,spread→0"),没有回填实测值 |
| `dump_grip_b.py` | dump b1 每条 episode 的 `action[7]` 原始 mm 轨迹形状,按相对峰值的 25%/75% 带定位"真抓取帧 = 张开之后首次掉回低带",并记其相位百分比,用来复核 phase-shortcut 结论的方向 | **夹爪约定被纠正:0mm = 闭合,约 76mm = 张开**,与此前记反了(windows 指出);该纠正连带影响所有基于 grip 阈值的判读方向 |
| `grasp_mode_classify.py` | grasp 打法分类 + 单模筛选:每 ep 取抓取瞬间(gripper 闭合帧)的 state,以 d = j4 − j7 为判别量,对 d 做数据驱动的 2 簇分裂(取最大间隙,不写死阈值,v7/v8 通用);给定 keep_mode 时打印该模式要保留的 episode_index 列表与筛后 r/std | (未记录;脚本沿用 coord_check 口径的同帧合成判据 TH4=71 / TH7=45) |
| `grasp_pose_b.py` | 遍历 b1 每条 episode,取"真闭"帧(action.grip 张开后首次掉进低带)那一刻 follower state 的 j1..j7,打印 7 轴 mean/std,给出抓取构型的中心与散布 | (未记录;该脚本自身无注释。下游 `grasp_window_b.py` 的 docstring 记其产出被用作构型阈值 j4≈93±8 与 j7≈23±8) |
| `grasp_window_b.py` | b1 训练集 grasp-window 统计(windows 规格):open>50mm / close<15mm,每 ep 取第一次 open→close 跃变(带 `min(grip[t:t+5])<15` 防噪)前后 ±10 帧(30fps 约 ±0.33s),池化输出 6 通道 mean/std/min/max、构型达标帧比例、以及完全没有跃变的 ep 数;检测信号用 action(leader,与 rollout 命令端可比),构型检查用 state(follower 实际臂位) | (未记录;docstring 只记了输入侧 —— 约定 0mm=闭/76mm=张,构型阈值 j4≈93±8 与 j7≈23±8 取自 `grasp_pose_b.py`) |
| `image_diff.py` | 量化真机帧与训练 grasp 帧(ep25 f164)的视觉距离,拆成亮度/色偏/直方图/结构几项,确诊"病B"是不是多重成因 | (未记录) |
| `inject_alljoints_b.py` | 对 j∈0..6 每个关节**单独**往 state[j] 注入 Δ∈{−10,0,10,15,20}°(图像保持正确,锚在真 approach@35% 与 grasp@48% 帧),量 chunk[0] 第 j 维随 Δ 的 slope;用于判 j7 是不是唯一失控轴,给 b2 的 single-vs-multi-axis recovery 定调 | (未记录;判据为 slope≈1 = 该轴零视觉纠正的正反馈(闭环会漂),slope≈0 = 靠图像锚定拉回) |
| `inject_horizon_b.py` | 与 `inject_alljoints_b.py` 同协议,但记录 chunk 全 10 步而非只 chunk[0],回答开环执行的后段 horizon 会不会重新用上视觉锚定 | (未记录;判据为 全 k 的 slope≈1 则整段 horizon 是纯 dead-reckoning、receding-horizon 也救不了;后段 slope 下降则缩短开环执行步数是便宜的缓解) |
| `inject_j4_test.py` | 拿真机 grasp_frames 的 NPZ,做 (a) state.j4 原样 vs (b) state.j4 改成 102 的对照,看 v6_gmask/7999 预测的 grip 闭不闭 | (未记录;判据为 (b) 闭 = 闭爪被关节 pose gate 住,靠 descend 辅助压到 102 就能抓、不必重采数据;(b) 不闭 = 必须换 diverse 数据。docstring 另记 v6 只 mask 了夹爪 state(dim7→0),j4(dim3) 不 mask,所以改 j4 确实会影响模型) |
| `inject_j7_b.py` | b1 版 j7 闭环正反馈硬测,与 A/v8 的 `inject_j7_feedback.py` 完全同协议、只换 CFG 与 DS:喂 in-dist 的 grasp/approach 帧,把 state 的 j7 注入 Δ∈{−10,0,+10,+15,+20}°(图像保持正确),量 chunk[0] 的绝对 j7 随 Δ 的斜率,用于和 A 对账"B 是否结构上治好了腕漂" | (未记录;判据为 斜率≈0 = 靠图像锚定把 j7 拉回流形的负反馈(真治好了),斜率≈1 = 跟随 state 不纠正的正反馈漂(与 A 一样没治好)) |
| `inject_j7_feedback.py` | 上一条的 A/v8 版本(CFG 与 DS 都是 v8),验 windows 提的 j7 闭环正反馈假设 | (未记录;docstring 记录的机理前提是 openpi 的 `absolute_out = predicted_delta(image,state) + injected_state`,故 斜率≈0 意味着真实漂来自图像、v9 的 crop/图像稳定才是解,斜率≈1 意味着病在过度信 state) |
| `measure_j7_drift.py` | 从 server_trace 量腕 j7(state_8[6])的闭环漂移:按 home-reset(j7 回升 >88 且此前已下扎到 <70)切 run,报每个 run 的下扎最低 j7 与卡住的吸引子位置 | **6k 基线实测:下扎最低约 51,卡住的吸引子约 80**,而训练抓取值是 37.8(即离目标差约 13°、并被吸引到 80 附近) |
| `openloop_audit.py` | 开环 teacher-forcing audit:喂训练原帧 obs(默认 ep 25/50)给 ckpt(默认 v5/4000),看模型预测的 grip 闭不闭,用来一刀切开"训练问题"与"部署/OOD 问题";曲线落 JSON | (未记录;脚本内判据为 预测 chunk0 grip 最小值 ≤20 判"闭、学会了",≤35 判"半闭",>35 判"分布内也不闭") |
| `openloop_audit_realframe.py` | 上一条的真机帧对照版:把真机失败 run 的逐帧 NPZ(图 + state)喂给 ckpt(默认 v6_gmask/7999),看预测 grip 闭不闭;NPZ 的 key 名按 high/wrist/state 子串自动探测 | (未记录;docstring 给出的判据是"训练帧→闭(13mm) 且 真机帧→张着不闭(>40) 即直接坐实 OOD",本次实测值没有回填) |
| `probe_b1_descent.py` | 探测 b1-3k 的"下扎驱动":挑 state.j4 低(approach、未下扎)的帧、以及 j4 在 40–60 之间(匹配在线卡住值)的帧喂进去,看模型 chunk 会不会命令 j4 上升到约 93 | docstring 记录的在线实测现象:**b1 在线 j4 卡在 46**(该脚本就是为区分"下扎驱动没问题、在线失败是闭环/OOD" 与 "模型根本没学会从 approach 发起下扎" 而写) |
| `serve.sh` | v1(`pi05_nero_pick_pink_sponge`)/2999 的推理 server,端口 9095,GPU1,trace 落 `server_trace.jsonl` | (未记录) |
| `serve_v2_1000.sh` | v2/1000 的推理 server,端口 9095,GPU3 | (未记录) |
| `serve_v2_3000.sh` | v2/3000 的推理 server,端口 9095,GPU3 | (未记录) |
| `serve_v2_4999.sh` | v2/4999 的推理 server,端口 9095,GPU3 | (未记录) |
| `serve_v2_4999_smoke.sh` | v2/4999 的冒烟测试 server,端口 9095,GPU0,trace 独立落 `server_trace_smoke.jsonl` 以免污染正式 trace | (未记录) |
| `serve_v3_1000.sh` | v3/1000 的推理 server,端口 9095,GPU0 | (未记录) |
| `serve_v3_4999.sh` | v3/4999 的推理 server,端口 9095,GPU0 | (未记录) |
| `serve_v4_4999.sh` | v4/4999 的推理 server,端口 9095;实际 `CUDA_VISIBLE_DEVICES=3`(顶部注释写的是 "GPU 0",与变量不一致,原样保留) | (未记录) |
| `serve_v500.sh` | v1/500 的推理 server,端口 9095,GPU1;显式 `unset OPENPI_TRACE`,即这一版**不落 trace** | (未记录) |
| `serve_v5_4000.sh` | v5/4000 的推理 server,端口 9095,GPU3,用于真机测"chunk 尾段 trick"(client 取 `chunk[-1][7]` 当夹爪命令) | (未记录;注释记录了两点前提 —— ckpt_4000 未 mask、按原样 serve,且该 trick 纯在 client 侧,server 不改) |
| `serve_v6_7999.sh` | v6_gmask/7999 的推理 server,端口 9095,GPU2,即 mask 版上真机 | (未记录;注释把这一轮定位为"病A 修复后验病B",即复读 state.grip 那条已由 mask 处理掉) |
| `serve_v7_15k.sh` | v7/14999(15k 末步)的推理 server,端口 9099,GPU2,独立 trace,作为真机终极候选去量腕 j7 漂移 | 注释记录的对照基线:**6k 是 51/80**(下扎最低 51 / 吸引子 80,与 `measure_j7_drift.py` 的 6k 基线一致),本 ckpt 自身的结果未记录 |
| `serve_v7_3000.sh` | v7/3000 的推理 server,端口 9095,GPU3,`MEM_FRACTION=0.3`;早期 ckpt 上真机试手感,主要看夹爪会不会 commit | 注释记录的实测:**该 ckpt 的 loss 约 0.006 且仍在下降,部署帧率约 20Hz**;config 侧记录 v7 = mask off + 官方 recipe + 数据已修 copycat |
| `serve_v7_6000.sh` | v7/6000 的推理 server,端口 9096(3k 的 9095 保留不动),GPU3 与 3k server 并存(各 MEM 0.3),独立 trace | 注释记录的部署决定:**异步预取已弃用,回到默认同步 broker** |
| `serve_v7_9000.sh` | v7/9000 的推理 server,端口 9097(3k 的 9095、6k 的 9096 都留着),独立 trace;实际 `CUDA_VISIBLE_DEVICES=2`(注释写的是 "GPU3 三 server 并存",与变量不一致,原样保留) | 注释记录的对照基线:**6k 腕漂卡在 83**,本轮就是验腕漂移会不会随 step 改善 |
| `sharpen_gripper.py` | Fix A:段式锐化 lerobot 数据集的 gripper 通道开合过渡 —— 以 25mm 为界切 open/closed 段,每段取 median 拉平,段间过渡压成 3 帧线性 RAMP,短于 4 帧的段并入邻段去噪;保留海绵厚度对应的非零闭合值,只动 dim7、关节 0–6 完全不碰,并剔除没有干净抓取的坏 ep | (未记录;docstring 注明当时只跑 `--dry-run` 在源数据上验证、没有落盘) |
| `test_ref_loopback.py` | 真机回环验收(windows 协议):喂一组已知单位的 state(j1=−2.86 deg、grasp 量级的 j4≈70 / j7≈38、夹爪 40mm 半开)加黑图给指定端口,验 server 端的单位转换层是否透明 | (未记录;通过判据为 返回 chunk 的关节量级必须是 deg(几十到上百,而非 rad 的 ±3),夹爪必须是 mm(0–76,而非归一化的 [0,1])) |
| `tf_eval_descent.py` | 离线 teacher-forcing 的主动下降段 j4 跟踪:每 ep 挑 j4 下降最快的若干帧喂真 obs 给 server,比模型命令的 j4-delta(pred−state)与训练 delta(action−state) | 本脚本结果未记录;docstring 记下了改换指标的理由 —— **grasp 帧的 j4 近乎静止,贴合度判别力不够**,真机"下扎浅"的核心在 approach 的主动下降段 |
| `tf_eval_grasp.py` | 离线开环 teacher-forcing:把训练集帧的原生 obs(cam_high + cam_wrist 480x640 + 16 维 state)逐帧喂给 v7 server,比模型 commanded 与训练 action,同时诊断下扎深度和闭爪信号(approach 开爪期会不会乱闭、grasp 会不会真闭、是否在复读 state.grip) | docstring 记录的实测差距:**commanded j4@grasp 约 37,而训练值约 93**,该脚本即用于看它随 step 收不收敛 |
| `tf_grip_b.py` | b1 离线 grasp-close 真测:锚到真抓取帧喂真 demo 帧,看 chunk 的夹爪输出是闭(趋于 0)还是张(趋于高 mm),另喂 approach-open 帧做对照,以区分"会闭 = 真机失败是闭环漂离流形" 与 "不会闭 = 根本没学会 grasp-close" | (未记录;docstring 只记了输入侧 —— 夹爪约定修正为 0mm=闭/高 mm=张,真抓取帧在约 48% 相位、approach-open 对照取约 35% 相位) |
| `tf_offline.py` | 离线 teacher-forcing(直接加载 ckpt,不连 server),量任意 v7 ckpt 两件事:A 抓取帧的 j4 贴合度 + 夹爪是否领先 state(即非 copycat);B 下降段的"模型命令下降幅度 / 训练下降幅度"= 下降比(判欠训的核心量) | (未记录) |
| `tf_offline_b1.py` | 离线 teacher-forcing 的 b1 版:喂 b1 训练真 obs 逐帧,看 grasp 帧模型能否合成 j4→96 | docstring 记录的在线实测现象:**b1 在线 j4=46**,判读规则是 离线能合成到约 96 则在线失败属部署/欠训差距,离线也合成不了则 3k 还没学会 grasp 协调。⚠ 本文件第 3 个常量行的注释写 "B 夹爪: 0mm=open/78mm=closed",与 `dump_grip_b.py` / `tf_grip_b.py` / `grasp_window_b.py` 后来纠正的 "0mm=闭/76mm=张" **方向相反**,原样保留以便复核受影响的判读 |
| `tf_offline_v8.py` | 离线 teacher-forcing 的 v8 版,核心新增指标是与 windows 在线 `_analyze_smoke_traj` 对齐的**复合 grasp 位姿合成度** —— 模型输出的 chunk 里有没有同一帧同时满足 j4<71 且 j7<45(真下扎 + 转腕到位) | docstring 记录的对照基线:**v7 在这个指标上 = 0**(从未合成过同帧 j4<71 且 j7<45)。判读规则:离线能合成而在线不能 = 闭环 OOD 漂(v8 覆盖不够);离线在线都能 = 治好;都不能 = 数据或模型容量问题 |
| `tf_probe_close.py` | 离线探针:喂训练集中"张爪后即将闭爪"的 obs,看模型 chunk 会不会命令闭爪(grip 绝对维下穿 50)并伴随肘推(j2),以判别真机"张爪后卡住不闭"是腕 OOD/闭环所致,还是闭合信号本身没学好 | (未记录) |
| `tf_probe_j7.py` | 离线探针:喂训练 approach 帧(从起始到张爪峰、每 25 帧采一次),看 6k 模型命令的腕 j7 本身是否带偏(系统性往高 roll 命令),以区分模型 bias 与纯闭环累积 | (未记录;判据为 命令的 j7 运动方向≈训练 则模型在分布内不带偏、真机漂来自闭环累积或图像 OOD;命令系统性高于训练 则是模型 bias) |
| `v4_post_train_auto.sh` | v4 训练后的无人值守流水线(archive 里唯一一个不是诊断工具的脚本):轮询等 `v4/4999` ckpt 落盘 → 再等 train 进程退出 → 调 `check_v4_plateau.py` 判最后 1000 step 的 loss 降幅是否 <10%(= plateau)→ 判 YES 就起 `serve_v4_4999.sh` 并轮询日志等 server listening;判 NO 就以 `--num-train-steps 15000 --resume` 自动续训到 15k(脚本注释说明 openpi 不支持跨 config 任意 resume,所以只能同名 config 续跑) | (未记录;脚本是流程编排,不产出测量结论,其中的 plateau 阈值 —— 最后 1k step loss 降幅 <10% 视为收敛不再续训 —— 是判据而非结果。公开版摘掉了两行内部通知 hook,摘除处留有注释,plateau 判定与续训逻辑一字未动) |
| `v7_cx_cond.py` | 在 v7 上独立复验 windows 的 conditioned-multimodal 发现:顶视海绵的水平位置 cx(从 cam_high 早期帧抠 magenta 得到)能不能预测 grasp 打法(j4 下扎深度) | 本脚本在 v7 上的复验结果未记录;docstring 记下了待复验的 windows v8 结果:**r(cx, j4) = −0.93,r(cx, j7) = +0.66** |
| `v7_grasp_coherence.py` | 对照 v7 训练集自身的 grasp-moment 合成度:抓取瞬间 j4 与 j7 是否已经双模/反相关,用来判病根一直在数据、还是 v8 才引入的新双模;grasp 帧定义为首次主动张开(>0.6·max)之后首次持续闭合(<0.35·max 连续 ≥8 帧),阈值沿用 coord_check 口径 TH4=71 / TH7=45 | 本脚本在 v7 上的结果未记录;docstring 记下了作为对照的 windows v8 audit 结果:**r(j4, j7) = −0.63**。另记 v7 数据的夹爪极性为 open≈99(高)/ close≈13(低) |
| `verify_image_axis.py` | 严谨实查图像轴(czj 要求不许推断、只打印真实 tensor):LeRobot 返回的 image shape/dtype/range → 过 `_nero_parse_image` 后的 shape → 过 `resize_with_pad(224,224)` 后的 shape | (未记录;判据为 480x640 原图经 `_nero_parse_image` 后必须是 HWC 的 (480,640,3),不能把 H/W 颠倒成 (640,480,3)) |
| `verify_v7_config.py` | 不碰数据集、只构造 TrainConfig 对象,打印 v7 的 name / pi05 / action_horizon / action_dim / mask_gripper_state / ema_decay / num_train_steps / batch_size / save_interval / keep_period / repo_id / weight_loader | (未记录;脚本内写死的期望值是 mask_gripper_state 应为 False(即去掉 mask)、ema_decay 应为 0.99) |
| `warm_ports.py` | 预热并诊断 WebSocket infer 通路:对给定端口(默认 9097 与 9099)发一帧全零 dummy obs(NERO 8 维 state + 双 cam 480x640x3),打印返回 chunk 的 shape 与耗时,失败则打印异常类型 | (未记录) |
| `watch_forceclose.py` | 实时 tail v4 的 trace,在"强制闭爪"测试中跟踪夹爪与关节的反应:判定强制闭爪态为 state.grip<8 但 chunk[0] 的 grip>30(爪已闭而模型还在命令开),再看模型会不会 REACT(chunk 末帧相对当前 state 的关节偏移 >5°,说明它看见了抓取)还是继续 hover | (未记录) |
| `watch_grasp.py` | 实时 tail v4 的 trace,只在与抓取相关时输出:对每个真实帧(非全零 state)跟踪 state.grip 与 chunk 的夹爪轨迹,在模型命令闭合(chunk 最小 grip <20mm)或 state 正在闭合时打印,另有心跳行 | (未记录) |

## 关于路径

原始脚本里的绝对路径已替换成环境变量,逻辑一行未改。Shell 脚本用 `${NERO_ROOT}` / `${NERO_DATA_ROOT}` /
`${NERO_CKPT}`(顶部有 guard,未 `source env.sh` 会直接报错退出);Python 脚本用
`os.environ.get("NERO_...", ...)`。其中 `NERO_DATA_ROOT` 是**数据集根目录**(后面接
`pick_pink_sponge_v2_clean` / `_v3` / `_v7` / `_v8` / `_b1`),保留数据集名才能看出每个诊断当初跑在哪份数据上。

一处保真度说明:v 系列(v2_clean/v3/v7/v8)与 b 系列(b1)在原机器上其实位于**两个不同的根目录**
(前者是当时 `HF_LEROBOT_HOME` 指向的数据集目录,后者是默认的 HF LeRobot 缓存),这里统一收敛到了
`NERO_DATA_ROOT` 一个变量下 —— 两者角色相同(都是 LeRobot 数据集根),但若要严格重放某个历史脚本,
需要自行确认对应数据集摆在这个根目录下。
