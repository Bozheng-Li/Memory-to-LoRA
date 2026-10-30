# MAB 数据怎么读

MAB 的基本单位是**一段长历史＋这段历史对应的一组问题与答案**。一行通常包含很多题；四个 Parquet 按测评能力分区，不是 train/dev/test。Parquet 是供程序读取的二进制表格，直接打开不会像普通文本一样清楚。

下面数量来自本地固定版本 7ea066982b140a19337e17e60d45d4076e042faf 的实际读取；本次没有运行模型测评。

## 1. 四个文件各测什么

| 文件/缩写 | 主要内容和问题 | 历史行数 | 题数 |
| --- | --- | ---: | ---: |
| Conflict_Resolution / CR | 编号事实中存在旧值与新值，问更新后的属性或关系 | 8 | 800 |
| Accurate_Retrieval / AR | 长文档、小说事件和多会话历史中的信息检索 | 22 | 2000 |
| Test_Time_Learning / TTL | 读分类示例后给新输入分类，或读历史对话后推荐电影 | 6 | 700 |
| Long_Range_Understanding / LRU | 长篇故事摘要、侦探故事推理 | 110 | 171 |
| 合计 | 四种能力混合，不能按一个同质任务理解 | 146 | 3671 |

AR 的 22 行由 2 行 RULER QA、15 行 EventQA 和 5 行 LongMemEval 派生历史组成；分别有 200、1500、300 题。TTL 是 5 行分类数据各 100 题，加 1 行电影推荐数据 200 题。LRU 是 100 行摘要各 1 题，加 10 行侦探故事共 71 题。行数多不代表独立来源多，也不代表占最终评分的权重更大。

## 2. 一行包含四个字段

| 字段 | 应怎样理解 |
| --- | --- |
| context | 先给 Agent 阅读的整段历史，一个长字符串；运行时再按顺序切块写入记忆 |
| questions | 历史读完后要回答的问题列表 |
| answers | 与问题按位置对应的标准答案列表；每题也是一个列表，可能有多个可接受答案 |
| metadata | 子集来源、题目 ID，部分任务还有时间、示例或答案相关标记 |

例如 questions[15] 对应 answers[15]，并通过 metadata.qa_pair_ids[15] 标识。训练、验证、测试划分不能从四个分区名称直接推出；尤其不能把同一历史的 100 道题当成 100 条独立历史。

## 3. 我们先用的 SH 6K 究竟是什么

**一段包含 455 条编号事实的英文历史，配 100 道单跳问题。** 在 Conflict_Resolution.parquet 的第 4 行（从 0 开始），metadata.source 是 factconsolidation_sh_6k。按 source 筛选比依赖行号稳定。

下面是原数据中的真实例子，序号没有改动：

~~~text
41. Germany is located in the continent of Europe.
...
435. Germany is located in the continent of Africa.

Question: Which continent is Germany located in?
Gold answer: Africa
~~~

意思是：旧记录说德国在欧洲；更晚的记录把它改为非洲。这道题按数据集规则应答 Africa，不能用地理常识覆盖知识池。**这是刻意构造的反事实更新测试，不是在宣称现实中的德国位于非洲。**

官方规则是“同一事实发生冲突时，编号更大的记录更新”，见 [官方模板快照](../research/benchmark_sources/mab_code/utils/templates.py)。这个规则在 query 模板中，原始 context 本身主要是编号事实；正式复现需明确规则何时给 writer/reader，不能改变协议后仍称完全相同的设置。

SH = single-hop，直接问一个属性；MH = multi-hop，需要连接多条关系。例如 MH 6K 会问“Igor of Kiev 的配偶死于哪里”，需要连接配偶是 Olga 与 Olga 更新后的死亡地点 Rodez。

## 4. CR 为什么有 8 行

| 长度设置名 | 每份 context 的编号事实数 | 本地 Qwen tokenizer 实测 token | SH 行号 / MH 行号 | 每种题组 |
| --- | ---: | ---: | --- | ---: |
| 6K | 455 | 6888 | 4 / 0 | 各 100 题 |
| 32K | 2310 | 37476 | 5 / 1 | 各 100 题 |
| 64K | 4580 | 75719 | 6 / 2 | 各 100 题 |
| 262K | 18332 | 317927 | 7 / 3 | 各 100 题 |

同一长度的 SH/MH 共用完全相同的 context，只换题组，所以 8 行只有 4 份不同的历史文本。不同长度不是直接的前缀扩展，题组也不同；不能当作“固定 100 题，只增加噪声”的严格配对实验。6K 是设置名称，不是事实条数，也不等于本地 tokenizer 恰好生成 6000 token；以上计数不含任务提示和特殊 token。

## 5. 怎样接入我们的记忆流程

~~~mermaid
flowchart LR
    A[一整段 context] --> B[按顺序切成事件或文本块]
    B --> C[Agent 管理局部与全局记忆]
    C --> D[历史读完后固定记忆]
    D --> E[逐题回答 questions]
    E --> F[评价器使用 answers 评分]
~~~

首轮先测 SH 6K，再测 SH 32K。writer 不预先看到测试问题或答案，也不整体读取含答案标记的 metadata；各题从同一历史记忆快照开始，不把其它题的答案写回。这能检查新事实生效、旧值干扰、退出后保留及后续 QA；真实工具任务的执行收益还需要另外验证。

可以直接打开 [455 条原始事实](../resources/data/mab/preview/SH6K_context.txt) 和 [100 道题及答案](../resources/data/mab/preview/SH6K_questions.tsv)，无需阅读 Parquet。两份导出已逐项核对，答案表仅供人查看和评价器使用。

文件清单与离线读取方法见 [resources/data/mab/README.md](../resources/data/mab/README.md)；首轮入口见 [first_eval.json](../resources/data/mab/first_eval.json)；完整子集分布见 [data_summary.json](../resources/data/mab/data_summary.json)，本次结构核查见 [mab_structure_20261029.json](../research/benchmark_sources/data_audit/mab_structure_20261029.json)。
