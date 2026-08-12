# 官方比赛录像候选清单

本文件整理了当前可用、适合研究用途的官方羽毛球比赛录像来源，优先用于：

- 构建 `metadata/matches.csv`
- 人工筛选主视角稳定的比赛
- 下载或缓存少量先导实验视频

详细链接清单见 [official_replay_candidates.csv](file:///mnt/d/badmintondataprocess/metadata/official_replay_candidates.csv)。

## 推荐优先级

### 1. BWF TV

优先原因：

- 官方来源
- 世界羽联赛事覆盖面广
- 主视角转播较稳定
- 单场比赛和整站 playlist 都比较容易找到

建议优先筛选：

- All England Open
- Malaysia Open
- World Tour Finals
- Thomas & Uber Cup Finals

注意：

- 部分地区会因为转播权出现 geo-block
- 同一 playlist 中可能混有采访、集锦和短视频，需要二次筛选

## 2. Olympics

优先原因：

- 奥运会录像制作标准高
- 重要场次标签和球员信息清晰
- 历史经典比赛多

注意：

- 很多页面标记为 `Exclusive`
- 常需要登录 Olympics 账号
- 有些页面是整场 session，不一定是单场比赛，需要后续切分

## 建议使用策略

### 小规模先导集

建议先选 3 到 5 场：

- 2 场 BWF 单场决赛
- 1 到 2 场 Olympics 金牌赛
- 1 场历史经典比赛

### 正式实验集

建议按以下结构扩展：

- 男单 `MS`
- 女单 `WS`
- 男双 `MD`
- 不同年份
- 不同赛事级别

## 元数据录入建议

把候选清单里的链接人工确认后，再写入正式 `matches.csv`，避免：

- 链接失效
- 地域限制导致不可用
- playlist 被误当成单场比赛
- 轮次和球员信息录错
