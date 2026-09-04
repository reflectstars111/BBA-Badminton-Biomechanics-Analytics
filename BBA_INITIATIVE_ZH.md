# 让每一段运动视频，都能真正帮助人进步

## BBA 通用人体运动分析平台倡议书

今天，几乎每个人都能用手机记录运动。

但视频拍完以后，我们通常只能凭感觉反复观看。真正的动作捕捉和生物力学分析依然需要昂贵设备、专业场地和复杂软件，普通运动者、基层教练和小型研究团队很难使用。

**我们想改变这件事。**

> 上传一段普通运动视频，让系统自动找到有效动作，理解人体如何移动，并生成看得懂、能比较、可复核的运动报告。

![BBA 实际羽毛球分析效果](https://raw.githubusercontent.com/reflectstars111/BBA-Badminton-Biomechanics-Analytics/main/assets/readme/analysis-overhead-china2018.png)

## 我们已经走出了第一步

BBA（Badminton Biomechanics Analytics）已经可以把一段包含采访、回放和切镜头的羽毛球视频，自动转化为：

- 清洗后的有效比赛回合；
- 球网两侧球员的骨骼与位置；
- 羽毛球飞行轨迹；
- 标准球场上的移动路线；
- 逐回合与全场数据；
- 一段可以直接观看的分析视频；
- 一套可以继续研究和复核的原始数据。

用户通过浏览器上传视频、选择视角、确认场地，就可以启动完整流程。分析默认在本地运行，视频不必交给第三方云服务。

**这不是一张概念图，也不只是给视频画几个框。它已经是一条能够实际运行的一键分析管线。**

## 但羽毛球只是起点

我们希望在现有成果上建立一个**通用人体运动分析平台**。

用户首次使用时，可以通过身高、正侧面视频和可选的 360° 全身视频建立个人身体档案。以后分析单目视频时，系统不再只套用“平均人体”，而会结合这个人的身体比例、视频骨骼、动作连续性、相机与地面关系以及人体关节约束，形成更稳定的个体化运动数据。

在同一个平台核心上，不同领域可以加载自己的运动知识包：

```text
一个通用人体运动核心
        |
        +-- 羽毛球：击球、挥拍、步法与回位
        +-- 健身：深蹲、弓步、俯卧撑与动作稳定性
        +-- 跑步：步频、触地、周期和左右差异
        +-- 其他运动：由教练、研究者和开发者继续扩展
```

我们选择先做好一个动作、一个场景，再逐步扩展，而不是急着宣称“什么都能分析”。下一项完整验证将从自重深蹲开始。

## 我们希望抵达的地方

未来，一位普通运动者只需要架起手机：

1. 录下自己的训练；
2. 选择正在进行的运动；
3. 获得带证据和可信度的动作分析；
4. 与自己的历史表现比较；
5. 把结果交给教练、研究者，或者留给下一次训练。

专业运动分析不应该永远只属于少数实验室和职业队伍。

**如果普通视频能够被可靠地理解，它就不再只是一段录像，而可以成为每个人认识自己、改进动作和记录进步的工具。**

---

- 项目仓库：[BBA · Badminton Biomechanics Analytics](https://github.com/reflectstars111/BBA-Badminton-Biomechanics-Analytics)
- 已有成果与使用方法：[README](https://github.com/reflectstars111/BBA-Badminton-Biomechanics-Analytics/blob/main/README.md)
- 更完整的项目介绍：[PROJECT_BRIEF_ZH.md](https://github.com/reflectstars111/BBA-Badminton-Biomechanics-Analytics/blob/main/PROJECT_BRIEF_ZH.md)
- 详细技术路线：[通用人体运动分析平台开发计划](https://github.com/reflectstars111/BBA-Badminton-Biomechanics-Analytics/blob/main/badmintondataprocess/docs/general_motion_platform_development_plan.md)

欢迎通过 GitHub Issues 留下想法，也欢迎把这份介绍转给可能感兴趣的人。
