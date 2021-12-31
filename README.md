# Mahjong_Game_RLCard_RL
基于RLCard平台的麻将mahjong博弈游戏代码，包括基于规则和基于Dueling DQN的Agent模型。
## 说明文档
### 运行环境
+ **语言**：Python3.8.11
+ **所需第三方库**：根目录下的requirements.txt包含了该环境中所有的包
### 目录结构
+ **Agent**：存放设计的Agent模型，其中HelperX为我们设计的基于规则的模型，Dueling DQN为我们设计的基于强化学习的模型
+ **experiments**：RLCard在运行过程总自动生成的结果
+ **Logger**：RLCard在训练过程中自动生成的测试结果日志
+ **Test**：测试文件夹，存放测试各种模型效果的文件
+ **TrainedModel**：存放训练好的各种模型，包括RLCard子代的DQN、NFSP、加了我们设计的两种优化的DQN以及我们设计的Dueling DQN模型，供Test文件夹下的测试文件加载测试
+ **Utils**：工具文件夹，用于存放工具包，包括计算各种层次的行为，奖励塑造等其他函数
+ **RLCardDQN.py**：RLCard官方的训练代码
+ **mainCode.py**：运行就会开始训练我们设计的Dueling DQN模型
#### 提示：不建议直接运行MainCode.py，需要查看效果可以直接运行**Test**文件夹下的TestXX.py，XX可以替换为各个模型的名字，运行即可查看各个模型的效果。