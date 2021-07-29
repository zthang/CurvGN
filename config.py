class Config():
    def __init__(self):
        self.is_train = True                   # 是否是在训练
        self.times = 3                         # 实验次数
        self.epoch_num = 200                   # 每次实验进行的轮数
        self.wait_total = 100                  # early stop所需轮数
        self.d_names = ['Cora']                # 实验所用数据集
        self.learning_rate = 0.005             # 学习率
        self.gamma1 = 2e-6                     # Reg1权重
        self.gamma2 = 3e-7                     # Reg2权重
        self.leaky_relu_negative_slope = 0.1   # leaky relu 负斜率
        self.loss_mode = 1                     # 0:cross_entropy 1:cross_entropy + gamma1*Reg1 + gamma2*Reg2
        self.curvature_activate_mode = 5       # 0:全连接层 1:ReLU 2:Leaky ReLU 3:PReLU single value 4:PReLU all channel 5:ELU