import pandas as pd

class Parameters:
    def __init__(self):
        # 定义参数并初始化默认值
        self.c = {}  # Travel cost of workers' trucks from node i to node j
        self.cpm = 0  # Maintenance cost of repairing a shared bike in collection centers
        self.cqm = 0  # Maintenance cost of repairing a shared bike on the road
        self.afa_i_s = {}  # Reward of providing station i with a usable bike in slot s
        self.bta_i_s = {}  # Reward of removing an unusable bike for station i in slot s
        self.Q_v = 0  # Capacity of the worker's truck to load shared bikes
        self.Q_f = 0  # Capacity of the collection center to park shared bikes
        self.t = {}  # Minimum travel time of the arc (i, j)
        self.sigma_q = 0  # Average time of pick up (drop off) an unusable bike
        self.sigma_p = 0  # Average time of repairing an unusable bike on the road
        self.sigma_s = 0  # Time length of each time slot
        self.d_f = {}  # Number of repaired shared bikes in collection center f
        self.h_i = {}  # Number of repaired shared bikes that need to be returned to station i
        self.p_i = {}  # Number of broken shared bikes that must be returned to collection centers for repair at station i
        self.q_i = {}  # Number of broken shared bikes that can be repaired directly on the road at station i
        self.T = 0  # Maximum operating duration of each worker
        self.M = 0  # A large number
        self.V = 0 # Speed of the worker's truck

    def set_default_values(self):
        # 设置默认值（可以根据实际需求调整）
        self.cpm = 2  # Maintenance cost of repairing a shared bike in collection centers
        self.cqm = 5  # Maintenance cost of repairing a shared bike on the road
        self.afa_i_s[(1, 1)] = 5  # Reward for providing station 1 with a usable bike in slot 1
        self.bta_i_s[(1, 1)] = 3  # Reward for removing an unusable bike for station 1 in slot 1
        self.Q_v = 50  # Truck capacity
        self.Q_f = 100  # Collection center capacity
        self.sigma_q = 5  # Average pick up/drop off time
        self.sigma_p = 10  # Average repair time on the road
        self.sigma_s = 60  # Time length of each time slot
        self.d_f = {10: 15, 11: 15, 12: 15}  # Number of repaired bikes in collection center f
        self.h_i = {0: 15, 1: 10, 2: 5, 3: 10, 4: 8, 5: 6, 6: 5, 7: 3, 8: 2, 9:2}  # Number of repaired bikes to return to station i
        self.p_i = {0: 15, 1: 10, 2: 5, 3: 10, 4: 8, 5: 6, 6: 5, 7: 3, 8: 2, 9:2}   # Number of broken bikes to return to collection center at station i
        self.q_i = {0: 15, 1: 10, 2: 5, 3: 10, 4: 8, 5: 6, 6: 5, 7: 3, 8: 2, 9:2}   # Number of broken bikes to repair on the road at station i
        self.T = 8  # Maximum operating duration
        self.M = 1000000  # A large number
        self.V = 10 # Speed of the worker's truck

# 需要单独定义的参数群：c, t, afa_i_s, bta_i_s, d_f, h_i, p_i, q_i

    def load_distance_matrix(self, csv_file_path):
        """
        从CSV文件加载距离矩阵并赋值给self.c和self.t
        使用从0开始的行和列索引，而不是原始节点名称
        CSV文件格式：第一行和第一列为节点名称（将被忽略），其余为节点间的旅行成本
        """
        # 使用pandas读取CSV文件
        df = pd.read_csv(csv_file_path, index_col=0)  # 第一列作为索引
        # 获取矩阵的数值部分（忽略第一行和第一列的标签）
        matrix = df.values  # 转换为numpy数组

        # 使用从0开始的索引赋值给self.c
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                self.c[(i, j)] = matrix[i][j]
                self.t[(i, j)] = matrix[i][j] / self.V

    def load_demand_matrix(self, csv_file_path):
        """
        从CSV文件加载累计需求矩阵并乘以倍率后赋值给self.afa_i_s和self.bta_i_s
        使用从0开始的行和列索引，而不是原始节点名称
        CSV文件格式：第一行和第一列为节点名称（将被忽略），其余为节点间的旅行成本
        """
        # 使用pandas读取CSV文件
        df = pd.read_csv(csv_file_path, index_col=0)  # 第一列作为索引
        # 获取矩阵的数值部分（忽略第一行和第一列的标签）
        matrix = df.values  # 转换为numpy数组

        # 使用从0开始的索引赋值给self.c
        for i in range(len(matrix)):
            for s in range(len(matrix[0])):
                self.afa_i_s[(s, i)] = round(3 * matrix[i][s], 2) # 保留两位小数
                self.bta_i_s[(s, i)] = round(1 * matrix[i][s], 2) # 保留两位小数


    def print_parameters(self):
        # 打印所有参数及其值
        print("Parameters:")
        print(f"c: {self.c}")
        print(f"cpm: {self.cpm}")
        print(f"cmq: {self.cqm}")
        print(f"afa_i_s: {self.afa_i_s}")
        print(f"bta_i_s: {self.bta_i_s}")
        print(f"Q_v: {self.Q_v}")
        print(f"Q_f: {self.Q_f}")
        print(f"t: {self.t}")
        print(f"sigma_q: {self.sigma_q}")
        print(f"sigma_p: {self.sigma_p}")
        print(f"sigma_s: {self.sigma_s}")
        print(f"d_f: {self.d_f}")
        print(f"h_i: {self.h_i}")
        print(f"p_i: {self.p_i}")
        print(f"q_i: {self.q_i}")
        print(f"T: {self.T}")
        print(f"M: {self.M}")

# 示例使用
if __name__ == "__main__":
    # 创建参数对象
    params = Parameters()

    # 设置默认值
    params.set_default_values()

    # 加载距离矩阵（假设有一个名为'增广距离矩阵-10站点.csv'的文件）
    try:
        params.load_distance_matrix("增广距离矩阵-10站点.csv")
    except FileNotFoundError:
        print("CSV file not found. Using default values for c.")

    params.load_demand_matrix("各时段累计需求-10站点.csv")



    # 打印参数
    # params.print_parameters()
    print("\nUsing parameters in the main program:")
    print(params.c[(29, 26)], params.t[(29, 26)])
    print(params.afa_i_s[(9, 6)], params.bta_i_s[(9, 33)])
    print(f"Truck capacity (Q_v): {params.Q_v}")
    print(f"Maximum operating duration (T): {params.T}")