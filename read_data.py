class Parameters:
    def __init__(self):
        # 定义参数并初始化默认值
        self.c = {}  # Travel cost of workers' trucks from node i to node j
        self.afa_i_s = {} # Reward of providing station i with a usable bike in slot s
        self.bta_i_s = {} # reward of removing an unusable bike for station i in slot s
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


    def set_default_values(self):
        # 设置默认值（可以根据实际需求调整）
        self.c[(1, 2)] = 10  # Example travel cost from node 1 to node 2
        self.c[(2, 3)] = 20  # Example travel cost from node 2 to node 3
        self.afa_i_s[(1, 1)] = 5  # Example reward for providing station 1 with a usable bike in slot 1
        self.bta_i_s[(1, 1)] = 3  # Example reward for removing an unusable bike for station 1 in slot 1
        self.Q_v = 50  # Truck capacity
        self.Q_f = 100  # Collection center capacity
        self.t[(1, 2)] = 15  # Example travel time from node 1 to node 2
        self.t[(2, 3)] = 25  # Example travel time from node 2 to node 3
        self.sigma_q = 5  # Average pick up/drop off time
        self.sigma_p = 10  # Average repair time on the road
        self.sigma_s = 60  # Time length of each time slot
        self.d_f = {9: 20, 10: 30, 11: 40}  # Number of repaired bikes in collection center f
        self.h_i = {3: 15, 4: 10, 5: 5}  # Number of repaired bikes to return to station i
        self.p_i = {3: 10, 4: 8, 5: 6}  # Number of broken bikes to return to collection center at station i
        self.q_i = {3: 5, 4: 3, 5: 2}  # Number of broken bikes to repair on the road at station i
        self.T = 8  # Maximum operating duration
        self.M = 1000000  # A large number
    def print_parameters(self):
        # 打印所有参数及其值
        print("Parameters:")
        print(f"c: {self.c}")
        print(f"Q_v: {self.Q_v}")
        print(f"Q_f: {self.Q_f}")
        print(f"t: {self.t}")
        print(f"sigma_q: {self.sigma_q}")
        print(f"sigma_p: {self.sigma_p}")
        print(f"d_f: {self.d_f}")
        print(f"h_i: {self.h_i}")
        print(f"p_i: {self.p_i}")
        print(f"q_i: {self.q_i}")
        print(f"T: {self.T}")
        print(f"M: {self.M}")



# # 创建参数对象
# params = Parameters()

# # 设置默认值
# params.set_default_values()

# # 打印参数
# params.print_parameters()

# # 使用参数
# print("\nUsing parameters in the main program:")
# print(f"Travel cost from node 1 to node 2: {params.c[(1, 2)]}")
# print(f"Truck capacity (Q_v): {params.Q_v}")
# print(f"Maximum operating duration (T): {params.T}")

