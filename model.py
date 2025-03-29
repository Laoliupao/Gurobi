from gurobipy import Model, GRB
from read_data import Parameters
from gurobipy import quicksum

# 创建 Gurobi 模型
model = Model("BikeSharingOptimization")

# 使用 for 循环生成集合
V = [v for v in range(-4, 0)]  # 工人集合,-4到-1
N = [n for n in range(0,10)]  # 站点集合，0到9
F = [f for f in range(10, 13)]  # 维修中心集合，10到12


# 使用 for 循环生成字典
# 手动定义每个 f 的虚拟节点数量
nodes_per_f = {10: 3, 11: 3, 12: 3}

# 将 Fai 定义为一个二维数组（列表的列表）
base_node = len(N) + len(F)   # 起始编号

FAI = []  # 存储生成的虚拟节点集合
current_node = base_node  # 当前编号

for f in F:
    num_nodes = nodes_per_f[f]  # 当前 f 的虚拟节点数量
    FAI.append([current_node + i for i in range(num_nodes)])  # 生成连续的虚拟节点编号
    current_node += num_nodes  # 更新下一个 f 的起始编号

# 定义 N_START，接着 FAI 的最后一个编号继续编号
N_START = [current_node + i for i in range(len(V))]  # 生成 N_START 的编号
current_node += len(V)  # 更新当前编号
# 定义 N_END
N_END = [current_node + i for i in range(len(V))]  # 生成 N_END 的编号
current_node += len(V)  # 更新当前编号

# 定义 N_ALL，将所有节点集合合并
N_ALL = N.copy()  # 创建 N 的独立副本
# 将 Fai 中的所有虚拟节点加入（Fai 是一个二维列表，需要展平）
for FAI_nodes in FAI:
    N_ALL.extend(FAI_nodes)

# 加入 N_START 和 N_END
N_ALL.extend(N_START)
N_ALL.extend(N_END)

# 输出 N_ALL
# print("N_ALL:", N_ALL)


# 展平二维虚拟节点集合FAI
FAI_FLAT = [node for sublist in FAI for node in sublist]
# 定义节点来源集合
i_sources = N_START + N + FAI_FLAT  # 起始节点集合
j_sources = N + FAI_FLAT + N_END    # 终止节点集合

# 修改A的生成方式（自动去重）
A = list({(i, j) for i in i_sources for j in j_sources if i != j})

S = [s for s in range(31, 51)]  # 所有时间段，31到50

# 创建参数对象
params = Parameters()
# 设置默认值
params.set_default_values()
# 加载距离矩阵（假设有一个名为'增广距离矩阵-10站点.csv'的文件）
params.load_distance_matrix("增广距离矩阵-10站点.csv")
# 加载需求矩阵（假设有一个名为'demand.csv'的文件）
params.load_demand_matrix("各时段累计需求-10站点.csv")
# 打印参数
# params.print_parameters()

#创建变量
X = [(i, j, v) for (i, j) in A for v in V]
L_BA = [i for i in N_ALL]
L_UNDERLINE = [i for i in N_ALL]
THETA = [i for i in FAI_FLAT]
EPSILON = [i for i in FAI_FLAT]
DELTA = [i for i in N]
TAO = [i for i in N_ALL]
ETA = [(i,j) for i in FAI_FLAT for j in FAI_FLAT]
LAMDA = [(i,s) for i in N for s in S]

x_ijv = model.addVars(X, vtype=GRB.BINARY, name='x_ijv') 
l_ba_i = model.addVars(L_BA, vtype=GRB.INTEGER, lb=0, name='l_ba') 
l_underline_i = model.addVars(L_UNDERLINE, vtype=GRB.INTEGER, lb=0, name='l_underline')
theta_i = model.addVars(THETA, vtype=GRB.INTEGER, lb=0, name='theta') 
epsilon_i = model.addVars(EPSILON, vtype=GRB.INTEGER, lb=0, name='epsilon') 
delta_i = model.addVars(DELTA, vtype=GRB.INTEGER, lb=0, name='delta') 
tao_i = model.addVars(TAO, vtype=GRB.CONTINUOUS, lb=0, name='tao') 
eta_ij = model.addVars(ETA, vtype=GRB.BINARY, name='eta_ij')    
lamda_is = model.addVars(LAMDA, vtype=GRB.BINARY, name='lamda_is')

# 添加约束
# 式2（约束1）
for v in V:
    n_start_v = N_START[v - V[0]]
    # 获取所有以 n_start_v 为起始节点的边
    edges_for_v = [(i, j) for (i, j) in A if i == n_start_v]
    model.addConstr(sum(x_ijv[i, j, v] for (i, j) in edges_for_v) <= 1,name=f'constraint1_{v}')

# 式3 约束2
for v in V:
    # 获取当前工作者的起始节点
    n_start_v = N_START[v - V[0]]
    
    # 遍历 N_START 中除 n_start_v 以外的所有节点
    for i in set(N_START) - {n_start_v}:
        # 找到所有以 i 为起点的边 (i,j) ∈ A
        edges_from_i = [(i, j) for (i_j, j) in A if i_j == i]
        
        # 添加约束：x_{(i,j)}^v 的和为 0
        model.addConstr(sum(x_ijv[i, j, v] for (i, j) in edges_from_i) == 0, name = f'constraint2_{i}_{v}')

# 式4 约束3
for v in V:
    # 获取当前工作者的终点节点
    n_end_v = N_END[v - V[0]]
    
    # 找到所有以 n_end_v 为终点的边 (i, n_end_v) ∈ A
    edges_to_n_end_v = [(i, j) for (i, j) in A if j == n_end_v]
    
    # 添加约束：x_{(i, n_end_v)}^v 的和不超过 1
    model.addConstr(sum(x_ijv[i, j, v] for (i, j) in edges_to_n_end_v) <= 1, name = f'constraint3_{v}')

# 式5 约束4
for v in V:
    n_end_v = N_END[v - V[0]]  # 获取工作者 v 的终止节点
    # 遍历 NEND 中除 n_end_v 外的所有节点 j
    for j in N_END:
        if j != n_end_v:  # 排除工作者 v 的终止节点
            # 对所有满足 (i, j) in A 的 i，求和 x_{(i,j)}^v
            model.addConstr(sum(x_ijv[i, j, v] for (i, j_arc) in A if j_arc == j) == 0, name = f'constraint4_{v}_{j}')

# 式6 约束5
for i in N:
    # 找到所有以 i 为起点的边 (i,j) ∈ A
    edges_from_i = [(i, j) for (i_j, j) in A if i_j == i]
    
    # 添加约束：对于每个节点 i，x_{(i,j)}^v 的和等于 1
    model.addConstr(sum(x_ijv[i, j, v] for (i, j) in edges_from_i for v in V) == 1, name = f'constraint5_{i}')

# 式7 约束6
for i in FAI_FLAT:  # 遍历每个节点 i ∈ Φ
    # 找到所有以 i 为起点的边 (i,j) ∈ A
    edges_from_i = [(i, j) for (i_j, j) in A if i_j == i]
    
    # 添加约束：对于每个节点 i，x^{(\nu)}_{(i,j)} 的和不超过 1
    model.addConstr(sum(x_ijv[i, j, nu] for (i, j) in edges_from_i for nu in V) <= 1, name = f'constraint6_{i}')

# 式8 约束7
for i in N + FAI_FLAT:  # 遍历每个节点 i ∈ N ∪ Φ
    for v in V:    # 遍历每个工作者 v ∈ V
        # 找到所有以 i 为终点的边 (j,i) ∈ A
        edges_to_i = [(j, i) for (j, i_j) in A if i_j == i]
        
        # 找到所有以 i 为起点的边 (i,j) ∈ A
        edges_from_i = [(i, j) for (i_j, j) in A if i_j == i]
        
        # 添加约束：x^v_{(j,i)} 的和等于 x^v_{(i,j)} 的和
        model.addConstr(sum(x_ijv[j, i, v] for (j, i) in edges_to_i) == sum(x_ijv[i, j, v] for (i, j) in edges_from_i), name = f'constraint7_{i}_{v}')

# 式9 约束8
for f in F:  # 遍历每个收集中心 f ∈ F
    # 获取 f 对应的虚拟节点集合 B_f
    B_f = FAI[F.index(f)]  # F = [9, 10, 11]，所以 F.index(9) = 0, F.index(10) = 1, 等等
    
    for i in B_f:  # 遍历每个节点 i ∈ B_f
        for v in V:  # 遍历每个工作者 v ∈ V
            # 找到所有以 i 为起点且终点 j ∈ B_f \ {i} 的边 (i,j) ∈ A
            edges_within_B_f = [(i, j) for (i_j, j) in A if i_j == i and j in B_f and j != i]
            
            # 添加约束：x_{(i,j)}^v 的和为 0
            model.addConstr(sum(x_ijv[i, j, v] for (i, j) in edges_within_B_f) == 0, name = f'constraint8_{f}_{i}_{v}')

# 式10 约束9
for (i, j) in A:  # 遍历每条边 (i,j) ∈ A
    if i in N:  # 确保 i ∈ N
        # 计算左侧表达式
        left_hand_side = (
            tao_i[i] +
            params.sigma_q * (params.p_i[i] + params.q_i[i] - delta_i[i] + params.h_i[i]) +
            params.sigma_p * delta_i[i] +
            params.t[(i, j)] -
            params.M * (1 - sum(x_ijv[i, j, v] for v in V))
        )
        # 添加约束：左侧 ≤ τ_j
        model.addConstr(left_hand_side <= tao_i[j], name = f'constraint9_{i}_{j}')

# 式11 约束10
for (i, j) in A:  # 遍历每条边 (i,j) ∈ A
    if i in FAI_FLAT:  # 确保 i ∈ Φ
        # 计算左侧表达式
        left_hand_side = (
            tao_i[i] +  # τ_i
            params.sigma_q * (epsilon_i[i] + theta_i[i]) +  # σ_q * (ε_i + θ_i)
            params.t[(i, j)] -  # t_(i,j)
            params.M * (1 - sum(x_ijv[i, j, v] for v in V))  # -M * (1 - Σ_v x_(i,j)^v)
        )
        # 添加约束：左侧 ≤ τ_j
        model.addConstr(left_hand_side <= tao_i[j], name = f'constraint10_{i}_{j}')

# 式12 约束11
for i in N_ALL:  # 遍历所有节点 i ∈ N_all
    model.addConstr(tao_i[i] <= params.T, name = f'constraint11_{i}')

# 式13 约束12
for (i, j) in A:  # 遍历每条边 (i,j) ∈ A
    if j in N:  # 确保 j ∈ N
        # 计算左侧表达式
        left_hand_side = (
            l_underline_i[i] +  # l_i
            params.p_i[j] + (params.q_i[j] - delta_i[j]) -  # p_j + (q_j - δ_j)
            params.M * (1 - sum(x_ijv[i, j, v] for v in V))  # -M * (1 - Σ_v x_(i,j)^v)
        )
        # 添加约束：左侧 ≤ l_j
        model.addConstr(left_hand_side <= l_underline_i[j], name = f'constraint12_{i}_{j}')

# 式14 约束13
for (i, j) in A:  # 遍历每条边 (i,j) ∈ A
    if j in FAI_FLAT:  # 确保 j ∈ Φ
        # 计算左侧表达式
        left_hand_side = (
            l_underline_i[i] - theta_i[j] +  # l_i - θ_j
            params.M * (1 - sum(x_ijv[i, j, v] for v in V))  # M * (1 - Σ_v x_(i,j)^v)
        )
        # 添加约束：左侧 ≥ l_j
        model.addConstr(left_hand_side >= l_underline_i[j], name = f'constraint13_{i}_{j}')

# 式15 约束14
for (i, j) in A:  # 遍历每条边 (i,j) ∈ A
    if j in N:  # 确保 j ∈ N
        # 计算左侧表达式
        left_hand_side = (
            l_ba_i[i] - params.h_i[j] +  # l̅_i - h_j
            params.M * (1 - sum(x_ijv[i, j, v] for v in V))  # M * (1 - Σ_v x_(i,j)^v)
        )
        # 添加约束：左侧 ≥ l_j
        model.addConstr(left_hand_side >= l_ba_i[j], name = f'constraint14_{i}_{j}')

# 式16 约束15
for (i, j) in A:  # 遍历每条边 (i,j) ∈ A
    if j in FAI_FLAT:  # 确保 j ∈ Φ
        # 计算左侧表达式
        left_hand_side = (
            l_ba_i[i] + epsilon_i[j] -  # l̅_i + ε_j
            params.M * (1 - sum(x_ijv[i, j, v] for v in V))  # -M * (1 - Σ_v x_(i,j)^v)
        )
        # 添加约束：左侧 ≤ l_j
        model.addConstr(left_hand_side <= l_ba_i[j], name = f'constraint15_{i}_{j}')

# 式17 约束16
for i in N_ALL:  # 遍历所有节点 i ∈ N_all
    model.addConstr(l_ba_i[i] + l_underline_i[i] <= params.Q_v, name = f'constraint16_{i}')

# 式18 约束17
for i in N_START:  # 遍历所有节点 i ∈ N_start
    model.addConstr(l_ba_i[i] + l_underline_i[i] == 0, name = f'constraint17_{i}')

# 式19 约束18
for j in N_END:
    # 找到所有以 j 为终点的边 (i,j) ∈ A
    edges_to_j = [(i, j) for (i, j_arc) in A if j_arc == j]
    # 计算 sum(l_bar_i + l_underline_i) 并添加约束
    model.addConstr(sum(l_ba_i[i] + l_underline_i[i] for (i, m) in edges_to_j) == 0, name = f'constraint18_{j}')

# 式20 约束19
for f in F:  # 遍历每个收集中心 f ∈ F
    # 获取 f 对应的虚拟节点集合 B_f
    B_f = FAI[F.index(f)]
    # 约束：Σ_{b_f ∈ B_f} ε_{b_f} ≤ d_f
    model.addConstr(sum(epsilon_i[b_f] for b_f in B_f) <= params.d_f[f],name = f'constraint19_{f}')

# 式21 约束20
for i in N:  # 遍历每个共享单车站点 i ∈ N
    model.addConstr(delta_i[i] <= params.q_i[i], name = f'constraint20_{i}')

# 线性化式22中的二次项
# 定义 y_ij 和 z_ij 的 (i, j) 对集合（其中 i, j ∈ FAI_FLAT）
Y = [(i, j) for i in FAI_FLAT for j in FAI_FLAT if i != j]
Z = [(i, j) for i in FAI_FLAT for j in FAI_FLAT if i != j]

# 将 y_ij 和 z_ij 作为整数变量添加到模型中
y_ij = model.addVars(Y, vtype=GRB.INTEGER, lb=0, name='y_ij')
z_ij = model.addVars(Z, vtype=GRB.INTEGER, lb=0, name='z_ij')

# 线性化约束：y_ij = theta_i * eta_ij
for i in FAI_FLAT:
    for j in FAI_FLAT:
        if i != j:
            # y_ij <= theta_i
            model.addConstr(y_ij[i, j] <= theta_i[i], name = f'constraint30_{i}_{j}')
            # y_ij <= M * eta_ij（M 是一个大常数，例如 params.M）
            model.addConstr(y_ij[i, j] <= params.M * eta_ij[i, j], name = f'constraint31_{i}_{j}')
            # y_ij >= theta_i - M * (1 - eta_ij)
            model.addConstr(y_ij[i, j] >= theta_i[i] - params.M * (1 - eta_ij[i, j]), name = f'constraint32_{i}_{j}')
            # y_ij >= 0（已在 addVars 中通过 lb=0 强制执行）

# 线性化约束：z_ij = epsilon_i * eta_ij
for i in FAI_FLAT:
    for j in FAI_FLAT:
        if i != j:
            # z_ij <= epsilon_i
            model.addConstr(z_ij[i, j] <= epsilon_i[i], name = f'constraint34_{i}_{j}')
            # z_ij <= M * eta_ij
            model.addConstr(z_ij[i, j] <= params.M * eta_ij[i, j], name = f'constraint35_{i}_{j}')
            # z_ij >= epsilon_i - M * (1 - eta_ij)
            model.addConstr(z_ij[i, j] >= epsilon_i[i] - params.M * (1 - eta_ij[i, j]), name = f'constraint36_{i}_{j}')
            # z_ij >= 0（已在 addVars 中通过 lb=0 强制执行）


# 式22 约束21
for f in F:  # 遍历每个收集中心 f ∈ F
    B_f = FAI[F.index(f)]  # 获取 f 对应的虚拟节点集合 B_f
    for j in B_f:  # 遍历每个虚拟节点 j ∈ B_f
        # 对所有 i ∈ B_f 且 i != j 的 y_ij[i, j] - z_ij[i, j] 求和
        sum_expr = sum(y_ij[i, j] - z_ij[i, j] for i in B_f if i != j)
        model.addConstr(params.d_f[f] + sum_expr <= params.Q_f, name = f'constraint21_{f}_{j}')

# 式23 约束22
for i in FAI_FLAT:  # 遍历每个虚拟节点 i ∈ Φ
    for j in FAI_FLAT:  # 遍历每个虚拟节点 j ∈ Φ
        # 约束：τ_i ≤ τ_j + M (1 - n_{i j})
        model.addConstr(tao_i[i] <= tao_i[j] + params.M * (1 - eta_ij[i, j]), name = f'constraint22_{i}_{j}')

# 式24 约束23
for i in FAI_FLAT:  # 遍历每个虚拟节点 i ∈ Φ
    for j in FAI_FLAT:  # 遍历每个虚拟节点 j ∈ Φ
        # 约束：τ_j ≤ τ_i + M n_{i j}
        model.addConstr(tao_i[j] <= tao_i[i] + params.M * eta_ij[i, j], name = f'constraint23_{i}_{j}')

# 线性化约束（式25和式26）
AUXI1 = [(i, s) for i in N for s in S] # 定义辅助变量集合 AUXI1
AUXI2 = [(i, s) for i in N for s in S] # 定义辅助变量集合 AUXI2

auxi1 = model.addVars(AUXI1, vtype=GRB.BINARY, name='auxi1')
auxi2 = model.addVars(AUXI2, vtype=GRB.BINARY, name='auxi2')

# 式39 约束38
for i in N:  # 遍历所有节点 i ∈ N
    for s in S:  # 遍历所有时间段 s ∈ S
        model.addConstr(
            tao_i[i] <= (s-1) * params.sigma_s + params.M * auxi1[i, s],
            name=f"constraint_38_{i}_{s}"
        )

# 式40 约束39
for i in N:  # 遍历所有节点 i ∈ N
    for s in S:  # 遍历所有时间段 s ∈ S
        model.addConstr(
            tao_i[i] >= (s-1) * params.sigma_s - params.M * (1 - auxi1[i, s]),
            name=f"constraint_39_{i}_{s}"
        )
# 式41 约束40
for i in N:  # 遍历所有节点 i ∈ N
    for s in S:  # 遍历所有时间段 s ∈ S
        model.addConstr(
            tao_i[i] >= s * params.sigma_s - params.M * (1 - auxi1[i, s]),
            name=f"constraint_40_{i}_{s}"
        )

# 式42 约束41
for i in N:  # 遍历所有节点 i ∈ N
    for s in S:  # 遍历所有时间段 s ∈ S
        model.addConstr(
            tao_i[i] <= s * params.sigma_s + params.M * auxi1[i, s],
            name=f"constraint_41_{i}_{s}"
        )

# 式43 约束42
for i in N:
    for s in S:
        model.addConstr(
            lamda_is[i, s] >= auxi1[i, s] +auxi2[i, s] - 1,
            name=f"constraint_42_{i}_{s}"
        )

# 式44 约束43
for i in N:
    for s in S:
        model.addConstr(
            lamda_is[i, s] <= auxi1[i, s],
            name=f"constraint_43_{i}_{s}"
        )

# 式45 约束44
for i in N:
    for s in S:
        model.addConstr(
            lamda_is[i, s] <= auxi2[i, s],
            name=f"constraint_44_{i}_{s}"
        )

# 式26 约束25
for i in N:
    model.addConstr(quicksum(lamda_is[i, s] for s in S) == 1, name = f'constraint25_{i}')

# 线性化约束
K = [(i, s) for i in N for s in S] # 定义辅助变量集合 K
k = model.addVars(K, vtype=GRB.INTEGER, lb=0, name='k')

# 式46 约束45
for i in N:
    for s in S:
        model.addConstr(k[i, s] <= delta_i[i], name = f"constraint45_{i}_{s}")

# 式47 约束46
for i in N:
    for s in S:
        model.addConstr(k[i, s] <= params.M * lamda_is[i, s], name=f"constraint46_{i}_{s}")

# 式48 约束47
for i in N:
    for s in S:
        model.addConstr(k[i, s] >= delta_i[i] - params.M * (1 - lamda_is[i, s]),name=f"constraint47_{i}_{s}")

# 约束 (50)：k >= 0
# 已经在 addVars 中通过 lb=0 强制执行，无需额外添加


# 定义目标函数
# 第一部分：∑_{i∈N} ∑_{s∈S} α_is * λ_is * (δ_i + h_i)
term1 = quicksum(
    params.afa_i_s[i, s] * lamda_is[i, s] * params.h_i[i] + params.afa_i_s[i, s] * k[i, s]
    for i in N
    for s in S
)

# 第二部分：∑_{i∈N} ∑_{s∈S} β_is * λ_is * (p_i + q_i)
term2 = quicksum(
    params.bta_i_s[i, s] * lamda_is[i, s] * (params.p_i[i] + params.q_i[i])
    for i in N
    for s in S
)

# 第三部分：-∑_{i∈N} c_p^m * (p_i + q_i - δ_i)
term3 = -quicksum(
    params.cpm * (params.p_i[i] + params.q_i[i] - delta_i[i])
    for i in N
)

# 第四部分：-∑_{i∈N} c_q^m * δ_i
term4 = -quicksum(
    params.cqm * delta_i[i]
    for i in N
)

# 第五部分：-∑_{v∈V} ∑_{(i,j)∈A} c_{(i,j)}^r * x_{(i,j)}^v
term5 = -quicksum(
    params.c[i, j] * x_ijv[i, j, v]
    for v in V
    for (i, j) in A
)

# 总目标函数 F
objective = term1 + term2 + term3 + term4 + term5

# 设置目标函数（最大化）
model.setObjective(objective, GRB.MAXIMIZE)

# 设置求解参数
model.setParam('OutputFlag', 1)  # 输出求解过程
model.setParam('TimeLimit', 3600)  # 设置最大求解时间（单位：秒）  

# 求解模型
model.optimize()     

# 输出求解结果
# 1. 基本求解结果
if model.status == GRB.OPTIMAL:
    print("最优解已找到！")
    print("目标函数值:", model.objVal)
elif model.status == GRB.INFEASIBLE:
    print("模型无解（不可行）。")
    # 计算不可行性原因
    model.computeIIS()
    model.write("model.ilp")
    print("不可行约束已写入 'model.ilp' 文件，请查看。")
elif model.status == GRB.UNBOUNDED:
    print("模型无界。")
elif model.status == GRB.TIME_LIMIT:
    print("达到时间限制，求解未完成。")
    if model.SolCount > 0:
        print("找到的可行解目标值:", model.objVal)
else:
    print("求解状态:", model.status)

# 2. 输出变量值
if model.status == GRB.OPTIMAL or model.SolCount > 0:
    print("\n变量值（仅输出非零值）：")
    for v in model.getVars():
        if abs(v.x) > 1e-6:  # 只输出非零值（避免浮点误差）
            print(f"{v.varName} = {v.x}")

    # 单独输出 x_ijv 变量（路径选择变量）
    print("\nx_ijv 变量值（仅输出非零值）：")
    for (i, j, v) in X:
        if abs(x_ijv[i, j, v].x) > 1e-6:
            print(f"x_ijv[{i},{j},{v}] = {x_ijv[i, j, v].x}")

    # 单独输出 delta_i 变量（站点调整量）
    print("\ndelta_i 变量值（仅输出非零值）：")
    for i in N:
        if abs(delta_i[i].x) > 1e-6:
            print(f"delta_i[{i}] = {delta_i[i].x}")

    # 单独输出 tao_i 变量（到达时间）
    print("\ntao_i 变量值（仅输出非零值）：")
    for i in N_ALL:
        if abs(tao_i[i].x) > 1e-6:
            print(f"tao_i[{i}] = {tao_i[i].x}")

    # 3. 输出约束的松弛值
    print("\n约束的松弛值（仅输出非零值）：")
    for constr in model.getConstrs():
        slack = constr.getAttr('Slack')
        if abs(slack) > 1e-6:
            print(f"约束 {constr.ConstrName}: 松弛值 = {slack}")

    # 4. 输出目标函数各部分的值
    term1_value = sum(
        params.afa_i_s[i, s] * lamda_is[i, s].x * (delta_i[i].x + params.h_i[i])
        for i in N
        for s in S
    )
    term2_value = sum(
        params.bta_i_s[i, s] * lamda_is[i, s].x * (params.p_i[i] + params.q_i[i])
        for i in N
        for s in S
    )
    term3_value = -sum(
        params.cpm * (params.p_i[i] + params.q_i[i] - delta_i[i].x)
        for i in N
    )
    term4_value = -sum(
        params.cqm * delta_i[i].x
        for i in N
    )
    term5_value = -sum(
        params.c[i, j] * x_ijv[i, j, v].x
        for v in V
        for (i, j) in A
    )

    print("\n目标函数各部分值：")
    print(f"term1 (α_is * λ_is * (δ_i + h_i)) = {term1_value}")
    print(f"term2 (β_is * λ_is * (p_i + q_i)) = {term2_value}")
    print(f"term3 (-c_p^m * (p_i + q_i - δ_i)) = {term3_value}")
    print(f"term4 (-c_q^m * δ_i) = {term4_value}")
    print(f"term5 (-c_(i,j)^r * x_(i,j)^v) = {term5_value}")
    print(f"总目标值: {term1_value + term2_value + term3_value + term4_value + term5_value}")

# 5. 将结果保存到文件
with open("gurobi_results.txt", "w") as f:
    f.write("Gurobi 求解结果\n")
    f.write("================\n")
    if model.status == GRB.OPTIMAL:
        f.write("最优解已找到！\n")
        f.write(f"目标函数值: {model.objVal}\n")
    else:
        f.write(f"求解状态: {model.status}\n")

    if model.status == GRB.OPTIMAL or model.SolCount > 0:
        f.write("\n变量值（仅输出非零值）：\n")
        for v in model.getVars():
            if abs(v.x) > 1e-6:
                f.write(f"{v.varName} = {v.x}\n")

        f.write("\nx_ijv 变量值（仅输出非零值）：\n")
        for (i, j, v) in X:
            if abs(x_ijv[i, j, v].x) > 1e-6:
                f.write(f"x_ijv[{i},{j},{v}] = {x_ijv[i, j, v].x}\n")

        f.write("\ndelta_i 变量值（仅输出非零值）：\n")
        for i in N:
            if abs(delta_i[i].x) > 1e-6:
                f.write(f"delta_i[{i}] = {delta_i[i].x}\n")

        f.write("\ntao_i 变量值（仅输出非零值）：\n")
        for i in N_ALL:
            if abs(tao_i[i].x) > 1e-6:
                f.write(f"tao_i[{i}] = {tao_i[i].x}\n")

        f.write("\n约束的松弛值（仅输出非零值）：\n")
        for constr in model.getConstrs():
            slack = constr.getAttr('Slack')
            if abs(slack) > 1e-6:
                f.write(f"约束 {constr.ConstrName}: 松弛值 = {slack}\n")

        f.write("\n目标函数各部分值：\n")
        f.write(f"term1 (α_is * λ_is * (δ_i + h_i)) = {term1_value}\n")
        f.write(f"term2 (β_is * λ_is * (p_i + q_i)) = {term2_value}\n")
        f.write(f"term3 (-c_p^m * (p_i + q_i - δ_i)) = {term3_value}\n")
        f.write(f"term4 (-c_q^m * δ_i) = {term4_value}\n")
        f.write(f"term5 (-c_(i,j)^r * x_(i,j)^v) = {term5_value}\n")
        f.write(f"总目标值: {term1_value + term2_value + term3_value + term4_value + term5_value}\n")

print("结果已保存到 'gurobi_results.txt' 文件。")