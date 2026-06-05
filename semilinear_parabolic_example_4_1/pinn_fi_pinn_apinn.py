import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy.stats import multivariate_normal, truncnorm
import scipy

# torch.set_default_dtype(torch.float64)
# 设置随机种子
# np.random.seed(1234)
# torch.manual_seed(1234)
# torch.cuda.manual_seed(1234)
# torch.cuda.manual_seed_all(1234)

###################### 定义超参数 ##################
parser = argparse.ArgumentParser()
parser.add_argument('--e', type=int, default=801, help='Epochs')
parser.add_argument('--number', type=list, default=[200, 50], help='内点，边界取点数')
parser.add_argument('--ranges', type=list, default=[[-1, 1], [0, 0.163]], help='空间，时间范围')
parser.add_argument('--beta', type=list, default=[0.04, 1], help='coefficient')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--net', type=list, default=[2, 50, 50, 50, 2], help='网络架构')
parser.add_argument('--filename', type=str, default='3-PINN_0dot163_0dot1_8_5_2', help='保存数据文件名')
parser.add_argument('--resample', type=int, default=0, choices=[0, 1], help='是否启用重采样 (0:禁用, 1:启用)')
parser.add_argument('--use_time_loss', type=int, default=0, choices=[0, 1],
                    help='是否使用时间离散损失 (0:禁用, 1:启用)')
parser.add_argument('--use_weight', type=int, default=0, choices=[0, 1], help='是否使用自适应权重(0:禁用, 1:启用)')
parser.add_argument('--weight_p', type=float, default=1.0, help='加权指数 p')
# 添加损失项的权重参数
parser.add_argument('--w_pde', type=float, default=0.1, help='PDE损失权重')
parser.add_argument('--w_bc', type=float, default=8, help='边界损失权重')
parser.add_argument('--w_ini', type=float, default=5, help='初始条件损失权重')
parser.add_argument('--w_g', type=float, default=2, help='耦合项损失权重')
parser.add_argument('--w_time', type=float, default=1.0, help='时间离散损失权重')

# 添加FI-PINNs参数
parser.add_argument('--use_fi', type=int, default=0, choices=[0, 1], help='是否启用FI-PINNs自适应采样')
parser.add_argument('--epsilon_r', type=float, default=0.01, help='残差阈值')
parser.add_argument('--epsilon_p', type=float, default=0.05, help='失败概率阈值')
parser.add_argument('--sais_N1', type=int, default=500, help='SAIS中每次迭代的样本数')
parser.add_argument('--sais_p0', type=float, default=0.05, help='SAIS中用于更新的样本比例')
parser.add_argument('--sais_N2', type=int, default=2000, help='SAIS中用于估计失败概率的样本数')
parser.add_argument('--sais_max_iter', type=int, default=5, help='SAIS最大迭代次数')
parser.add_argument('--fi_interval', type=int, default=100, help='FI-PINNs自适应采样间隔(epoch)')
parser.add_argument('--max_new_points', type=int, default=100, help='每次添加的最大新点数')

# 添加APINN自适应权重参数
parser.add_argument('--use_apinn', type=int, default=0, choices=[0, 1],
                    help='是否启用APINN自适应权重算法 (0:禁用, 1:启用)')
parser.add_argument('--apinn_alpha', type=float, default=7,
                    help='APINN权重调整系数α')
parser.add_argument('--apinn_interval', type=int, default=50,
                    help='APINN权重更新间隔(epoch)')
parser.add_argument('--apinn_window', type=int, default=50,
                    help='APINN计算平均损失的窗口大小')

args = parser.parse_args()

# 全局参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
delta = 3
filename = args.filename


######################### 网络定义 #########################
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入层
        self.linearIn = nn.Linear(args.net[0], args.net[1])
        nn.init.xavier_normal_(self.linearIn.weight)
        nn.init.constant_(self.linearIn.bias, 0)

        # 隐藏层
        self.linear = nn.ModuleList()
        for _ in range(len(args.net) - 3):  # 调整隐藏层数量
            layer = nn.Linear(args.net[1], args.net[1])
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
            self.linear.append(layer)

        # 输出层
        self.layer1 = nn.Linear(args.net[1], args.net[1])
        self.linearOut = nn.Linear(args.net[1], args.net[-1])
        nn.init.xavier_normal_(self.linearOut.weight)
        nn.init.constant_(self.linearOut.bias, 0)

    def forward(self, X):
        x = torch.tanh(self.linearIn(X))
        for layer in self.linear:
            x = torch.tanh(layer(x))
        return self.linearOut(x)


############################# 函数定义 ######################
# 边界和源函数
bd_fun = lambda x: 0
source_fun_1 = lambda x: 0
initial_fun = lambda x: torch.cos(torch.pi * x[:, 0:1] / 2)


# 梯度计算
def grad(y, x):
    dydx, = torch.autograd.grad(
        outputs=y,
        inputs=x,
        retain_graph=True,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )
    return dydx


# 物理算子
def Operator(U, gU, X, domain):
    U_t = grad(U, X)[:, 1:2]
    U_tt = grad(U_t, X)[:, 1:2]
    U_x = grad(U, X)[:, 0:1]
    U_xx = grad(U_x, X)[:, 0:1]
    # gU_t = grad(gU, X)[:, 1:2]

    if domain == 'domain_1':
        output =  U_t - U_xx-delta*torch.exp(U)
    else:
        output = args.beta[1] * (U_tt - U_xx) + U
    return output


# 自适应时间加权函数
def group_time_weight_loss(inputs, residuals, p=1.0):
    with torch.no_grad():
        t_vals = inputs[:, 1]
        # 将时间进行粗略离散（避免浮点精度问题），比如保留3位有效数字
        rounded_t = torch.round(t_vals * 1000) / 1000
        unique_times = torch.unique(rounded_t)

    loss = 0.0
    count = 0

    for t in unique_times:
        mask = (torch.abs(t_vals - t) < 1e-5)
        res_t = residuals[mask]

        if res_t.numel() == 0:
            continue

        Tb = args.ranges[1][1]
        weight = 1.0 / (Tb - t + 1e-8) ** p

        loss += weight * torch.mean(res_t ** 2)
        count += 1

    return loss / count if count > 0 else torch.tensor(0.0, device=inputs.device)


############################### 采样函数 #####################################
def grow_data(ranges, method, domain):
    x_mesh = torch.linspace(ranges[0][0], ranges[0][1], steps=args.number[1])
    y_mesh = torch.linspace(ranges[1][0], ranges[1][1], steps=args.number[1])
    X, Y = torch.meshgrid(x_mesh, y_mesh, indexing="ij")

    # 生成内部点
    if method == 'mesh':
        x_i = torch.stack([X.reshape(-1), Y.reshape(-1)]).T.to(device).requires_grad_(True)
        # 生成边界点
        xb1 = torch.stack([X[0], Y[0]]).T  # x_min
        xb2 = torch.stack([X[-1], Y[0]]).T  # x_max
        if domain == 'domain_1':
            x_b_lower = xb1.to(device).requires_grad_(True)  # 下边界
            x_b_upper = xb2.to(device).requires_grad_(True)  # 上边界
            x_ini = torch.stack([X[:, 0], Y[:, 0]]).T.to(device).requires_grad_(True)  # t=0
            return x_i, x_b_lower, x_b_upper, x_ini
        else:
            x_b = torch.cat([xb2, torch.stack([X[:, 0], Y[:, 0]]).T,
                             torch.stack([X[:, -1], Y[:, -1]]).T], dim=0).to(device)
            x_ini = torch.stack([X[:, 0], Y[:, 0]]).T.to(device)
            return x_i, x_b, x_ini
    else:
        coor = torch.tensor([[ranges[0][0], ranges[1][0]], [ranges[0][1], ranges[1][1]]]).to(device)
        x_i = coor[0] + (coor[1] - coor[0]) * torch.rand(args.number[0], 2, device=device).requires_grad_(True)

        # 生成边界点
        xb1 = torch.stack([X[0], Y[0]]).T.to(device)  # x_min
        xb2 = torch.stack([X[-1], Y[0]]).T.to(device)  # x_max

        if domain == 'domain_1':
            x_b_lower = xb1.to(device).requires_grad_(True)  # 下边界
            x_b_upper = xb2.to(device).requires_grad_(True)  # 上边界
            x_ini = torch.stack([X[:, 0], Y[:, 0]]).T.to(device).requires_grad_(True)  # t=0
            return x_i, x_b_lower, x_b_upper, x_ini
        else:
            x_b = torch.cat([xb2, torch.stack([X[:, 0], Y[:, 0]]).T,
                             torch.stack([X[:, -1], Y[:, -1]]).T], dim=0).to(device)
            x_ini = torch.stack([X[:, 0], Y[:, 0]]).T.to(device)
            return x_i, x_b, x_ini


def generate_solution_on_grid(model, x_test, grid_shape):
    model.eval()
    u_pred = model(x_test)[:, 0:1]

    u_pred_x = grad(u_pred, x_test)[:, 0:1]
    u_pred_t = grad(u_pred, x_test)[:, 1:2]

    u_pred_x_reshaped = u_pred_x.reshape(grid_shape).T.detach().cpu().numpy()
    u_pred_t_reshaped = u_pred_t.reshape(grid_shape).T.detach().cpu().numpy()
    u_pred_reshaped = u_pred.reshape(grid_shape).T.detach().cpu().numpy()

    return u_pred_x_reshaped, u_pred_t_reshaped, u_pred_reshaped


def plot_heatmap(solution, x_vals, y_vals, name='solution.pdf', title='Solution Heatmap', vmin=None, vmax=None):
    folder = f'./results/{filename}'
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(x_vals, y_vals, solution, cmap='viridis', levels=100, vmin=vmin, vmax=vmax)
    plt.colorbar(contour, label='Value')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(folder, name), dpi=300, bbox_inches='tight')
    plt.close()


def plot_relative_error(error, x_vals, y_vals, name='relative_error.pdf'):
    folder = f'./results/{filename}'
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(x_vals, y_vals, error, cmap='viridis')
    cbar = plt.colorbar(contour, label='Relative Error')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Relative Error Heatmap')
    plt.grid(True)
    plt.savefig(os.path.join(folder, name), dpi=300, bbox_inches='tight')
    plt.close()


# 为每个点找到最近的时间步
def find_nearest_timestep(current_points, all_points):
    """
    为每个点找到相同空间位置但时间更早的最近点。
    current_points: Tensor[N, 2], 每行为 (x, t)
    all_points: Tensor[M, 2], 每行为 (x, t)
    返回:
        nearest_points: List[Tensor or None]
        time_differences: List[float or None]
    """
    nearest_points = []
    time_differences = []

    for i, point in enumerate(current_points):
        x, t = point[0], point[1]

        # 找到相同 x 的所有点（允许浮点误差）
        same_x_points = all_points[torch.abs(all_points[:, 0] - x) < 1e-5]

        # 仅保留时间更早的点
        earlier_points = same_x_points[same_x_points[:, 1] < t]

        if earlier_points.size(0) > 0:
            # 计算时间差并选择最小的
            diffs = t - earlier_points[:, 1]
            min_idx = torch.argmin(diffs)
            nearest_point = earlier_points[min_idx]
            time_diff = diffs[min_idx].item()

            nearest_points.append(nearest_point)
            time_differences.append(time_diff)
        else:
            nearest_points.append(None)
            time_differences.append(None)

    return nearest_points, time_differences


######################## FI-PINNs 自适应采样 ########################
def sais_algorithm(model, domain_ranges, epsilon_r, epsilon_p, N1, p0, N2, max_iter, max_new_points):
    """
    Self-Adaptive Importance Sampling (SAIS) for FI-PINNs
    返回: 失败概率估计, 新采样点
    """
    # 定义域范围
    x_min, x_max = domain_ranges[0]
    t_min, t_max = domain_ranges[1]
    domain_volume = (x_max - x_min) * (t_max - t_min)

    # 初始化: 先验分布为均匀分布
    prior_dist = lambda size: torch.stack([
        torch.rand(size, device=device) * (x_max - x_min) + x_min,
        torch.rand(size, device=device) * (t_max - t_min) + t_min
    ], dim=1).requires_grad_(True)

    # 初始建议分布 (均匀分布)
    proposal_mean = torch.tensor([(x_min + x_max) / 2, (t_min + t_max) / 2], device=device)
    proposal_cov = torch.eye(2, device=device) * torch.tensor([(x_max - x_min) ** 2 / 12, (t_max - t_min) ** 2 / 12],
                                                              device=device)

    # SAIS迭代
    for k in range(max_iter):
        # 从当前建议分布采样
        samples = prior_dist(N1)

        # 计算残差
        # with torch.no_grad():
        u_pred = model(samples)[:, 0:1]
        g_pred = model(samples)[:, 1:2]
        residuals = torch.abs(Operator(u_pred, g_pred, samples, domain='domain_1'))
        g_values = residuals - epsilon_r

        # 按残差降序排序
        sorted_indices = torch.argsort(residuals.squeeze(), descending=True)
        sorted_samples = samples[sorted_indices]
        sorted_residuals = residuals[sorted_indices]
        sorted_g = g_values[sorted_indices]

        # 计算失败点数量
        N_eta = torch.sum(sorted_g > 0).item()
        N_p = int(p0 * N1)

        # 检查是否满足停止条件
        if N_eta >= N_p:
            break

        # 更新建议分布 (使用前N_p个点)
        top_samples = sorted_samples[:N_p]
        proposal_mean = torch.mean(top_samples, dim=0)
        proposal_cov = torch.cov(top_samples.T)

        # 确保协方差矩阵正定
        if torch.any(torch.isnan(proposal_cov)) or torch.any(torch.isinf(proposal_cov)):
            proposal_cov = torch.eye(2, device=device) * 1e-4

    # 最终建议分布
    final_proposal_mean = proposal_mean
    final_proposal_cov = proposal_cov

    # 从最终建议分布采样
    final_samples = prior_dist(N2)

    # 计算残差和失败指标
    # with torch.no_grad():
    u_pred_final = model(final_samples)[:, 0:1]
    g_pred_final = model(final_samples)[:, 1:2]
    residuals_final = torch.abs(Operator(u_pred_final, g_pred_final, final_samples, domain='domain_1'))
    g_values_final = residuals_final - epsilon_r
    failure_indicator = (g_values_final > 0).float()

    # 估计失败概率
    failure_prob = torch.mean(failure_indicator).item()

    # 选择新点: 残差最大的前 max_new_points 个点
    _, top_indices = torch.topk(residuals_final.squeeze(), min(max_new_points, N2))
    new_points = final_samples[top_indices].detach().clone()

    return failure_prob, new_points, final_proposal_mean, final_proposal_cov


############################### 训练函数 #####################################
def train(model_1):
    # 初始化
    use_adam = False  # True 使用 Adam，False 使用 LBFGS
    if use_adam:
        optimizer = torch.optim.Adam(model_1.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.LBFGS(
            model_1.parameters(),
            lr=1.0,
            max_iter=20,
            history_size=100,
            line_search_fn="strong_wolfe"
        )

    # 生成数据
    x_train_1, x_b_lower, x_b_upper, x_ini = grow_data(args.ranges, method='random', domain='domain_1')
    x_test_1, _, _, _ = grow_data(args.ranges, method='mesh', domain='domain_1')

    # 存储原始训练点，避免修改原始点
    original_x_train_1 = x_train_1.detach().clone()
    current_x_train_1 = original_x_train_1.clone().requires_grad_(True)

    # 存储所有训练点（用于查找最近时间步）
    all_train_points = [current_x_train_1.detach().clone()]

    # 准备参考解
    x_ref = np.linspace(args.ranges[0][0], args.ranges[0][1], args.number[1])
    t_eval = np.linspace(args.ranges[1][0], args.ranges[1][1], args.number[1])
    dx = x_ref[1] - x_ref[0]

    # 初始条件（排除边界）
    u0_flat = np.cos(np.pi * x_ref[1:-1] / 2)

    def ode_rhs(t, u_flat):
        u = np.zeros_like(x_ref)
        u[1:-1] = u_flat
        u_xx = np.zeros_like(u)
        u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
        du_dt = u_xx + 3 * np.exp(u)
        return du_dt[1:-1]

    sol = solve_ivp(ode_rhs, [0, args.ranges[1][1]], u0_flat, t_eval=t_eval, method='Radau')

    # 重构参考解
    u_ref = np.zeros((len(x_ref), len(t_eval)))
    u_ref[1:-1, :] = sol.y

    # 存储变量
    Loss_1 = []
    L2_errors = []
    failure_probs = []  # 存储失败概率

    # APINN相关变量初始化
    if args.use_apinn:
        # 初始权重 (初始条件, 下边界, 上边界, PDE损失)
        lambda_weights = {
            'ini': args.w_ini,
            'bc_lower': args.w_bc,
            'bc_upper': args.w_bc,
            'pde': args.w_pde,

        }
        # 存储损失历史
        loss_history = {key: [] for key in lambda_weights}
        # 存储权重历史
        weight_history = {key: [] for key in lambda_weights}
        # 添加耦合项和时间损失的固定权重
        fixed_weights = {
            'time': args.w_time,
            'g': 0.0
        }

    start_time = time.time()

    # 训练循环
    for epoch in range(args.e):
        def closure():
            global loss_ini,bc_lower_loss,bc_upper_loss,pde_loss_1,loss_g
            optimizer.zero_grad()

            # 使用当前训练点
            u_1 = model_1(current_x_train_1)[:, 0:1]
            gu_1 = model_1(current_x_train_1)[:, 1:2]
            lap_1 = Operator(u_1, gu_1, current_x_train_1, domain='domain_1')
            source_1 = source_fun_1(current_x_train_1)
            residuals = lap_1 - source_1
            if args.use_weight == 1:
                pde_loss_1 = group_time_weight_loss(current_x_train_1, residuals, p=args.weight_p)
            else:
                pde_loss_1 = torch.mean(residuals ** 2)

            # 边界损失
            bc_lower_pred = model_1(x_b_lower)[:, 0:1]
            bc_lower_loss = torch.mean((bc_lower_pred - 0) ** 2)  # bd_fun always returns 1

            bc_upper_pred = model_1(x_b_upper)[:, 0:1]
            bc_upper_loss = torch.mean((bc_upper_pred - 0) ** 2)  # bd_fun always returns 1

            # 初始条件损失
            u_ini_pred = model_1(x_ini)[:, 0:1]
            ext_ini = initial_fun(x_ini)
            loss_ini = torch.mean((u_ini_pred - ext_ini) ** 2)

            # 耦合项损失
            loss_g = 0.0

            # 时间离散损失 - 新添加的项
            loss_time = torch.tensor(0.0, device=device)
            if args.use_time_loss and epoch > 10:  # 前10个epoch不添加，让模型先初步收敛
                # 查找每个点的最近时间步
                nearest_points, time_diffs = find_nearest_timestep(current_x_train_1, current_x_train_1)

                valid_points = []
                for i, (point, nearest, h) in enumerate(zip(current_x_train_1, nearest_points, time_diffs)):
                    if nearest is not None and h is not None and h > 1e-6:
                        # 计算当前点的值
                        u_current = u_1[i]
                        gu_current = gu_1[i]

                        # 计算当前点的拉普拉斯算子
                        u_x = grad(u_current, current_x_train_1)[i, 0:1]
                        u_xx = grad(u_x, current_x_train_1)[i, 0:1]

                        # 计算最近点的值
                        nearest_output = model_1(nearest.unsqueeze(0))
                        gu_nearest = nearest_output[:, 1:2]

                        # 计算时间离散约束
                        F_u = torch.exp(u_current)  # 根据方程定义 F(u) = exp(u)
                        constraint = gu_current - gu_nearest + delta * h + (h * u_xx) / F_u

                        # 添加到损失
                        loss_time += torch.mean(constraint ** 2)
                        valid_points.append(1)

                if valid_points:
                    loss_time = loss_time / len(valid_points)
                else:
                    loss_time = torch.tensor(0.0, device=device)

            # 组合损失
            if args.use_apinn:
                loss = (lambda_weights['ini'] * loss_ini +
                        lambda_weights['bc_lower'] * bc_lower_loss +
                        lambda_weights['bc_upper'] * bc_upper_loss +
                        lambda_weights['pde'] * pde_loss_1 +
                        fixed_weights['g'] * loss_g +
                        fixed_weights['time'] * loss_time)
            else:
                # 原始固定权重组合
                loss = (args.w_ini * loss_ini +
                        args.w_bc * (bc_lower_loss + bc_upper_loss) / 2 +  # 平均边界损失
                        args.w_pde * pde_loss_1 +
                        args.w_g * loss_g +
                        args.w_time * loss_time)

            loss.backward(retain_graph=True)
            return loss

        optimizer.step(closure)

        # APINN权重更新逻辑
        if args.use_apinn and epoch % args.apinn_interval == 0 and epoch > 0:
            # 存储当前损失
            current_losses = {
                'ini': loss_ini.item(),
                'bc_lower': bc_lower_loss.item(),
                'bc_upper': bc_upper_loss.item(),
                'pde': pde_loss_1.item(),

            }

            for key in current_losses:
                loss_history[key].append(current_losses[key])
                # 保持窗口大小
                if len(loss_history[key]) > args.apinn_window:
                    loss_history[key].pop(0)

            # 计算平均损失
            avg_losses = {}
            for key in current_losses:
                if len(loss_history[key]) > 0:
                    avg_losses[key] = np.mean(loss_history[key])

            if len(avg_losses) == 4:  # 确保所有损失项都有数据
                min_loss = min(avg_losses.values())
                max_loss = max(avg_losses.values())

                # 计算比率并更新权重
                ratio = max_loss / min_loss if min_loss > 1e-8 else float('inf')

                if ratio > 10:  # 论文中的阈值条件
                    for key in avg_losses:
                        # 计算调整系数 R_j
                        R_j = (avg_losses[key] - min_loss) / (max_loss - min_loss)
                        # 更新权重 λ_j = 1 + α * R_j
                        lambda_weights[key] = 1.0 + args.apinn_alpha * R_j
                        weight_history[key].append(lambda_weights[key])

            print(f"Epoch {epoch}: APINN权重更新 - "
                  f"初始: {lambda_weights['ini']:.2f}, "
                  f"下边界: {lambda_weights['bc_lower']:.2f}, "
                  f"上边界: {lambda_weights['bc_upper']:.2f}, "
                  f"PDE: {lambda_weights['pde']:.2f}, ")

        # FI-PINNs自适应采样
        if args.use_fi and epoch % args.fi_interval == 0 and epoch > 0:
            failure_prob, new_points, proposal_mean, proposal_cov = sais_algorithm(
                model_1,
                args.ranges,
                args.epsilon_r,
                args.epsilon_p,
                args.sais_N1,
                args.sais_p0,
                args.sais_N2,
                args.sais_max_iter,
                args.max_new_points
            )

            failure_probs.append(failure_prob)
            print(f"Epoch {epoch}: Failure probability = {failure_prob:.4f}")

            # 如果失败概率大于阈值，添加新点
            if failure_prob > args.epsilon_p and len(new_points) > 0:
                # 创建新的训练点集合（包含原始点和新点）
                updated_train_points = torch.cat([
                    current_x_train_1.detach().clone(),
                    new_points.requires_grad_(True)
                ], dim=0).requires_grad_(True)

                # 更新当前训练点
                current_x_train_1 = updated_train_points
                all_train_points.append(updated_train_points.detach().clone())
                print(f"Epoch {epoch}: Added {len(new_points)} new points, total points: {len(current_x_train_1)}")

        # 每10个epoch记录一次
        if epoch % 10 == 0:
            # 计算当前解的L2误差
            _, _, solution_on_grid = generate_solution_on_grid(
                model_1, x_test_1, (args.number[1], args.number[1]))

            # 创建内部点掩码（排除边界）
            inner_mask = (x_ref > args.ranges[0][0] + 1e-6) & (x_ref < args.ranges[0][1] - 1e-6)

            # 计算相对L2误差
            if np.linalg.norm(u_ref[inner_mask, :].T) > 1e-8:
                error = solution_on_grid[:, inner_mask] - u_ref[inner_mask, :].T
                rel_l2_error = np.linalg.norm(error) / np.linalg.norm(u_ref[inner_mask, :].T)
            else:
                rel_l2_error = np.nan
            L2_errors.append(rel_l2_error)

            # 记录时间
            t = time.time() - start_time
            start_time = time.time()

            print(f"Epoch:{epoch}, Time: {t:.2f}s, L2 Error: {rel_l2_error:.4e}")

            # 保存损失和误差数据
            folder = f'./results/{filename}'
            os.makedirs(folder, exist_ok=True)
            np.save(os.path.join(folder, 'Loss.npy'), Loss_1)
            np.save(os.path.join(folder, 'L2_errors.npy'), L2_errors)
            if args.use_fi:
                np.save(os.path.join(folder, 'failure_probs.npy'), failure_probs)

            # 保存APINN权重历史
            if args.use_apinn:
                np.save(os.path.join(folder, 'apinn_weight_history.npy'), weight_history)

            # 绘制损失曲线
            plt.figure(figsize=(10, 6))
            plt.plot(Loss_1)
            plt.yscale('log')
            plt.title('Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(folder, 'loss_curve.pdf'), dpi=300, bbox_inches='tight')
            plt.close()

            # 绘制全局相对误差下降曲线
            plt.figure(figsize=(10, 6))
            # 修改这里：创建与L2_errors相同长度的epochs数组
            epochs = np.arange(0, len(L2_errors)) * 10  # 每个点代表10个epoch
            plt.semilogy(epochs, L2_errors)
            plt.title('Global Relative L2 Error')
            plt.xlabel('Epochs')
            plt.ylabel('Relative Error')
            plt.grid(True, which="both", ls="--")
            plt.savefig(os.path.join(folder, 'global_error_curve.pdf'), dpi=300, bbox_inches='tight')
            plt.close()

            # 绘制失败概率曲线
            if args.use_fi and len(failure_probs) > 0:
                plt.figure(figsize=(10, 6))
                fi_epochs = np.arange(1, len(failure_probs) + 1) * args.fi_interval
                plt.semilogy(fi_epochs, failure_probs, 'o-')
                plt.axhline(y=args.epsilon_p, color='r', linestyle='--', label='Threshold')
                plt.title('Failure Probability')
                plt.xlabel('Epochs')
                plt.ylabel('Probability')
                plt.legend()
                plt.grid(True, which="both", ls="--")
                plt.savefig(os.path.join(folder, 'failure_prob_curve.pdf'), dpi=300, bbox_inches='tight')
                plt.close()

            # 绘制APINN权重变化曲线
            if args.use_apinn:
                plt.figure(figsize=(12, 8))
                for key in weight_history:
                    if weight_history[key]:
                        plt.plot(range(0, len(weight_history[key]) * args.apinn_interval, args.apinn_interval),
                                 weight_history[key], label=key)
                plt.xlabel('Epochs')
                plt.ylabel('Weight')
                plt.title('APINN Adaptive Weights Evolution')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(folder, 'apinn_weight_evolution.pdf'), dpi=300)
                plt.close()

            # 生成最终解
            x_vals = np.linspace(args.ranges[0][0], args.ranges[0][1], args.number[1])
            y_vals = np.linspace(args.ranges[1][0], args.ranges[1][1], args.number[1])

            # 模型预测
            _, _, solution_on_grid = generate_solution_on_grid(
                model_1, x_test_1, (args.number[1], args.number[1]))

            # 参考解
            ref_solution = u_ref.T

            # 误差
            abs_error = np.abs(solution_on_grid - ref_solution)

            # 相对误差（排除边界）
            inner_mask = (x_ref > args.ranges[0][0] + 1e-6) & (x_ref < args.ranges[0][1] - 1e-6)
            rel_error = np.zeros_like(abs_error)
            # 避免除以零
            ref_abs = np.abs(ref_solution[:, inner_mask]) + 1e-8
            rel_error[:, inner_mask] = abs_error[:, inner_mask] / ref_abs

            # 终止时刻的相对误差计算（L2范数）
            final_time_index = -1  # 最后一个时间步
            # final_time = y_vals[final_time_index]

            # 提取终止时刻的预测解和参考解（排除边界点）
            final_pred = solution_on_grid[final_time_index, inner_mask]  # 模型预测解
            final_ref = ref_solution[final_time_index, inner_mask]  # 参考解

            # 计算相对误差的L2范数
            error = final_pred - final_ref
            l2_error = np.linalg.norm(error) / np.linalg.norm(final_ref)
            print(f"final_time_l2: {l2_error:.4e}")  # 输出示例: final_time_l2: 1.2345e-03

            # 可视化
            plot_heatmap(solution_on_grid, x_vals, y_vals, 'solution.pdf', 'Model Solution')
            plot_heatmap(ref_solution, x_vals, y_vals, 'reference.pdf', 'Reference Solution')
            plot_heatmap(abs_error, x_vals, y_vals, 'absolute_error.pdf', 'Absolute Error', vmin=0)
            plot_relative_error(rel_error, x_vals, y_vals, 'relative_error.pdf')

            # 终止时刻的相对误差可视化
            final_time_index = -1  # 最后一个时间步
            final_time = y_vals[final_time_index]
            final_rel_error = rel_error[final_time_index, :]

            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, final_rel_error)
            plt.title(f'Relative Error at Final Time (t={final_time:.4f})')
            plt.xlabel('x')
            plt.ylabel('Relative Error')
            plt.yscale('log')
            plt.grid(True, which="both", ls="--")
            plt.savefig(os.path.join(folder, f'final_time_error_{final_time:.4f}.pdf'), dpi=300,
                        bbox_inches='tight')
            plt.close()

            # 保存终止时刻的相对误差数据
            final_error_data = np.column_stack((x_vals, final_rel_error))
            np.savetxt(os.path.join(folder, f'final_time_error_{final_time:.4f}.txt'), final_error_data,
                       header='x, relative_error', comments='')

    # 训练结束后保存最终模型
    torch.save(model_1.state_dict(), f'./checkpoints/{filename}_final_model.pth')


############################### 主程序 #####################################
if __name__ == "__main__":
    # 确保模型目录存在
    model_dir = './checkpoints'
    os.makedirs(model_dir, exist_ok=True)

    model_1 = Net().to(device)
    train(model_1)