import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy.stats import multivariate_normal, truncnorm
import scipy

# Set random seeds
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

###################### Define Hyperparameters ##################
parser = argparse.ArgumentParser()
parser.add_argument('--e', type=int, default=701, help='Epochs')
parser.add_argument('--number', type=list, default=[2500, 200], help='内点，边界取点数')
parser.add_argument('--ranges', type=list, default=[[-1, 1], [0, 0.5]], help='空间，时间范围')
parser.add_argument('--beta', type=list, default=[0.04, 1], help='coefficient')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--net', type=list, default=[2, 50, 50, 50, 2], help='网络架构')
parser.add_argument('--filename', type=str, default='ATR-BPINN_0dot16664_0dot1_10_5_2', help='保存数据文件名')
parser.add_argument('--resample', type=int, default=0, choices=[0, 1], help='是否启用重采样 (0:禁用, 1:启用)')
parser.add_argument('--use_time_loss', type=int, default=0, choices=[0, 1],
                    help='是否使用时间离散损失 (0:禁用, 1:启用)')
parser.add_argument('--use_weight', type=int, default=0, choices=[0, 1], help='是否使用自适应权重(0:禁用, 1:启用)')
parser.add_argument('--weight_p', type=float, default=1.0, help='加权指数 p')
# 添加损失项的权重参数
parser.add_argument('--w_pde', type=float, default=0.1, help='PDE损失权重')
parser.add_argument('--w_bc', type=float, default=10, help='边界损失权重')
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
parser.add_argument('--apinn_alpha', type=float, default=9,
                    help='APINN权重调整系数α')
parser.add_argument('--apinn_interval', type=int, default=50,
                    help='APINN权重更新间隔(epoch)')
parser.add_argument('--apinn_window', type=int, default=50,
                    help='APINN计算平均损失的窗口大小')
parser.add_argument('--gamma', type=float, default=5000.0, help='Gradient threshold for blow-up detection')
parser.add_argument('--epsilon_t', type=float, default=1e-6, help='Convergence tolerance for blow-up time')
parser.add_argument('--atr_max_iter', type=int, default=20, help='Max iterations for ATR algorithm')
parser.add_argument('--inner_epochs', type=int, default=1000, help='Epochs for inner training loop')

args = parser.parse_args()

# Global parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
delta = 3
filename = args.filename


######################### Network Definition #########################
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


############################# Function Definitions ######################
# Existing functions remain the same
bd_fun = lambda x: 1
source_fun_1 = lambda x: 0
initial_fun = lambda x: torch.exp(-torch.cos(torch.pi * x[:, 0:1] / 2))


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
    gU_t = grad(gU, X)[:, 1:2]

    if domain == 'domain_1':
        output = delta + U_xx / torch.exp(U) + gU_t - U_x / torch.exp(U)  # 格式2
    else:
        output = args.beta[1] * (U_tt - U_xx) + U
    return output


def group_time_weight_loss(inputs, residuals, p=1.0):
    # Existing function
    pass


############################### Sampling Functions #####################################
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
            x_b = torch.cat([xb1, xb2], dim=0).to(device).requires_grad_(True)
            x_ini = torch.stack([X[:, 0], Y[:, 0]]).T.to(device).requires_grad_(True)  # t=0
        else:
            x_b = torch.cat([xb2, torch.stack([X[:, 0], Y[:, 0]]).T,
                             torch.stack([X[:, -1], Y[:, -1]]).T], dim=0).to(device)
            x_ini = torch.stack([X[:, 0], Y[:, 0]]).T.to(device)
    else:
        coor = torch.tensor([[ranges[0][0], ranges[1][0]], [ranges[0][1], ranges[1][1]]]).to(device)
        x_i = coor[0] + (coor[1] - coor[0]) * torch.rand(args.number[0], 2, device=device).requires_grad_(True)

        # 生成边界点
        xb1 = torch.stack([X[0], Y[0]]).T.to(device)  # x_min
        xb2 = torch.stack([X[-1], Y[0]]).T.to(device)  # x_max

        if domain == 'domain_1':
            x_b = torch.cat([xb1, xb2], dim=0).to(device).requires_grad_(True)
            x_ini = torch.stack([X[:, 0], Y[:, 0]]).T.to(device).requires_grad_(True)  # t=0
        else:
            x_b = torch.cat([xb2, torch.stack([X[:, 0], Y[:, 0]]).T,
                             torch.stack([X[:, -1], Y[:, -1]]).T], dim=0).to(device)
            x_ini = torch.stack([X[:, 0], Y[:, 0]]).T.to(device)

    return x_i, x_b, x_ini


def generate_solution_on_grid(model, x_test, grid_shape):
    model.eval()
    g_pred = model(x_test)[:, 1:2]
    u_pred = -torch.log(torch.abs(g_pred))

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



############################### Training Function with ATR #####################################
def train(model_1):
    # Initialize ATR parameters
    T_initial = args.ranges[1][1]  # Initial time domain [t0, T]
    t_b_estimate = T_initial  # Initial blow-up time estimate
    convergence_tolerance = args.epsilon_t
    max_atr_iterations = args.atr_max_iter


    # ATR outer loop
    for atr_iter in range(max_atr_iterations):
        print(f"ATR Iteration {atr_iter + 1}/{max_atr_iterations}")
        print(f"Current time domain: [0, {t_b_estimate:.6f}]")

        # Update time range for current ATR iteration
        current_ranges = [args.ranges[0], [args.ranges[1][0], t_b_estimate]]
        gamma_threshold = args.ranges[1][1] / current_ranges[1][1] * args.gamma
        # gamma_threshold = args.gamma
        print(gamma_threshold)

        # Initialize optimizer for inner training
        use_adam = False
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

        # Generate data for current time domain
        x_train_1, x_b_1, x_ini = grow_data(current_ranges, method='random', domain='domain_1')
        x_test_1, _, _ = grow_data(current_ranges, method='mesh', domain='domain_1')

        # Store original training points
        original_x_train_1 = x_train_1.detach().clone()
        current_x_train_1 = original_x_train_1.clone().requires_grad_(True)
        all_train_points = [current_x_train_1.detach().clone()]

        # Prepare reference solution for current time domain
        x_ref = np.linspace(current_ranges[0][0], current_ranges[0][1], args.number[1])
        t_eval = np.linspace(current_ranges[1][0], current_ranges[1][1], args.number[1])
        dx = x_ref[1] - x_ref[0]

        # Initial condition (excluding boundaries)
        u0_flat = np.cos(np.pi * x_ref[1:-1] / 2)

        # def ode_rhs(t, u_flat):
        #     u = np.zeros_like(x_ref)
        #     u[1:-1] = u_flat
        #     u_xx = np.zeros_like(u)
        #     u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
        #     u_x = np.zeros_like(u)
        #     u_x[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        #     du_dt = -u_x + u_xx + 3 * np.exp(u)
        #     return du_dt[1:-1]

        # sol = solve_ivp(ode_rhs, [0, t_b_estimate], u0_flat, t_eval=t_eval, method='Radau')

        # Reconstruct reference solution
        # u_ref = np.zeros((len(x_ref), len(t_eval)))
        # u_ref[1:-1, :] = sol.y

        # Store variables for inner training
        Loss_1 = []
        L2_errors = []
        failure_probs = []
        start_time = time.time()

        # Inner training loop
        for epoch in range(args.inner_epochs):
            def closure():
                optimizer.zero_grad()

                # Use current training points
                u_1 = model_1(current_x_train_1)[:, 0:1]
                gu_1 = model_1(current_x_train_1)[:, 1:2]
                lap_1 = Operator(u_1, gu_1, current_x_train_1, domain='domain_1')
                source_1 = source_fun_1(current_x_train_1)
                residuals = lap_1 - source_1

                if args.use_weight == 1:
                    pde_loss_1 = group_time_weight_loss(current_x_train_1, residuals, p=args.weight_p)
                else:
                    pde_loss_1 = torch.mean(residuals ** 2)

                # Boundary loss
                bc_pred = model_1(x_b_1)[:, 1:2]
                bc_loss_1 = torch.mean((bc_pred - 1) ** 2)

                # Initial condition loss
                u_ini_pred = model_1(x_ini)[:, 1:2]
                ext_ini = initial_fun(x_ini)
                loss_ini = torch.mean((u_ini_pred - ext_ini) ** 2)

                # Coupling term loss
                loss_g = torch.mean((gu_1 - torch.exp(-u_1)) ** 2)

                # Time discretization loss
                loss_time = torch.tensor(0.0, device=device)
                if args.use_time_loss and epoch > 10:
                    nearest_points, time_diffs = find_nearest_timestep(current_x_train_1, current_x_train_1)
                    valid_points = []

                    for i, (point, nearest, h) in enumerate(zip(current_x_train_1, nearest_points, time_diffs)):
                        if nearest is not None and h is not None and h > 1e-6:
                            u_current = u_1[i]
                            gu_current = gu_1[i]
                            u_x = grad(u_current, current_x_train_1)[i, 0:1]
                            u_xx = grad(u_x, current_x_train_1)[i, 0:1]
                            nearest_output = model_1(nearest.unsqueeze(0))
                            gu_nearest = nearest_output[:, 1:2]
                            F_u = torch.exp(u_current)
                            constraint = gu_current - gu_nearest + delta * h + (h * u_xx) / F_u
                            loss_time += torch.mean(constraint ** 2)
                            valid_points.append(1)

                    if valid_points:
                        loss_time = loss_time / len(valid_points)
                    else:
                        loss_time = torch.tensor(0.0, device=device)

                # Combined loss
                loss = (args.w_pde * pde_loss_1 + args.w_bc * bc_loss_1 +
                        args.w_g * loss_g + args.w_ini * loss_ini + args.w_time * loss_time)
                loss.backward(retain_graph=True)
                return loss

            optimizer.step(closure)

            # FI-PINNs adaptive sampling (if enabled)
            if args.use_fi and epoch % args.fi_interval == 0 and epoch > 0:
                failure_prob, new_points, proposal_mean, proposal_cov = sais_algorithm(
                    model_1,
                    current_ranges,
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

                if failure_prob > args.epsilon_p and len(new_points) > 0:
                    updated_train_points = torch.cat([
                        current_x_train_1.detach().clone(),
                        new_points.requires_grad_(True)
                    ], dim=0).requires_grad_(True)

                    current_x_train_1 = updated_train_points
                    all_train_points.append(updated_train_points.detach().clone())
                    print(f"Epoch {epoch}: Added {len(new_points)} new points, total points: {len(current_x_train_1)}")

            # Logging and evaluation (every 10 epochs)
            if epoch % 10 == 0:
                # Calculate L2 error
                _, _, solution_on_grid = generate_solution_on_grid(
                    model_1, x_test_1, (args.number[1], args.number[1]))

                inner_mask = (x_ref > current_ranges[0][0] + 1e-6) & (x_ref < current_ranges[0][1] - 1e-6)

                # if np.linalg.norm(u_ref[inner_mask, :].T) > 1e-8:
                #     error = solution_on_grid[:, inner_mask] - u_ref[inner_mask, :].T
                #     rel_l2_error = np.linalg.norm(error) / np.linalg.norm(u_ref[inner_mask, :].T)
                # else:
                #     rel_l2_error = np.nan
                # L2_errors.append(rel_l2_error)

                t = time.time() - start_time
                start_time = time.time()
                # print(f"Epoch:{epoch}, Time: {t:.2f}s, L2 Error: {rel_l2_error:.4e}")

        # After inner training, detect blow-up
        print("Detecting blow-up...")
        model_1.eval()

        # Create a dense grid for evaluation
        x_eval = torch.linspace(current_ranges[0][0], current_ranges[0][1], steps=100, device=device)
        t_eval = torch.linspace(current_ranges[1][0], current_ranges[1][1], steps=100, device=device)
        X, T = torch.meshgrid(x_eval, t_eval, indexing="ij")
        eval_points = torch.stack([X.reshape(-1), T.reshape(-1)]).T.requires_grad_(True)

        # Calculate solution and time gradient
        u_pred_g = model_1(eval_points)[:, 1:2]
        g_pred_u = -torch.log(torch.abs(u_pred_g))
        u_t = grad(g_pred_u, eval_points)[:, 1:2]

        # Find points where time gradient exceeds threshold
        blow_up_mask = (torch.abs(u_t) > gamma_threshold).squeeze()
        blow_up_points = eval_points[blow_up_mask]

        if len(blow_up_points) > 0:
            # Find the minimum time where blow-up occurs
            new_t_b = torch.min(blow_up_points[:, 1]).item()
            print(f"Blow-up detected at t = {new_t_b:.6f}")
        else:
            new_t_b = t_b_estimate
            print("No blow-up detected in current domain")

        # Check for convergence
        if abs(new_t_b - t_b_estimate) < convergence_tolerance:
            print(f"ATR converged at iteration {atr_iter + 1}")
            print(f"Final blow-up time estimate: {new_t_b:.6f}")
            break
        else:
            t_b_estimate = new_t_b
            print(f"Updated blow-up time estimate: {t_b_estimate:.6f}")

            # If we've reached the minimum time, break
            if t_b_estimate <= current_ranges[1][0] + 1e-6:
                print("Reached minimum time limit")
                break

    # Save final model
    torch.save(model_1.state_dict(), f'./checkpoints/{filename}_final_model.pth')
    return t_b_estimate


############################### Main Program #####################################
if __name__ == "__main__":
    # Ensure model directory exists
    model_dir = './checkpoints'
    os.makedirs(model_dir, exist_ok=True)

    model_1 = Net().to(device)
    blow_up_time = train(model_1)
    print(f"Estimated blow-up time: {blow_up_time:.6f}")