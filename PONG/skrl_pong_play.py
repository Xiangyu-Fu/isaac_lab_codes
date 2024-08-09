import torch
import torch.nn as nn

# 导入构建强化学习系统的skrl组件
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# 设置随机种子以确保结果的可重复性
set_seed(42)  # 例如使用 `set_seed(42)` 来设置固定的随机种子


# 定义共享模型（包括随机和确定性模型）使用混合类
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        # 定义神经网络结构
        self.net = nn.Sequential(nn.Linear(self.num_observations, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 128),
                                 nn.ELU())

        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(128, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}



# 加载并包装Isaac Lab环境
env = load_isaaclab_env(task_name="Isaac-Pong-Flat-GO-1-Play-v0", num_envs=4)
env = wrap_env(env)

device = env.device
print(f"Device: {device}")


# 实例化一个记忆模块作为回滚缓冲区（可以使用任何记忆模块）
memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)


# 实例化智能体的模型（函数逼近器）。
# PPO需要两个模型，详见文档
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # 使用相同实例：共享模型


# 配置并实例化智能体（详见文档以获取所有选项）
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 24  # memory_size
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 4  # 24 * 4096 / 24576
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = False
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# 将日志记录到TensorBoard并写入检查点（以时间步为单位）
cfg["experiment"]["write_interval"] = 60
cfg["experiment"]["checkpoint_interval"] = 500
cfg["experiment"]["directory"] = "runs/torch/Isaac-Unitree"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# 配置并实例化RL训练器
cfg_trainer = {"timesteps": 10000, "headless": False}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# # 开始训练
# trainer.train()


# 训练完成后，加载训练好的模型
agent.load("runs/torch/Isaac-Unitree/24-08-09_10-52-46-273937_PPO/checkpoints/best_agent.pt")

# 设置智能体为测试模式
agent.set_mode("eval")

# 运行并测试智能体
states = env.reset()
states = states[0].cpu().numpy()
total_reward = 0
timestep = 0  # 初始化时间步

for _ in range(1000):  # 运行1000个时间步
    actions = agent.act(torch.tensor(states).cuda(0), timestep, 1000)
    next_states, rewards, _, dones, infos = env.step(actions[0])
    states = next_states.cpu().numpy()
    total_reward += rewards.sum().item()

    # 可视化环境
    env.render()

    timestep += 1  # 增加时间步

    if dones.all():
        break

print(f"Total reward: {total_reward}")

# 运行多个回合并评估智能体的表现
num_episodes = 10
total_rewards = []

for episode in range(num_episodes):
    states = env.reset()
    episode_reward = 0
    timestep = 0  # 每个回合重置时间步

    while True:
        actions, _ = agent.act(states, timestep, 1000)
        next_states, rewards, dones, infos = env.step(actions)
        states = next_states
        episode_reward += rewards.sum().item()

        timestep += 1  # 增加时间步

        if dones.all():
            break

    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Reward = {episode_reward}")

average_reward = sum(total_rewards) / num_episodes
print(f"Average Reward over {num_episodes} episodes: {average_reward}")
