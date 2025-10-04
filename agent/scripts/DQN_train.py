import os
import torch
from memory import ReplayMemory
import DQN_model
import env
from torch.utils.tensorboard import SummaryWriter
from logger import setup_logger
from datetime import datetime
import torch.optim as optim

# гиперпараметры
n_obs = 35
n_act = 5
num_episodes = 2000
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.95
EPS_END = 0.01
EPS_DECAY = 250_000
epsilon = 0.95
TAU = 0.005
steps_done = 0
device = 'cpu'

# логгеры
tb_dir = os.path.join('../logs', 'tensorboard')
writer = SummaryWriter(log_dir=tb_dir)
datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logger = setup_logger()

# выбор сети для обучения
command = input('какую сеть обучать(old/new)')

if command == 'new':
    # сети
    target_network = DQN_model.DQN(n_obs, n_act)
    policy_network = DQN_model.DQN(n_obs, n_act)
    target_network.load_state_dict(policy_network.state_dict())

    # оптимизатор и память
    memory = ReplayMemory(200_000)
    optimizer = optim.AdamW(policy_network.parameters(), lr=1e-4, amsgrad=True)

elif command == 'old':
    # принимает абсолютный путь к модели
    model = torch.load('C:/ml_project/agent/models/new_models/save_by_episode_150.pth')
    steps_done = 300_000
    # сети
    target_network = DQN_model.DQN(n_obs, n_act)
    policy_network = DQN_model.DQN(n_obs, n_act)

    policy_network.load_state_dict(model['net_state_dict'])
    target_network.load_state_dict(policy_network.state_dict())

    # оптимизатор и память
    optimizer = optim.AdamW(policy_network.parameters(), lr=1e-4, amsgrad=True)
    optimizer.load_state_dict(model['optimizer_state_dict'])
    memory = model['memory']


# окружение
envrmnt = env.UniDrive()
a_space = envrmnt.action_space
s_space = envrmnt.observation_space
envrmnt.connection()
while True:
    if envrmnt.conn is not None:
        break

# цикл обучения
for episode in range(1, num_episodes + 1):

    # перезагрузка эпизода
    state = envrmnt.reset()

    done = False
    total_reward = 0

    while not done:
        # выбор действия
        action, epsilon, steps_done = DQN_model.select_action(state, policy_network, a_space, steps_done)
        print(state)

        # выполнение действия
        next_state, reward, term, trunk, _ = envrmnt.step(action)

        # маркер конца эпизода
        done = term or trunk
        if done is True:
            done = 1
        else:
            done = 0
        memory.push(state, action, reward, next_state, done)

        # обновление модели
        loss = DQN_model.optimize_model(target_net=target_network, policy_net=policy_network, memory=memory,
                                        BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, optimizer=optimizer, TAU=TAU)

        # логирование по шагу
        if loss is not None:
            writer.add_scalar('Loss/train', loss, steps_done)
            writer.add_scalar('total_reward/train', total_reward, steps_done)
            writer.add_scalar('reward/train', reward, steps_done)

        # изменение метрик
        state = next_state
        total_reward += reward

    # логирование по эпизоду
    writer.add_scalar('Reward_for_episode/train', total_reward, episode)
    writer.add_scalar('Epsilon', epsilon, episode)
    logger.info(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    # сохранение чекпойнта
    if episode % 50 == 0:

        torch.save(
            {
                'net_state_dict': policy_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'steps_done': steps_done,
                'memory': memory
            },
            f"C:/ml_project/agent/models/checkpoints_pNet/save_by_episode_"f"{episode}.pth"
        )

        print(f"✔ Модель сохранена на {episode} эпизоде")

# финальное сохранение
os.makedirs('../models', exist_ok=True)
torch.save({
                'net_state_dict': policy_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'steps_done': steps_done,
                'memory': memory
            }, "C:/ml_project/agent/models/new_models/new_final_policy_net.pth")
print("✔ Обучение завершено. Финальная модель сохранена.")

writer.close()
envrmnt.close()
