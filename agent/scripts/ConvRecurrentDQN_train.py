import os
import torch
from Sequence_replay_buffer import SequenceReplayBuffer
import ConvRecurrentDQN
import env
from torch.utils.tensorboard import SummaryWriter
from logger import setup_logger
from datetime import datetime
import torch.optim as optim

# гиперпараметры
n_rays = 30
n_aux = 5
n_act = 5
num_episodes = 1000
B = 8
S = 16
current_B = 0
current_S = 0
gamma = 0.99
tau = 0.005
eps_start = 0.95
eps_end = 0.01
eps_decay = 25_000
epsilon = 0.95
steps_done = 0
device = 'cpu'

# логгеры
tb_dir = os.path.join('../logs', 'tensorboard')
writer = SummaryWriter(log_dir=tb_dir)
datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logger = setup_logger()

target_network = ConvRecurrentDQN.ConvRecurrentDQN()
policy_network = ConvRecurrentDQN.ConvRecurrentDQN()
target_network.load_state_dict(policy_network.state_dict())

memory = SequenceReplayBuffer(200_000)
optimizer = optim.AdamW(policy_network.parameters(), lr=1e-4, amsgrad=True)

# окружение
envrmnt = env.UniDrive()
a_space = envrmnt.action_space
s_space = envrmnt.observation_space
envrmnt.connection()
while True:
    if envrmnt.conn is not None:
        break

for episode in range(1, num_episodes + 1):
    # начало нового эпизода (env), создание эпизода в памяти
    state = envrmnt.reset()
    memory.start_episode()

    done = False
    total_reward = 0

    while not done:
        # выбор действия
        action, hidden, epsilon, steps_done = ConvRecurrentDQN.select_action(
            state=state, hidden=None, net=policy_network, action_space=a_space, steps_done=steps_done, EPS_END=eps_end,
            EPS_START=eps_start, EPS_DECAY=eps_decay)

        print(current_S, current_B, epsilon)

        # шаг окружения
        next_state, reward, term, trunc, _ = envrmnt.step(action)

        done = term or trunc
        if done is True:
            done = 1
        else:
            done = 0

        # добавление в буфер перехода
        memory.push(state, action, reward, next_state, done)
        current_S += 1

        # добавление эпизода в память, если его длинна достаточна
        if current_S == S:
            current_S = 0
            memory.finish_episode()
            memory.start_episode()

        current_B = memory.num_episodes()

        # обновление модели, если в памяти набран батч нужного размера (B эпизодов)
        if current_B >= B:
            batch = memory.sample(B, S)
            loss = ConvRecurrentDQN.optimize_model(policy_network, target_network, batch, gamma, tau, device, optimizer)

            # логирование по шагу
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
