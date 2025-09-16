import torch
import torch.nn as nn
import torch.nn.functional as f
import random
import math


GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 25000
TAU = 0.005
action_space = []


class DQN(nn.Module):

    def __init__(self, n_observations=35, n_actions=5):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = f.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        x = f.relu(self.layer3(x))
        return self.layer4(x)


def select_action(state, net, a_space, steps_done):
    sample = random.random()
    eps_treshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_treshold:
        with torch.no_grad():
            return (net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).max(1).indices.view(1, 1),
                    eps_treshold, steps_done)
    else:
        action = a_space.sample()
        return torch.tensor([[action]], dtype=torch.long), eps_treshold, steps_done


def optimize_model(target_net, policy_net, memory, BATCH_SIZE, GAMMA, optimizer, TAU):
    # проверка количества опыта
    if len(memory) < BATCH_SIZE:
        return

    # семплируем батч переходов
    transitions = memory.sample(batch_size=BATCH_SIZE)

    # преобразовываем в тензоры
    states = torch.tensor([trans[0] for trans in transitions], dtype=torch.float32, device='cpu')
    actions = torch.tensor([trans[1] for trans in transitions], dtype=torch.int64, device='cpu').unsqueeze(1)
    rewards = torch.tensor([trans[2] for trans in transitions], dtype=torch.float32, device='cpu').unsqueeze(1)
    next_states = torch.tensor([trans[3] for trans in transitions], dtype=torch.float32, device='cpu')
    dones = torch.tensor([trans[4] for trans in transitions], dtype=torch.float32, device='cpu').unsqueeze(1)

    # вычисление Q(s, a) - предсказание policy-сети
    q_values = policy_net(states).gather(1, actions)
    # gather: из каждого вектора Q(outputs) берём ту колонку, что соответствует действию

    # вычисляем target Q-значения
    with torch.no_grad():
        # Q_target(s', a*) = reward + gamma * max_a' Q_target(s', a') * (1 - done)
        q_next = target_net(next_states).max(1)[0].unsqueeze(1)
        q_target = rewards + (1-dones) * GAMMA * q_next

    # расчет лосс функции (MSE)
    loss = f.mse_loss(q_values, q_target)

    # оптимизация policy-сети
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # обновление параметров target-сети (soft update)
    for param, target_param in zip(policy_net.parameters(), target_net.parameters()):
        target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    return loss.item()
