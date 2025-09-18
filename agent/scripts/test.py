import torch
import DQN_model
import env

n_obs = 35
n_act = 5
steps_done = 1_000_000_000

# загрузка модели, внутрь torch.load() необходимо писать абсолютный путь
model = torch.load('')
policy_net = DQN_model.DQN(n_obs, n_act)
policy_net.load_state_dict(model['net_state_dict'])
policy_net.eval()

# подключение к окружению
envrmnt = env.UniDrive()
a_space = envrmnt.action_space
s_space = envrmnt.observation_space
envrmnt.connection()
while True:
    if envrmnt.conn is not None:
        break

# цикл работы
while True:

    state = envrmnt.reset()
    done = False

    while not done:
        # выбор действия
        action, epsilon, steps_done = DQN_model.select_action(state, policy_net, a_space, steps_done)

        # выполнение действия
        next_state, reward, term, trunk, _ = envrmnt.step(action)

        # маркер конца эпизода
        done = term or trunk
        if done is True:
            done = 1
        else:
            done = 0

        # изменение метрик
        state = next_state
