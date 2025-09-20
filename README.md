russian

В этом репозитории представлен проект, в сфере deep RL. на юнити было создано окружение, представляющее из
себя трассу с машиной, управляемой нейронной сетью. Награды реализованы с помощью ревард-чекпоинтов, расставлен-
ных по трассе, также награды выдаются за движение в их направлении.

Окружение возвращает вектор наблюдений, состоящий из 30 рейкаст-дистанций до стен, прямой и угловой скорости,
градус между машиной и чекпоинтом, а также расстояния до него. На маскимальную скорость машины установлено
ограничение.

Собраная версия env находится в этом репозитории, полный юнити проект хранится на гугл диске по ссылке:
https://drive.google.com/drive/folders/18ium_S-1SNj1n2-Kza_LtWpYzhI-5fML?usp=drive_link

в agent/scripts хранится исходный код, связь с env через tcp порт и обертку в архитектуру gymnasium для удобной 
работы, алгоритм dqn, цикл обучения и теста агента.



english

This repository presents a project in the field of deep reinforcement learning (deep RL). A Unity environment was created,
representing a track with a car controlled by a neural network. Rewards are implemented using reward checkpoints placed along
the track, as well as additional rewards for moving in their direction.

The environment returns an observation vector consisting of 30 raycast distances to walls, linear and angular velocity, the
angle between the car and the checkpoint, and the distance to it. A maximum speed limit is applied to the car.

A built version of the environment is included in this repository, while the full Unity project is stored on Google Drive
at the following link:
https://drive.google.com/drive/folders/18ium_S-1SNj1n2-Kza_LtWpYzhI-5fML?usp=drive_link

In agent/scripts, you can find the source code, the TCP communication with the environment, and a Gymnasium wrapper for
convenient integration. It also contains the DQN algorithm, as well as the agent’s training and testing loop..
