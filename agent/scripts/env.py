from typing import Optional
import gymnasium as gym
import socket
import json
import numpy as np
from gymnasium import spaces


class UniDrive(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, host='127.0.0.1', port=5005):
        super(UniDrive, self).__init__()

        # инициализация сокета
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.host = host
        self.port = port
        self.conn = None
        self.buffer = b""
        self.ID_resp = 0

        # пространство состояний и действий
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(35,), dtype=np.float32)

    # подключение к юнити
    def connection(self):
        self.sock.listen(1)
        print(f"Listening on {self.host}:{self.port}")
        conn, addr = self.sock.accept()
        print(f"Connected by {addr}")
        self.conn = conn

    # сброс среды
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,

    ):
        while True:
            self.conn.sendall((json.dumps({'command': 'reset', 'action': 10, 'ID': self.ID_resp})+'\n').encode('utf-8'))
            data = self.conn.recv(4096)
            self.buffer += data
            line, self.buffer = self.buffer.split(b'\n', 1)
            data = json.loads(line)

            state = np.array(data['state'], dtype=np.float32)
            ID_req = int(data['ID'])

            if self.ID_resp != ID_req:
                continue

            else:
                if self.ID_resp == 1001:
                    self.ID_resp = 0
                else:
                    self.ID_resp += 1
                return state

    # отправка действия
    def step(self, action):
        while True:
            self.conn.sendall((json.dumps({'command': 'step', 'action': int(action),
                                           'ID': self.ID_resp})+'\n').encode('utf-8'))

            data = self.conn.recv(4096)
            self.buffer += data
            line, self.buffer = self.buffer.split(b'\n', 1)
            data = json.loads(line)

            obs = np.array(data['state'], dtype=np.float32)
            reward = float(data['reward'])
            term = bool(data['terminated'])
            trunc = bool(data['truncated'])
            ID_req = int(data['ID'])
            info = data.get('info', {})

            if self.ID_resp != ID_req:
                continue

            else:

                if self.ID_resp == 1001:
                    self.ID_resp = 0
                else:
                    self.ID_resp += 1

                return obs, reward, term, trunc, info

    # закрытие среды, конец эпизода
    def close(self):
        self.sock.close()
