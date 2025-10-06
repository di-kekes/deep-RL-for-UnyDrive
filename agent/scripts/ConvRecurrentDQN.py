import torch
import torch.nn as nn
import torch.nn.functional as f


class ConvRecurrentDQN(nn.Module):
    def __init__(self, n_rays=30, n_aux=5, n_actions=5, lstm_hidden=128):
        super().__init__()
        # conv for raycasts
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU()
        )

        # comp conv output size
        conv_out = 64 * n_rays

        # scalar mlp
        self.scalar_net = nn.Sequential(
            nn.Linear(n_aux, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # LSTM input size = conv_out + scalar_feature_size
        self.lstm_input_size = conv_out + 64
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=lstm_hidden, batch_first=True)

        # head = compute q values
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, states_seq, hidden=None):
        # states_seq: (B, S, obs_dim) where obs_dim = n_rays + n_aux
        B, S, obs_dim = states_seq.shape
        n_rays = obs_dim - 5
        rays = states_seq[:, :, :n_rays].contiguous().view(B*S, 1, n_rays)
        aux = states_seq[:, :, n_rays:].contiguous().view(B*S, 5)

        conv_feat = self.conv(rays)             # (B*S, 64, n_rays)
        conv_feat = conv_feat.view(B*S, -1)     # (B*S, 64*n_rays)
        scalar_feat = self.scalar_net(aux)      # (B*S, 64)

        x = torch.cat([conv_feat, scalar_feat], dim=1)  # (B*S, lstm_input_size)
        x = x.view(B, S, -1)                                   # (B, S, lstm_input_size)

        lstm_out, hidden = self.lstm(x, hidden)     # lstm_out: (B, S, hidden)
        last = lstm_out[:, -1, :]                   # (B, hidden)
        q = self.head(last)                         # (B, n_actions)
        return q, hidden

    def optimize_model(self, policy_net, target_net, batch, gamma, device, optimizer):
        """
        compute loss
        optimize model
        """
        states, actions, rewards, next_states, dones = batch
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        B, S, obs = states.shape

        # Predicted Q for last timestep from policy_net (use full sequence to get hidden)
        q_pred, _ = policy_net(states)      # (B, n_actions)

        # actions for last step:
        actions_last = actions[:, -1]       # (B)
        q_pred_a = q_pred.gather(1, actions_last.unsqueeze(1)).squeeze(1)      # (B)

        # расчет таргета:
        next_state_last = next_states[:, -1, :].unsqueeze(1)

        # выбор действия с помощью double DQN
        with torch.no_grad():
            # выбор policy сетью
            q_next_policy, _ = policy_net(next_state_last)
            next_actions = q_next_policy.argmax(dim=1, keepdim=True)

            # оценка target сетью
            q_next_target, _ = target_net(next_state_last)
            q_target_next_val = q_next_target.gather(1, next_actions).squeeze(1)

            reward_last = rewards[:, -1]
            done_last = dones[:, -1]
            targets = reward_last + gamma * (1.0 - done_last) * q_target_next_val

        loss = f.smooth_l1_loss(q_pred_a, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
        optimizer.step()

        return loss.item()
