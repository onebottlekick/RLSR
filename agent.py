import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, padding=3, dilation=3)
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4)

        # pi network
        self.conv_5_pi = nn.Conv2d(64, 64, kernel_size=3, padding=3, dilation=3)
        self.conv_6_pi = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.W_xr = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.W_hr = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.W_xz = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.W_hz = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.W_xh = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.W_hh = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        # Should not apply softmax here because of vanishing values caused by log(softmax)
        # Use softmax when calculate prob, use log_softmax in MyEntropy instead of log(softmax)
        self.conv_7_pi = nn.Conv2d(64, n_actions, kernel_size=3, padding=1)

        # v network
        self.conv_5_v = nn.Conv2d(64, 64, kernel_size=3, padding=3, dilation=3)
        self.conv_6_v = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.conv_7_v = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # reward map convolution
        self.conv_R = nn.Conv2d(1, 1, kernel_size=33, padding=16, bias=False)

    def pi_and_v(self, X_in):
        X = X_in[:, 0:3, :, :]
        X = F.relu(self.conv_1(X))
        X = F.relu(self.conv_2(X))
        X = F.relu(self.conv_3(X))
        X = F.relu(self.conv_4(X))

        # pi network
        X_t = F.relu(self.conv_5_pi(X))
        X_t = F.relu(self.conv_6_pi(X_t))

        # ConvGRU
        H_t1 = X_in[:, -64:, :, :]
        R_t = torch.sigmoid(self.W_xr(X_t) + self.W_hr(H_t1))
        Z_t = torch.sigmoid(self.W_xz(X_t) + self.W_hz(H_t1))
        H_tilde_t = torch.tanh(self.W_xh(X_t) + self.W_hh(R_t * H_t1))
        H_t = Z_t * H_t1 + (1 - Z_t) * H_tilde_t

        pi = self.conv_7_pi(H_t)

        # v network
        X_v = F.relu(self.conv_5_v(X))
        X_v = F.relu(self.conv_6_v(X_v))
        v = self.conv_7_v(X_v)

        return pi, v, H_t

    def conv_smooth(self, v):
        return self.conv_R(v)

    def choose_best_actions(self, state):
        pi, v, inner_state = self.pi_and_v(state)
        actions_prob = torch.softmax(pi, dim=1)
        actions = torch.argmax(actions_prob, dim=1)
        return actions, v, inner_state