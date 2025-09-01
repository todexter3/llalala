import math
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli
from config import Config


class ActorCritic(nn.Module):
    """
    Shared trunk -> (actor_continuous 2D mean, actor_log_std[2]) + (actor_binary logit 1D), critic V(s)
    - continuous dims: [size_ratio(0..1 via sigmoid), price_offset_bps (-max..max via tanh*max)]
    - binary dim: post_only via Bernoulli(logit)
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.activation = nn.LeakyReLU(0.1) if config.activation=='leaky_relu' else (nn.ELU() if config.activation=='elu' else nn.ReLU())

        self.feature_extractor = self._build_mlp(config.state_dim, config.hidden_dim, config.n_layers-1)

        # continuous heads (means)
        self.mean_size = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim//2), self.activation, nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim//2, 1), nn.Sigmoid()    # 0..1
        )
        self.mean_bps = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim//2), self.activation, nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim//2, 1), nn.Tanh()       # -1..1 -> scaled
        )
        # binary head (logit)
        self.post_logit = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim//2), self.activation, nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim//2, 1)                  # logits
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim//2), self.activation, nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim//2, 1)
        )

        # log_std as parameters for 2 continuous dims (size, bps) - stored in sensible scale
        # we parameterize in the native units of outputs (size in 0..1; bps in [-max,max])
        self.log_std = nn.Parameter(torch.tensor([
            math.log(self.config.initial_std_size),
            math.log(self.config.initial_std_bps)
        ], dtype=torch.float32))

        self._init_weights()

    def _build_mlp(self, inp, hid, layers):
        L = [nn.Linear(inp, hid), self.activation, nn.Dropout(self.config.dropout)]
        for _ in range(layers-1):
            L += [nn.Linear(hid, hid), self.activation, nn.Dropout(self.config.dropout)]
        return nn.Sequential(*L)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor):
        z = self.feature_extractor(state)
        # means
        size_mean = self.mean_size(z)                                 # in (0,1)
        bps_mean = self.mean_bps(z) * self.config.max_price_offset_bps  # in [-max_bps, max_bps]
        cont_mean = torch.cat([size_mean, bps_mean], dim=-1)          # [B,2]

        # stds (positive)
        cont_std = torch.exp(torch.clamp(self.log_std, -20, 5))       # shape [2]
        cont_std = cont_std.unsqueeze(0).expand_as(cont_mean)         # [B,2]

        # binary logit
        post_logit = self.post_logit(z).squeeze(-1)                   # [B]

        # value
        value = self.critic(z).squeeze(-1)                            # [B]
        # print(f"cont_mean: {cont_mean.shape}, cont_std: {cont_std.shape}, post_logit: {post_logit.shape}, value: {value.shape}")
        return cont_mean, cont_std, post_logit, value

    def get_action(self, state: torch.Tensor, training: bool=True):
        cont_mean, cont_std, post_logit, value = self.forward(state)

        if training:
            cont_dist = Normal(cont_mean, cont_std)
            cont_action = cont_dist.sample()                          # [B,2]
            cont_logprob = cont_dist.log_prob(cont_action).sum(dim=-1)

            bin_dist = Bernoulli(logits=post_logit)
            bin_action = bin_dist.sample()                            # [B]
            bin_logprob = bin_dist.log_prob(bin_action)

            action = torch.cat([cont_action, bin_action.unsqueeze(-1)], dim=-1)  # [B,3]
            log_prob = cont_logprob + bin_logprob
            return action, log_prob
        else:
            # deterministic: means for continuous, 1*(sigmoid>=0.5) for binary
            bin_prob = torch.sigmoid(post_logit)
            bin_act = (bin_prob >= 0.5).float().unsqueeze(-1)
            action = torch.cat([cont_mean, bin_act], dim=-1)
            return action, None