import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Bernoulli
from typing import Dict
from config import Config
from models import ActorCritic
from utils import RunningMeanStd


class PPOAgent:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = ActorCritic(config).to(self.device)

        # Param groups: no weight_decay for log_std
        decay_params, no_decay_params = [], []
        for n, p in self.network.named_parameters():
            if 'log_std' in n:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        self.optimizer = optim.AdamW(
            [{'params': decay_params, 'weight_decay': 1e-5},
             {'params': no_decay_params, 'weight_decay': 0.0}],
            lr=config.lr_actor
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.lr_decay)

        self.buffer = []
        self.training_step = 0

        self.reward_stats = RunningMeanStd()
        self.advantage_stats = RunningMeanStd()

    def select_action(self, state: np.ndarray, training: bool=True):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if training:
                action_t, logprob_t = self.network.get_action(s, training=True)
                return action_t.cpu().numpy()[0], float(logprob_t.cpu().numpy()[0])
            else:
                action_t, _ = self.network.get_action(s, training=False)
                return action_t.cpu().numpy()[0], None

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.buffer.append({
            'state': state,
            'action': action,      # [size, bps, post]
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.config.batch_size:
            return {}

        states = torch.tensor([e['state'] for e in self.buffer], dtype=torch.float32, device=self.device)
        actions = torch.tensor([e['action'] for e in self.buffer], dtype=torch.float32, device=self.device)  # [N,3]
        rewards = torch.tensor([e['reward'] for e in self.buffer], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([e['next_state'] for e in self.buffer], dtype=torch.float32, device=self.device)
        dones = torch.tensor([e['done'] for e in self.buffer], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor([e['log_prob'] for e in self.buffer], dtype=torch.float32, device=self.device)

        # Normalize rewards if enabled
        if self.config.use_reward_normalization:
            self.reward_stats.update(rewards.detach().cpu().numpy())
            rewards = (rewards - self.reward_stats.mean) / (self.reward_stats.std + 1e-8)

        # Compute values and next values
        with torch.no_grad():
            cont_mean, cont_std, post_logit, values = self.network(states)
            _, _, _, next_values = self.network(next_states)
            
            # Store old values for value clipping
            old_values = values.clone()

        # Compute returns and advantages using corrected GAE
        returns, advantages = self._compute_gae(rewards, values, next_values, dones)

        # Normalize advantages
        if self.config.use_advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to numpy for shuffling then back to tensors
        dataset = {
            'states': states,
            'actions': actions,
            'returns': returns,
            'advantages': advantages,
            'old_log_probs': old_log_probs,
            'old_values': old_values
        }

        metrics = self._update_policy(dataset)
        
        self.scheduler.step()
        self.buffer.clear()
        self.training_step += 1

        return metrics

    def _compute_gae(self, rewards, values, next_values, dones):
        """Fixed GAE computation"""
        T = len(rewards)
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        gamma, lam = self.config.gamma, self.config.gae_lambda
        
        # Proper next values: shift values by 1, use next_values for last timestep
        next_v = torch.zeros_like(values)
        next_v[:-1] = values[1:]  # Shift values forward
        next_v[-1] = next_values[-1]  # Use actual next value for last step

        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_v[t] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
        return returns, advantages

    def _update_policy(self, dataset):
        """Updated policy learning with proper batching and value clipping"""
        actor_losses, critic_losses, entropies = [], [], []
        kl_divs, clipfracs, value_clipfracs = [], [], []

        N = len(dataset['states'])
        
        for epoch in range(self.config.n_epochs_per_update):
            # Shuffle indices for each epoch
            indices = torch.randperm(N, device=self.device)
            
            for start in range(0, N, self.config.batch_size):
                end = min(start + self.config.batch_size, N)
                batch_indices = indices[start:end]
                
                # Extract batch data
                batch_states = dataset['states'][batch_indices]
                batch_actions = dataset['actions'][batch_indices]
                batch_returns = dataset['returns'][batch_indices]
                batch_advantages = dataset['advantages'][batch_indices]
                batch_old_log_probs = dataset['old_log_probs'][batch_indices]
                batch_old_values = dataset['old_values'][batch_indices]

                # Forward pass
                cont_mean, cont_std, post_logit, values = self.network(batch_states)

                # Compute current log probabilities
                cont_dist = Normal(cont_mean, cont_std)
                cont_log_prob = cont_dist.log_prob(batch_actions[:, :2]).sum(dim=-1)

                bin_dist = Bernoulli(logits=post_logit)
                bin_log_prob = bin_dist.log_prob(batch_actions[:, 2])

                new_log_probs = cont_log_prob + bin_log_prob

                # Compute ratio and surrogate loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                cont_entropy = cont_dist.entropy().sum(dim=-1).mean()
                bin_entropy = bin_dist.entropy().mean()
                entropy = cont_entropy + bin_entropy
                
                # Value function loss with clipping
                if hasattr(self.config, 'use_value_clipping') and self.config.use_value_clipping:
                    values_clipped = batch_old_values + torch.clamp(
                        values - batch_old_values, -self.config.clip_ratio, self.config.clip_ratio
                    )
                    value_loss_unclipped = F.mse_loss(values, batch_returns, reduction='none')
                    value_loss_clipped = F.mse_loss(values_clipped, batch_returns, reduction='none')
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    
                    # Track clipping fraction
                    value_clipfrac = (value_loss_clipped > value_loss_unclipped).float().mean()
                    value_clipfracs.append(value_clipfrac.item())
                else:
                    value_loss = F.mse_loss(values, batch_returns)

                # Total loss
                total_loss = actor_loss + self.config.value_loss_coef * value_loss - self.config.entropy_beta * entropy

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                actor_losses.append(actor_loss.item())
                critic_losses.append(value_loss.item())
                entropies.append(entropy.item())
                
                # Additional metrics for monitoring
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - new_log_probs).mean()
                    clipfrac = (torch.abs(ratio - 1.0) > self.config.clip_ratio).float().mean()
                    kl_divs.append(kl_div.item())
                    clipfracs.append(clipfrac.item())

        # Return comprehensive metrics
        metrics = {
            'actor_loss': float(np.mean(actor_losses)),
            'critic_loss': float(np.mean(critic_losses)),
            'entropy': float(np.mean(entropies)),
            'kl_divergence': float(np.mean(kl_divs)),
            'clip_fraction': float(np.mean(clipfracs)),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        if value_clipfracs:
            metrics['value_clip_fraction'] = float(np.mean(value_clipfracs))
            
        return metrics

    def save(self, path: str):
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'training_step': self.training_step,
            'reward_stats': {'mean': float(self.reward_stats.mean),
                             'std': float(self.reward_stats.std),
                             'count': float(self.reward_stats.count)}
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt['network_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.training_step = ckpt['training_step']
        if 'reward_stats' in ckpt:
            self.reward_stats.mean = ckpt['reward_stats']['mean']
            self.reward_stats.std = ckpt['reward_stats']['std']
            self.reward_stats.count = ckpt['reward_stats']['count']