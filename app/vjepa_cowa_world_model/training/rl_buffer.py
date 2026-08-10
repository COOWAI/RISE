from dataclasses import dataclass
from typing import Dict, List

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    bootstrap_values: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    """Compute truncation-aware GAE(lambda) advantages and returns."""
    if values.shape != bootstrap_values.shape:
        raise ValueError(
            f"values and bootstrap_values must have the same shape, got {values.shape} vs {bootstrap_values.shape}"
        )
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(values[0])
    terminated = terminated.to(dtype=values.dtype, device=values.device)
    truncated = truncated.to(dtype=values.dtype, device=values.device)
    while terminated.ndim < values.ndim:
        terminated = terminated.unsqueeze(-1)
    while truncated.ndim < values.ndim:
        truncated = truncated.unsqueeze(-1)

    for step in reversed(range(steps)):
        terminal_bootstrap_mask = 1.0 - terminated[step]
        done_recursion_mask = 1.0 - torch.maximum(terminated[step], truncated[step])
        delta = rewards[step] + gamma * bootstrap_values[step] * terminal_bootstrap_mask - values[step]
        gae = delta + gamma * gae_lambda * done_recursion_mask * gae
        advantages[step] = gae

    returns = advantages + values
    return advantages, returns


@dataclass
class RolloutBatch:
    observations: Dict[str, torch.Tensor]
    actions: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class RolloutBuffer:
    def __init__(self, gamma=0.99, gae_lambda=0.95, normalize_advantage=True):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantage = normalize_advantage
        self.reset()

    def reset(self):
        self.observations: List[Dict[str, torch.Tensor]] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.terminated: List[torch.Tensor] = []
        self.truncated: List[torch.Tensor] = []
        self.truncation_bootstrap_values: List[torch.Tensor] = []

    @staticmethod
    def _to_storage_tensor(tensor):
        return tensor.detach().cpu()

    def _to_storage_observation(self, observation):
        return {key: self._to_storage_tensor(value) for key, value in observation.items()}

    def add(self, observation, action, log_prob, value, reward, terminated, truncated, truncation_bootstrap_value):
        self.observations.append(self._to_storage_observation(observation))
        self.actions.append(self._to_storage_tensor(action))
        self.log_probs.append(self._to_storage_tensor(log_prob))
        self.values.append(self._to_storage_tensor(value))
        self.rewards.append(self._to_storage_tensor(reward))
        self.terminated.append(self._to_storage_tensor(terminated))
        self.truncated.append(self._to_storage_tensor(truncated))
        self.truncation_bootstrap_values.append(self._to_storage_tensor(truncation_bootstrap_value))

    def __len__(self):
        return len(self.actions)

    def _stack_observations(self):
        if not self.observations:
            raise ValueError("RolloutBuffer is empty")
        keys = self.observations[0].keys()
        return {key: torch.stack([obs[key] for obs in self.observations], dim=0) for key in keys}

    def finalize(self, last_value):
        rewards = torch.stack(self.rewards, dim=0)
        values = torch.stack(self.values, dim=0)
        terminated = torch.stack(self.terminated, dim=0)
        truncated = torch.stack(self.truncated, dim=0)
        truncation_bootstrap_values = torch.stack(self.truncation_bootstrap_values, dim=0).to(dtype=values.dtype)
        final_value = last_value.detach().cpu().to(dtype=values.dtype)
        if final_value.shape != values[-1].shape:
            if final_value.numel() != values[-1].numel():
                raise ValueError(f"last_value shape {final_value.shape} does not match value shape {values[-1].shape}")
            final_value = final_value.reshape_as(values[-1])
        bootstrap_values = torch.empty_like(values)
        if values.shape[0] > 1:
            bootstrap_values[:-1] = values[1:]
        bootstrap_values[-1] = final_value
        truncation_mask = truncated.bool()
        while truncation_mask.ndim < bootstrap_values.ndim:
            truncation_mask = truncation_mask.unsqueeze(-1)
        bootstrap_values = torch.where(truncation_mask, truncation_bootstrap_values, bootstrap_values)
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            terminated=terminated,
            truncated=truncated,
            bootstrap_values=bootstrap_values,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)

        batch = RolloutBatch(
            observations=self._stack_observations(),
            actions=torch.stack(self.actions, dim=0),
            old_log_prob=torch.stack(self.log_probs, dim=0),
            old_value=values,
            returns=returns,
            advantages=advantages,
        )
        return batch

    @staticmethod
    def iter_minibatches(batch: RolloutBatch, mini_batch_size, shuffle=True):
        batch_size = batch.actions.shape[0]
        indices = torch.arange(batch_size)
        if shuffle:
            indices = indices[torch.randperm(batch_size)]

        for start in range(0, batch_size, mini_batch_size):
            idx = indices[start : start + mini_batch_size]
            yield RolloutBatch(
                observations={k: v[idx] for k, v in batch.observations.items()},
                actions=batch.actions[idx],
                old_log_prob=batch.old_log_prob[idx],
                old_value=batch.old_value[idx],
                returns=batch.returns[idx],
                advantages=batch.advantages[idx],
            )
