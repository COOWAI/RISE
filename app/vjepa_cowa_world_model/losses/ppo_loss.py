import torch


def gaussian_log_prob(actions, mean, log_std):
    """Compute diagonal Gaussian log probability per sample."""
    var = torch.exp(2.0 * log_std)
    log_two_pi = torch.log(actions.new_tensor(2.0 * torch.pi))
    return -0.5 * (((actions - mean) ** 2) / var + 2.0 * log_std + log_two_pi).sum(dim=-1)


def gaussian_entropy(log_std):
    """Compute entropy of a diagonal Gaussian."""
    return (log_std + 0.5 * (1.0 + torch.log(log_std.new_tensor(2.0 * torch.pi)))).sum(dim=-1)


def ppo_loss(
    *,
    policy_mean,
    policy_log_std,
    value,
    actions,
    old_log_prob,
    returns,
    advantages,
    old_value=None,
    clip_eps=0.2,
    value_clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
):
    """Standard clipped PPO objective for diagonal Gaussian policy."""
    log_prob = gaussian_log_prob(actions, policy_mean, policy_log_std)
    ratio = torch.exp(log_prob - old_log_prob)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    if old_value is None:
        value_loss = 0.5 * ((value - returns) ** 2).mean()
    else:
        value_pred_clipped = old_value + torch.clamp(value - old_value, -value_clip_eps, value_clip_eps)
        value_loss_unclipped = (value - returns) ** 2
        value_loss_clipped = (value_pred_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

    entropy = gaussian_entropy(policy_log_std).mean()

    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()

    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "clip_fraction": clip_fraction,
        "approx_kl": 0.5 * ((old_log_prob - log_prob) ** 2).mean(),
        "log_prob": log_prob,
        "ratio": ratio,
    }
