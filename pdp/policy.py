'''
File implementing the higher-level policy API (e.g. querying actions and computing losses).
Abstracts away the lower-level architecture details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from pdp.modules import TransformerForDiffusion
from pdp.utils.normalizer import LinearNormalizer


class DiffusionPolicy(nn.Module):
    def __init__(
        self, 
        model: TransformerForDiffusion,
        noise_scheduler: DDPMScheduler,
        **kwargs
    ):
        super().__init__()

        self.model = model
        self.obs_dim = self.model.obs_dim
        self.action_dim = self.model.output_dim
        self.T_obs = self.model.T_obs
        self.T_action = self.model.T_action

        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        self.normalizer = None # set by set_normalizer

    @property
    def T_range(self):
        return self.T_obs + self.T_action - 1

    def get_optim_groups(self, weight_decay):
        return self.model.get_optim_groups(weight_decay)
    
    # ========= inference  ============
    def conditional_sample(self, cond_data, cond_mask, cond=None):
        model = self.model
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=cond_data.shape, 
            dtype=cond_data.dtype,
            device=cond_data.device,
        )

        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory[cond_mask] = cond_data[cond_mask]
            model_output = model(trajectory, t, cond)
    
            # compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory).prev_sample
        
        trajectory[cond_mask] = cond_data[cond_mask] 
        return trajectory

    def predict_action(self, obs_dict):
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        assert obs_dict['obs'].shape[1:] == (self.T_obs, self.obs_dim)
        nobs = self.normalizer.normalize(obs_dict)['obs']
        B, _, obs_dim = nobs.shape

        # Handle different ways of passing observation
        cond = nobs[:, :self.T_obs]
        shape = (B, self.T_action, self.action_dim)
        cond_data = torch.zeros(size=shape, device=nobs.device, dtype=nobs.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # Run sampling
        nsample = self.conditional_sample(cond_data, cond_mask, cond=cond)
        
        # Unnormalize prediction and extract action
        naction_pred = nsample[..., :self.action_dim]
        nresult = {'action': naction_pred}
        result = self.normalizer.unnormalize(nresult)
        return result
    
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer

    def get_optimizer(self, weight_decay, learning_rate, betas):
        return self.model.configure_optimizers(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas)
            )
    
    def forward(self, batch):
        return self.compute_loss(batch)
        
    def compute_loss(self, batch):
        nbatch = self.normalizer.normalize({
            'obs': batch['obs'],
            'action': batch['action']
        })  
        obs = nbatch['obs']
        action = nbatch['action']
        
        cond = obs[:, :self.T_obs]
        start = self.T_obs - 1
        end = start + self.T_action
        trajectory = action[:, start:end]
            
        # generate impainting mask
        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        B = trajectory.shape[0]
        K = self.noise_scheduler.config.num_train_timesteps

        # Sample a random timestep for each image
        timesteps = torch.randint(0, K, (B,), device=trajectory.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask
        
        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
