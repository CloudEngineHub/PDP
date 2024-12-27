'''
File implementing the higher-level policy API (e.g. querying actions and computing losses).
Abstracts away the lower-level architecture details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from pdp.modules import TransformerForDiffusion, LowdimMaskGenerator
from pdp.utils.normalizer import LinearNormalizer


class DiffusionPolicy(nn.Module):
    def __init__(
        self, 
        model: TransformerForDiffusion,
        noise_scheduler: DDPMScheduler,
        num_inference_steps,
        pred_action_steps_only=False,
        **kwargs
    ):
        super().__init__()
        assert num_inference_steps is not None

        self.model = model
        self.obs_dim = self.model.cond_dim
        self.action_dim = self.model.output_dim
        self.horizon = self.model.horizon
        self.n_obs_steps = self.model.n_obs_steps
        self.n_action_steps = self.model.n_action_steps

        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim, obs_dim=0,
            max_n_obs_steps=self.n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = None # set by set_normalizer
        self.num_inference_steps = num_inference_steps
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

    def get_optim_groups(self, weight_decay):
        return self.model.get_optim_groups(weight_decay)
    
    # ========= inference  ============
    def conditional_sample(self,                 # included text, TAKARA
            condition_data, text, clean_traj, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, text, clean_traj, t, cond) # included text, TAKARA
    
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, obs_dict):
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, obs_dim = nobs.shape

        # Handle different ways of passing observation
        cond = nobs[:, :self.n_obs_steps]
        shape = (B, self.horizon, self.action_dim)
        if self.pred_action_steps_only:
            shape = (B, self.n_action_steps, self.action_dim)
        cond_data = torch.zeros(size=shape, device=nobs.device, dtype=nobs.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # Run sampling
        nsample = self.conditional_sample(cond_data, cond_mask, cond=cond, **self.kwargs)
        
        # Unnormalize prediction and extract action
        naction_pred = nsample[..., :self.action_dim]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = self.n_obs_steps - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
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
        
        # Handle different ways of passing observation
        cond = None
        trajectory = action
        cond = obs[:, :self.n_obs_steps]
        if self.pred_action_steps_only:
            start = self.n_action_steps - 1
            end = start + self.n_action_steps
            trajectory = action[:, start:end]
            
        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        B = trajectory.shape[0]
        K = self.noise_scheduler.config.num_train_timesteps,

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
        pred = self.model(noisy_trajectory, text_embeddings, clean_trajs, timesteps, cond)

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


    def exponential_decay_with_control_and_cap(self, x, control_x, control_y):
        # Calculate the decay rate b based on the control point, where at x=0, y=1, and at x=control_x, y=control_y and everything after control_x is capped at control_y 
        b = -torch.log(control_y) / control_x
        decay_values = torch.exp(-b * x)

        # Cap the values where x is greater than control_x
        capped_values = torch.where(x > control_x, torch.maximum(decay_values, control_y), decay_values)
        return capped_values


class EMAModel:
    pass