'''
File containing our transformer architecture and relevant pytorch submodules
'''
import math
import logging
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TransformerForDiffusion(nn.Module):
    def __init__(
        self,
        input_dim, output_dim, obs_dim, emb_dim, T_obs, T_action,
        n_encoder_layers=4, n_decoder_layers=4, n_head=4,
        p_drop_emb=0.1, p_drop_attn=0.1,
        obs_type=None, causal_attn=False, past_action_visible=False,
    ):
        super().__init__()
        assert T_obs is not None
        assert obs_type == 'ref', f'Only support ref type observation for now'
        self.causal_attn = causal_attn  
        self.past_action_visible = past_action_visible
      
        self.obs_dim = obs_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.T_obs = T_obs
        self.T_action = T_action
        self.obs_type = obs_type

        # Conditional encoder
        T_cond = 1 + self.T_obs
        self.time_emb = SinusoidalPosEmb(self.emb_dim)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, self.emb_dim))
        self.cond_obs_emb = nn.Linear(self.obs_dim, self.emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=n_head,
            dim_feedforward=4*self.emb_dim,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers
        )

        # Decoder for action denoising
        self.pos_emb = nn.Parameter(torch.zeros(1, self.T_action, self.emb_dim))
        self.input_emb = nn.Linear(self.input_dim, self.emb_dim)
        self.drop = nn.Dropout(p_drop_emb)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.emb_dim,
            nhead=self.emb_dim,
            dim_feedforward=4*self.emb_dim,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers
        )
        self.ln_f = nn.LayerNorm(self.emb_dim)
        self.head = nn.Linear(self.emb_dim, output_dim)

        # Attention mask
        if self.causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            mask = (torch.triu(torch.ones(self.T_action, self.T_action)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
    
    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout, 
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.MultiheadAttention,
            Rearrange,
            nn.SiLU,
            nn.Sequential
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name  
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def forward(self, sample, timestep, cond=None, **kwargs):
        """
        sample: (B, T_action, input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B, T_obs, cond_dim)
        """
        assert torch.is_tensor(timestep)
        assert cond.shape == (sample.shape[0], self.T_obs, self.obs_dim)
        assert sample.shape == (cond.shape[0], self.T_action, self.input_dim)
        if len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)

        # Encoder for conditioning
        timesteps = timestep.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B, 1, obs_dim)

        cond_emb = self.cond_obs_emb(cond)
        cond_emb = torch.cat([time_emb, cond_emb], dim=1)
        # (B, T_cond, obs_dim)

        tc = cond_emb.shape[1]
        cond_pos_emb = self.cond_pos_emb[:, :tc, :]
        x = self.drop(cond_emb + cond_pos_emb)
        x = self.encoder(x)
        memory = x 
        # (B, T_cond, obs_dim)

        # Decoder for action prediction
        input_emb = self.input_emb(sample)
        t = sample.shape[1]
        pos_emb = self.pos_emb[:, :t, :]
        x = self.drop(input_emb + pos_emb)
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.mask,
            memory_mask=None
        )
        # (B, T, obs_dim)
        # NOTE: We don't need a memory mask because the conditioning is always on past information
        
        x = self.ln_f(x)
        x = self.head(x)
        # (B, T, output_dim)
        return x
