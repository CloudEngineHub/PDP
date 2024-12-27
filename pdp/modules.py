'''
File containing our transformer architecture and relevant pytorch submodules
'''
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


class TimeEmbedder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Mish(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def forward(self, timesteps, pos_emb):
        return self.time_embed(pos_emb(timesteps))


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        # Cross-attention: query is compared against key and value from a different source
        attn_output, attn_output_weights = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights)
        return attn_output, attn_output_weights


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = Rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


class TransformerForDiffusion(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        n_action_steps: int = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        obs_type = None, 
        task = None, 
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool=False, 
        time_as_cond: bool=True, 
        obs_as_cond: bool=False,
        n_cond_layers: int = 0,
        text_mask_prob: float = 0.1,
        past_action_visible: bool = False,
        film_conditioning: bool = False,
    ):
        super().__init__()
        assert n_obs_steps is not None
        self.casual_attn = causal_attn  
        self.past_action_visible = past_action_visible
        self.film_conditioning = film_conditioning
        self.task = task
      
        self.cond_dim = cond_dim
        self.output_dim = output_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        T = horizon

        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps
        
        self.obs_type = obs_type

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)

        self.n_emb= n_emb
        self.my_pos_emb = SinusoidalPosEmb(n_emb)
        self.my_time_emb = TimeEmbedder(n_emb)

        self.cond_obs_emb = None
        self.cond_obs_emb = nn.Sequential(
            nn.Linear(cond_dim, 1024),
            nn.Mish(),
            nn.Linear(1024, n_emb)
        )
        self.cond_ref_emb_layer  = nn.Sequential(
            nn.Linear(216, 1024),
            nn.Mish(),
            nn.Linear(1024, n_emb)
        )
        
        if self.film_conditioning:
            self.film1 = nn.Sequential(
                nn.Mish(),
                nn.Linear(n_emb*3, 2 * n_emb *n_obs_steps), #cond dim, cond channels
            )
            self.film2 = nn.Sequential(
                nn.Mish(),
                nn.Linear(n_emb, 2 * n_emb *(n_obs_steps)), #cond dim, cond channels # previously 2 * n_emb * n_obs_steps
            )
        
        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0:
            my_T_cond = T_cond      

            if self.task == 'track' and self.obs_type  == 'ref':
                my_T_cond = T_cond + n_obs_steps

            if self.task =='t2m' and self.obs_type == 't2m': 
                my_T_cond = T_cond # + 1   # IF we want to include text embedding as well
                
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, my_T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4*n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            # encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T - n_obs_steps # TAKARA :T   
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s-1) # add one dimension since time is the first token in cond
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                # import ipdb; ipdb.set_trace()

                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
        
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
    
    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            TimeEmbedder,
            DenseFiLM,
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.MultiheadAttention,
            CrossAttentionLayer,
            Rearrange,
            nn.SiLU,
            nn.Sequential)
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
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
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
        if self.cond_pos_emb is not None:
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
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        timesteps = timestep
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        ###############################################################################

        if self.obs_type == 'ref':
            assert self.task in ['t2m', 'track']
        if self.obs_type == 'phc':
            assert self.task == 'track', 'since obs type is phc, task must be track'

        # time_emb = self.time_emb(timesteps).unsqueeze(1)
        time_emb = self.my_time_emb(timesteps, self.my_pos_emb).unsqueeze(1) 

        ###############################################################################

        # process input 
        input_emb = self.input_emb(sample)
        cond_embeddings = time_emb

        if self.obs_type == 'ref':
            assert cond.shape[-1] > 360 
            cond_ref = cond[:,:,360:] 
            cond_obs = cond[:,:,:360]
        else:
            cond_obs = cond

        if self.task == 'track':
            if self.obs_type == 'ref':
                cond_obs_emb = self.cond_obs_emb(cond_obs)
                cond_ref_emb = self.cond_ref_emb_layer(cond_ref) 

                ref_mask = self.mask_batch(
                    cond_ref.shape[0], batch_mask_prob=0.1, training=self.training
                ).to(cond_ref.device)
                cond_ref_emb[ref_mask] *= 0 

                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb, cond_ref_emb], dim=1)
            elif self.obs_type == 'phc': 
                cond_obs_emb = self.cond_obs_emb(cond_obs)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)

        # elif self.task == 'both': 
        #     cond_obs_emb = self.cond_obs_emb(cond_obs)
        #     cond_ref_emb = self.cond_ref_emb_layer(cond_ref) 
        #     text_emb = self.embed_text(text).unsqueeze(1)
        #     text_mask = self.mask_batch(text_emb.shape[0], batch_mask_prob=0.0, training=self.training)

        else:
            # Old Text2Motion
            cond_obs_emb = self.cond_obs_emb(cond_obs)
            
            text_emb = self.embed_text(text).unsqueeze(1)
            text_mask = self.mask_batch(text_emb.shape[0], batch_mask_prob=0.0, training=self.training)

            scale_shifts = self.film1(torch.hstack([time_emb.squeeze(1), text_emb.squeeze(1), cond_obs_emb[:,-1].squeeze(1)])).view(-1, 2, self.n_obs_steps, self.n_emb)
            scale, shift = torch.chunk(scale_shifts, 2, dim=1)

            scale = scale.squeeze(1).clone()
            shift = shift.squeeze(1).clone()
            scale[text_mask] *=0 
            shift[text_mask] *=0
            
            scale = scale.squeeze(1)
            shift = shift.squeeze(1)
            cond_obs_emb = (1+ scale) * cond_obs_emb + shift 
            cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)

            # assert False, 'task not specified or supported'

        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[:, :tc, :]  # each position maps to a (learnable) vector
        assert position_embeddings.shape[1] == tc, 'position embedding shape mismatch for observation'
        
        x = self.drop(cond_embeddings + position_embeddings)
        x = self.encoder(x)
        memory = x 
        
        # (B,T_cond,n_emb)
        ###############################################################################

        # decoder   
        token_embeddings = input_emb
        t = token_embeddings.shape[1]

        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        assert position_embeddings.shape[1] >= t, 'position embedding shape mismatch for token'

        x = self.drop(token_embeddings + position_embeddings)
        
        if self.mask.shape[0] == 0:
            x = self.decoder(
                tgt=x,
                memory=memory,
                # tgt_mask=self.mask,
                # tgt_mask=self.mask[:12,:12], # old takara
            )
        else:
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                # tgt_mask=self.mask[:12,:12], # old takara
            )
                # (B,T,n_emb)
        
        # head      
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x

    def generate_mask_efficient(self, n, probs):
        # prob of each condition, Uncoditioned: (True, True),  text_masked: (False, True),  Ref masked: (True, False),
        outcomes = torch.tensor([[False,False], [True, True], [False, True], [True, False]])
        indices = torch.multinomial(torch.tensor(probs), n, replacement=True)
        mask = outcomes[indices]
        return mask[:,0], mask[:,1] 

    # true if its masked, false otherwise 
    def mask_batch(self, batch_size, batch_mask_prob=0.1, training=True):
        bs = batch_size
        if training and batch_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs) * batch_mask_prob).bool() #.view(bs, 1, 1)
            return mask   
        else:
            return torch.zeros(bs).bool()


class LowdimMaskGenerator(nn.Module):
    def __init__(
        self,
        action_dim, obs_dim,
        # obs mask setup
        max_n_obs_steps=2, 
        fix_obs_steps=True, 
        # action mask
        action_visible=False
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)

        # create all tensors on this device
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        # generate dim mask
        dim_mask = torch.zeros(size=shape, 
            dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[...,:self.action_dim] = True
        is_obs_dim = ~is_action_dim

        # generate obs mask
        if self.fix_obs_steps:
            obs_steps = torch.full((B,), 
            fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1, high=self.max_n_obs_steps+1, 
                size=(B,), generator=rng, device=device)
            
        steps = torch.arange(0, T, device=device).reshape(1,T).expand(B,T)
        obs_mask = (steps.T < obs_steps).T.reshape(B,T,1).expand(B,T,D)
        obs_mask = obs_mask & is_obs_dim

        # generate action mask
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1, 
                torch.tensor(0,
                    dtype=obs_steps.dtype, 
                    device=obs_steps.device))
            action_mask = (steps.T < action_steps).T.reshape(B,T,1).expand(B,T,D)
            action_mask = action_mask & is_action_dim

        mask = obs_mask
        if self.action_visible:
            mask = mask | action_mask
        
        return mask