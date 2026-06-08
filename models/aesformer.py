import warnings

from typing import List, Optional, Tuple
from transformers import BertModel, BertTokenizer  # FeatureExtractor, AutoTokenizer
from transformers.modeling_utils import ModuleUtilsMixin
import copy
from torch.nn import functional as F
from einops import rearrange

import torch
import torch.nn as nn   
from timm.models.layers import trunc_normal_
from .swin import SwinTransformer


class img_feature(nn.Module):
    def __init__(self, model_type='tiny'):
        super(img_feature, self).__init__()
        if model_type == 'tiny':
            self.img_model = SwinTransformer(num_classes=10)
        elif model_type == 'small':
            self.img_model = SwinTransformer(depths=[2, 2, 18, 2], num_classes=10, drop_path_rate=0.1)
        elif model_type == 'base':
            self.img_model = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                                             num_classes=10, drop_path_rate=0.1)

        # swin ckpt
        d = torch.load('checkpoints/swin/swin_base_patch4_window7_224.pth', map_location='cpu')
        print(self.img_model.load_state_dict(d['model'], strict=False))

    def forward(self, image):
        image_features = self.img_model.get_feature(image)
        return image_features

    def get_multi_featues(self, image):
        image_features = self.img_model.get_multi_features(image)
        return image_features


class bert_feature(nn.Module):
    def __init__(self, device):
        # bert ckpt
        checkpoint = 'ckp/zzd/pretrain_model/bert/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
        # hugging face
        # checkpoint = "bert-base-cased"
        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint)
        self.bert_model = BertModel.from_pretrained(checkpoint)


    def forward(self, text):
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        # text_features = self.get_text_features(**tokens)
        text_features = self.bert_model(**tokens)
        return text_features.last_hidden_state, tokens.data['attention_mask']

    def get_text_features(self, input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = None,):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device, dtype=attention_mask.dtype)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        feature = embedding_output
        for i, bert_layer in enumerate(self.encoder):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            feature = bert_layer(feature,
                                 attention_mask=extended_attention_mask,
                                 head_mask=layer_head_mask,
                                 encoder_hidden_states=encoder_hidden_states,
                                 encoder_attention_mask=encoder_extended_attention_mask,
                                 past_key_value=past_key_value,
                                 output_attentions=output_attentions,
                                 )[0]

        return feature

    def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        if not (attention_mask.dim() == 2 and self.config.is_decoder):
            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.iinfo(dtype).min
        # extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask


class Swin_Bert_vlmo_clip_mean_score_multi_features(nn.Module):
    def __init__(self, device, num_classes=10, dim=768, depth=6, heads=12, dim_head=64, dropout=0.,
                 norm_layer=nn.LayerNorm, model_type='tiny', type='img', queue_size=1024, temp=0.07, momentum=0.995):
        super(Swin_Bert_vlmo_clip_mean_score_multi_features, self).__init__()
        self.model_type = model_type
        self.type = type
        self.device = device
       
        self.heads_vl_v = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

        self.heads_vl_l = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

        self.head_v = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

        if model_type == 'base':
            self.proj = nn.Linear(1024, dim)
            self.proj_m = nn.Linear(1024, dim)

        self.feature_neck = multi_features(model_type)

        self.apply(self._init_weights)

        self.adapter_1 = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim)
        )
        self.init_weights(self.adapter_1)

        self.adapter_2 = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim)
        )
        self.init_weights(self.adapter_2)

        self.adapter_3 = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim)
        )
        self.init_weights(self.adapter_3)

        self.adapter_4 = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim)
        )
        self.init_weights(self.adapter_4)

        self.num_quality_prompts = 5
        self.prompt_temperature = 0.07
        self.quality_prompt_embeddings = nn.Parameter(torch.randn(self.num_quality_prompts, dim))
        trunc_normal_(self.quality_prompt_embeddings, std=0.02)
        self.prompt_fusion = nn.Sequential(
            nn.Linear(2 * dim + self.num_quality_prompts, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.init_weights(self.prompt_fusion[0])

        self.feature_vit = img_feature(model_type=model_type)
        if self.model_type == 'base':
            self.norm_v = nn.LayerNorm(dim)
            self.norm = nn.LayerNorm(dim)
            self._init_weights(self.norm)
            self._init_weights(self.norm_v)
        else:
            self.norm_v = copy.deepcopy(self.feature_vit.img_model.norm)
            self.norm = copy.deepcopy(self.feature_vit.img_model.norm)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if type == 'img':
            for p in self.feature_vit.parameters():
                p.requires_grad = False

            if model_type == 'base':
                for p in self.proj.parameters():
                    p.requires_grad = False

        self.feature = bert_feature(device=device)
        for p in self.feature.bert_model.parameters():
            p.requires_grad = False

        self.prompt_ctx_len = 32
        self.quality_categories = ["terrible", "bad", "average", "good", "perfect"]
        self.prompt_loss_temperature = 0.07
        self.lambda_prompt = 0.01
        self.prompt_gate = nn.Parameter(torch.tensor(0.1))

        self.prompt_context = nn.Parameter(torch.randn(self.num_quality_prompts, self.prompt_ctx_len, dim))
        trunc_normal_(self.prompt_context, std=0.02)

        self.prompt_cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.prompt_cross_norm1 = nn.LayerNorm(dim)
        self.prompt_cross_norm2 = nn.LayerNorm(dim)
        self.prompt_cross_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.prompt_cross_gate = nn.Parameter(torch.tensor(0.5))

        self.prompt_token_cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.prompt_token_cross_norm1 = nn.LayerNorm(dim)
        self.prompt_token_cross_norm2 = nn.LayerNorm(dim)
        self.prompt_token_cross_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.prompt_token_cross_gate = nn.Parameter(torch.tensor(0.5))

        self._init_weights(self.prompt_cross_norm1)
        self._init_weights(self.prompt_cross_norm2)
        self.init_weights(self.prompt_cross_ffn[0])
        self.init_weights(self.prompt_cross_ffn[3])
        self._init_weights(self.prompt_token_cross_norm1)
        self._init_weights(self.prompt_token_cross_norm2)
        self.init_weights(self.prompt_token_cross_ffn[0])
        self.init_weights(self.prompt_token_cross_ffn[3])

        print("prompt_cross_gate.requires_grad:", self.prompt_cross_gate.requires_grad)
        print("prompt_cross_gate:", self.prompt_cross_gate.item())
        print("prompt_token_cross_gate.requires_grad:", self.prompt_token_cross_gate.requires_grad)
        print("prompt_token_cross_gate:", self.prompt_token_cross_gate.item())

        anchor_tokens = self.feature.tokenizer(
            self.quality_categories,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt"
        )
        self.register_buffer("quality_anchor_input_ids", anchor_tokens["input_ids"])
        self.register_buffer("quality_anchor_attention_mask", anchor_tokens["attention_mask"])

        self.cross_encoder = copy.deepcopy(self.feature.bert_model.base_model.encoder.layer[12-depth:])
        self.vlmo_neck = pretrain_neck(self.cross_encoder, depth, type=self.type)

        self.norm_cl = norm_layer(dim)
        self.norm_cl_m = norm_layer(dim)
        self.visual_encoder_m = img_feature(model_type=model_type)
        self.text_encoder_m = bert_feature(device=device)
        self.momentum = momentum
        self.temp = nn.Parameter(torch.ones([]) * temp)
        # create the queue
        self.queue_size = queue_size
        self.register_buffer("image_queue", torch.randn(dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = F.normalize(self.image_queue, dim=0)
        self.text_queue = F.normalize(self.text_queue, dim=0)

        self.model_pairs = [[self.feature_vit, self.visual_encoder_m],
                            # [self.proj, self.proj_m],
                            # [self.visual_proj, self.visual_proj_m],
                            [self.feature, self.text_encoder_m],
                            # [self.text_proj, self.text_proj_m],
                            [self.norm_cl, self.norm_cl_m]]
        self.copy_params()
        # for p in self.feature.parameters():
        #     p.requires_grad = False

    def train_first_stage(self, image, text):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        x1 = self.feature_vit(image)
        if self.model_type == 'base':
            x1 = self.proj(x1)
        x2, attention_mask = self.feature(text)

        if self.training:
            image_embeds = self.norm_cl(x1)
            image_embeds = self.avgpool(image_embeds.transpose(1, 2))
            image_embeds = torch.flatten(image_embeds, 1)
            text_embeds = x2[:, 0, :]

            image_feat = F.normalize(image_embeds, dim=-1)
            text_feat = F.normalize(text_embeds, dim=-1)

            with torch.no_grad():
                self._momentum_update()
                image_embeds_m = self.visual_encoder_m(image)
                if self.model_type == 'base':
                    image_embeds_m = self.proj_m(image_embeds_m)
                image_embeds_m = self.norm_cl_m(image_embeds_m)  # B L C
                image_embeds_m = self.avgpool(image_embeds_m.transpose(1, 2))  # B C 1
                image_embeds_m = torch.flatten(image_embeds_m, 1)# B C 
                image_feat_m = F.normalize(image_embeds_m, dim=-1)
                image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
                text_embeds_m, _ = self.text_encoder_m(text)
                text_feat_m = F.normalize(text_embeds_m[:, 0, :], dim=-1)
                text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

                sim_i2t_m = image_feat_m @ text_feat_all / self.temp
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp
                # sim_i2t_m = self.logit_scale.exp() * image_feat_m @ text_feat_all
                # sim_t2i_m = self.logit_scale.exp() * text_feat_m @ image_feat_all

                sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                sim_targets.fill_diagonal_(1)
    
                sim_i2t_targets = sim_targets
                sim_t2i_targets = sim_targets

            sim_i2t = image_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ image_feat_all / self.temp

            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            loss_ita = (loss_i2t + loss_t2i) / 2

            self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        img_mask = torch.ones((x1.shape[0], x1.shape[1])).to(self.device)
        attention_mask = torch.cat([img_mask, attention_mask], dim=-1)
        input_shape = attention_mask.size()
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, self.device,
                                                                   attention_mask.dtype)

        x1, x2 = self.vlmo_neck(x1, x2, extended_attention_mask)
        x1 = self.norm(x1)  # B L C
        x1 = self.avgpool(x1.transpose(1, 2))  # B C 1
        x1 = torch.flatten(x1, 1)
        x2 = x2[:, 0, :]
        img_out = self.heads_vl_v(x1)
        text_out = self.heads_vl_l(x2)
        if self.training:
            return img_out, text_out, loss_ita
        else:
            return img_out, text_out

    def train_second_stage(self, image):
        image_feature = self.feature_vit(image)
        if self.model_type == 'base':
            image_feature = self.proj(image_feature)
        image_feature = self.adapter_1(image_feature) + image_feature

        image_feature, _ = self.vlmo_neck(image_feature, image_feature)

        image_feature = self.norm_v(image_feature)  # B L C
        image_feature = self.avgpool(image_feature.transpose(1, 2))  # B C 1
        image_feature = torch.flatten(image_feature, 1)

        x = self.head_v(image_feature)
        return x

    def train_second_stage_with_multi_features(self, image):
        image_feature = self.feature_vit.get_multi_featues(image)
        f1, f2, f3, f4 = self.feature_neck(image_feature)
        if self.model_type == 'base':
            f1, f2, f3, f4 = self.proj(f1), self.proj(f2), self.proj(f3), self.proj(f4),
        f1 = self.adapter_1(f1) + f1
        f2 = self.adapter_2(f2) + f2
        f3 = self.adapter_3(f3) + f3
        f4 = self.adapter_4(f4) + f4
        image_feature = torch.cat([f1, f2, f3, f4], dim=1)

        image_feature, _ = self.vlmo_neck(image_feature, image_feature)

        image_feature = self.norm_v(image_feature)  # B L C
        image_feature = self.avgpool(image_feature.transpose(1, 2))  # B C 1
        image_feature = torch.flatten(image_feature, 1)

        x = self.head_v(image_feature)
        return x

    # def fuse_with_learnable_quality_prompts(self, image_feature):
    #     image_norm = F.normalize(image_feature, dim=-1)
    #     prompt_norm = F.normalize(self.quality_prompt_embeddings, dim=-1)
    #     similarity_scores = image_norm @ prompt_norm.t()  # [B, 5]
    #     weights = torch.softmax(similarity_scores / self.prompt_temperature, dim=-1)
    #     weighted_prompt_feature = weights @ self.quality_prompt_embeddings  # [B, dim]
    #     fused_feature = torch.cat([image_feature, weighted_prompt_feature, similarity_scores], dim=-1)  # [B, 2*dim+5]
    #     fused_feature = self.prompt_fusion(fused_feature)  # [B, dim]
    #     return image_feature + fused_feature

    # def train_second_stage_with_learnable_prompts(self, image):
    #     image_feature = self.feature_vit.get_multi_featues(image)
    #     f1, f2, f3, f4 = self.feature_neck(image_feature)
    #     if self.model_type == 'base':
    #         f1, f2, f3, f4 = self.proj(f1), self.proj(f2), self.proj(f3), self.proj(f4),
    #     f1 = self.adapter_1(f1) + f1
    #     f2 = self.adapter_2(f2) + f2
    #     f3 = self.adapter_3(f3) + f3
    #     f4 = self.adapter_4(f4) + f4
    #     image_feature = torch.cat([f1, f2, f3, f4], dim=1)

    #     image_feature, _ = self.vlmo_neck(image_feature, image_feature)

    #     image_feature = self.norm_v(image_feature)  # B L C
    #     image_feature = self.avgpool(image_feature.transpose(1, 2))  # B C 1
    #     image_feature = torch.flatten(image_feature, 1)  # [B, dim]
    #     image_feature = self.fuse_with_learnable_quality_prompts(image_feature)

    #     x = self.head_v(image_feature)
    #     return x

    def get_anchored_quality_prompt_features(self):
        device = self.prompt_context.device
        input_ids = self.quality_anchor_input_ids.to(device)  # [5, L_anchor]
        anchor_mask = self.quality_anchor_attention_mask.to(device)  # [5, L_anchor]

        word_embeddings = self.feature.bert_model.embeddings.word_embeddings
        anchor_embeds = word_embeddings(input_ids)  # [5, L_anchor, dim]

        cls_id = self.feature.tokenizer.cls_token_id
        sep_id = self.feature.tokenizer.sep_token_id
        cls_ids = torch.full((self.num_quality_prompts, 1), cls_id, dtype=torch.long, device=device)
        sep_ids = torch.full((self.num_quality_prompts, 1), sep_id, dtype=torch.long, device=device)
        cls_embeds = word_embeddings(cls_ids)  # [5, 1, dim]
        sep_embeds = word_embeddings(sep_ids)  # [5, 1, dim]

        inputs_embeds = torch.cat([cls_embeds, self.prompt_context, anchor_embeds, sep_embeds], dim=1)  # [5, 1+M+L_anchor+1, dim]

        cls_mask = torch.ones((self.num_quality_prompts, 1), dtype=anchor_mask.dtype, device=device)
        ctx_mask = torch.ones((self.num_quality_prompts, self.prompt_ctx_len), dtype=anchor_mask.dtype, device=device)
        sep_mask = torch.ones((self.num_quality_prompts, 1), dtype=anchor_mask.dtype, device=device)
        attention_mask = torch.cat([cls_mask, ctx_mask, anchor_mask, sep_mask], dim=1)  # [5, 1+M+L_anchor+1]

        outputs = self.feature.bert_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        prompt_features = outputs.last_hidden_state[:, 0, :]  # [5, dim]
        return prompt_features

    def fuse_with_anchored_learnable_quality_prompts(self, image_feature, return_similarity=False):
        prompt_features = self.get_anchored_quality_prompt_features()  # [5, dim]
        image_norm = F.normalize(image_feature, dim=-1)
        prompt_norm = F.normalize(prompt_features, dim=-1)
        similarity_scores = image_norm @ prompt_norm.t()  # [B, 5]
        weights = torch.softmax(similarity_scores / self.prompt_temperature, dim=-1)
        weighted_prompt_feature = weights @ prompt_features  # [B, dim]
        concat_feature = torch.cat([image_feature, weighted_prompt_feature, similarity_scores], dim=-1)  # [B, 2*dim+5]
        prompt_delta = self.prompt_fusion(concat_feature)  # [B, dim]
        fused_feature = image_feature + self.prompt_gate * prompt_delta# [B, dim]
        #fused_feature = image_feature + prompt_delta  
        if return_similarity:
            return fused_feature, similarity_scores
        return fused_feature

    def train_second_stage_with_anchored_prompts(self, image, return_similarity=False):
        image_feature = self.feature_vit.get_multi_featues(image)
        f1, f2, f3, f4 = self.feature_neck(image_feature)
        if self.model_type == 'base':
            f1, f2, f3, f4 = self.proj(f1), self.proj(f2), self.proj(f3), self.proj(f4),
        f1 = self.adapter_1(f1) + f1
        f2 = self.adapter_2(f2) + f2
        f3 = self.adapter_3(f3) + f3
        f4 = self.adapter_4(f4) + f4
        image_feature = torch.cat([f1, f2, f3, f4], dim=1)

        image_feature, _ = self.vlmo_neck(image_feature, image_feature)

        image_feature = self.norm_v(image_feature)  # B L C
        image_feature = self.avgpool(image_feature.transpose(1, 2))  # B C 1
        image_feature = torch.flatten(image_feature, 1)  # [B, dim]
        if return_similarity:
            image_feature, similarity_scores = self.fuse_with_anchored_learnable_quality_prompts(
                image_feature,
                return_similarity=True
            )
        else:
            image_feature = self.fuse_with_anchored_learnable_quality_prompts(
                image_feature,
                return_similarity=False
            )

        x = self.head_v(image_feature)
        if return_similarity:
            return x, similarity_scores
        return x

    def fuse_with_anchored_prompts_cross_attention(self, image_feature, return_similarity=False):
        prompt_features = self.get_anchored_quality_prompt_features()  # [5, dim]
        image_norm = F.normalize(image_feature, dim=-1)
        prompt_norm = F.normalize(prompt_features, dim=-1)
        similarity_scores = image_norm @ prompt_norm.t()  # [B, 5]

        B = image_feature.size(0)
        query = image_feature.unsqueeze(1)  # [B, 1, dim]
        key_value = prompt_features.unsqueeze(0).expand(B, -1, -1)  # [B, 5, dim]

        attn_out, attn_weights = self.prompt_cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=True
        )
        # attn_out: [B, 1, dim], attn_weights: [B, 1, 5]

        query = self.prompt_cross_norm1(query + attn_out)  # [B, 1, dim]
        ffn_out = self.prompt_cross_ffn(query)  # [B, 1, dim]
        prompt_delta = self.prompt_cross_norm2(query + ffn_out)  # [B, 1, dim]
        prompt_delta = prompt_delta.squeeze(1)  # [B, dim]

        fused_feature = image_feature + self.prompt_cross_gate * prompt_delta  # [B, dim]
        if return_similarity:
            return fused_feature, similarity_scores, attn_weights
        return fused_feature

    def fuse_tokens_with_anchored_prompts_cross_attention(self, image_tokens, return_similarity=False, return_attn=False):
        prompt_features = self.get_anchored_quality_prompt_features()  # [5, dim]
        image_global = image_tokens.mean(dim=1)  # [B, dim]

        image_norm = F.normalize(image_global, dim=-1)
        prompt_norm = F.normalize(prompt_features, dim=-1)
        similarity_scores = image_norm @ prompt_norm.t()  # [B, 5]

        B = image_tokens.size(0)
        query = image_tokens  # [B, L, dim]
        key_value = prompt_features.unsqueeze(0).expand(B, -1, -1)  # [B, 5, dim]

        attn_out, attn_weights = self.prompt_token_cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=True
        )
        # attn_out: [B, L, dim], attn_weights: [B, L, 5]

        x = self.prompt_token_cross_norm1( attn_out)  # [B, L, dim]
        ffn_out = self.prompt_token_cross_ffn(x)  # [B, L, dim]
        prompt_delta = self.prompt_token_cross_norm2(x + ffn_out)  # [B, L, dim]
        fused_tokens = image_tokens + self.prompt_token_cross_gate * prompt_delta  # [B, L, dim]
        # fused_tokens = image_tokens + prompt_delta

        if return_similarity and return_attn:
            return fused_tokens, similarity_scores, attn_weights
        elif return_similarity:
            return fused_tokens, similarity_scores
        return fused_tokens

    def train_second_stage_with_anchored_prompts_cross_attention(self, image, return_similarity=False, return_attn=False):
        image_feature = self.feature_vit.get_multi_featues(image)
        f1, f2, f3, f4 = self.feature_neck(image_feature)
        if self.model_type == 'base':
            f1, f2, f3, f4 = self.proj(f1), self.proj(f2), self.proj(f3), self.proj(f4),
        f1 = self.adapter_1(f1) + f1
        f2 = self.adapter_2(f2) + f2
        f3 = self.adapter_3(f3) + f3
        f4 = self.adapter_4(f4) + f4
        image_feature = torch.cat([f1, f2, f3, f4], dim=1)

        image_feature, _ = self.vlmo_neck(image_feature, image_feature)

        image_feature = self.norm_v(image_feature)  # B L C
        image_feature = self.avgpool(image_feature.transpose(1, 2))  # B C 1
        image_feature = torch.flatten(image_feature, 1)  # [B, dim]

        if return_similarity or return_attn:
            image_feature, similarity_scores, attn_weights = self.fuse_with_anchored_prompts_cross_attention(
                image_feature,
                return_similarity=True
            )
        else:
            image_feature = self.fuse_with_anchored_prompts_cross_attention(
                image_feature,
                return_similarity=False
            )

        x = self.head_v(image_feature)  # [B, 10]
        if return_similarity and return_attn:
            return x, similarity_scores, attn_weights
        elif return_similarity:
            return x, similarity_scores
        return x

    def train_second_stage_with_anchored_prompts_token_cross_attention(self, image, return_similarity=False, return_attn=False):
        image_feature = self.feature_vit.get_multi_featues(image)
        f1, f2, f3, f4 = self.feature_neck(image_feature)
        if self.model_type == 'base':
            f1, f2, f3, f4 = self.proj(f1), self.proj(f2), self.proj(f3), self.proj(f4),
        f1 = self.adapter_1(f1) + f1
        f2 = self.adapter_2(f2) + f2
        f3 = self.adapter_3(f3) + f3
        f4 = self.adapter_4(f4) + f4
        image_feature = torch.cat([f1, f2, f3, f4], dim=1)  # [B, L, dim]

        image_feature, _ = self.vlmo_neck(image_feature, image_feature)

        image_feature = self.norm_v(image_feature)  # [B, L, dim]

        if return_similarity and return_attn:
            image_feature, similarity_scores, attn_weights = self.fuse_tokens_with_anchored_prompts_cross_attention(
                image_feature,
                return_similarity=True,
                return_attn=True
            )
        elif return_similarity:
            image_feature, similarity_scores = self.fuse_tokens_with_anchored_prompts_cross_attention(
                image_feature,
                return_similarity=True,
                return_attn=False
            )
        else:
            image_feature = self.fuse_tokens_with_anchored_prompts_cross_attention(
                image_feature,
                return_similarity=False,
                return_attn=False
            )

        image_feature = self.avgpool(image_feature.transpose(1, 2))  # [B, dim, 1]
        image_feature = torch.flatten(image_feature, 1)  # [B, dim]

        x = self.head_v(image_feature)  # [B, 10]
        if return_similarity and return_attn:
            return x, similarity_scores, attn_weights
        elif return_similarity:
            return x, similarity_scores
        return x

    def forward(self, image):
        image_feature = self.feature_vit.get_multi_featues(image)
        f1, f2, f3, f4 = self.feature_neck(image_feature)
        if self.model_type == 'base':
            f1, f2, f3, f4 = self.proj(f1), self.proj(f2), self.proj(f3), self.proj(f4),
        f1 = self.adapter_1(f1) + f1
        f2 = self.adapter_2(f2) + f2
        f3 = self.adapter_3(f3) + f3
        f4 = self.adapter_4(f4) + f4
        image_feature = torch.cat([f1, f2, f3, f4], dim=1)

        image_feature, _ = self.vlmo_neck(image_feature, image_feature)

        image_feature = self.norm_v(image_feature)  # B L C
        image_feature = self.avgpool(image_feature.transpose(1, 2))  # B C 1
        image_feature = torch.flatten(image_feature, 1)

        x = self.head_v(image_feature)
        return x

    def forward_with_learnable_prompts(self, image):
        return self.train_second_stage_with_learnable_prompts(image)

    def forward_with_anchored_prompts(self, image, return_similarity=False):
        return self.train_second_stage_with_anchored_prompts(image, return_similarity=return_similarity)

    # Cross-attention version training usage:
    # y_pred, similarity_scores = model.forward_with_anchored_prompts_cross_attention(
    #     img,
    #     return_similarity=True
    # )
    # loss_prompt, quality_label, quality_onehot = model.compute_prompt_loss(
    #     similarity_scores,
    #     y,
    #     target_is_distribution=True
    # )
    # optimizer = torch.optim.AdamW(
    #     model.get_anchored_prompt_cross_attention_param_groups(),
    #     betas=(0.9, 0.99),
    #     weight_decay=1e-4
    # )
    def forward_with_anchored_prompts_cross_attention(self, image, return_similarity=False, return_attn=False):
        return self.train_second_stage_with_anchored_prompts_cross_attention(
            image,
            return_similarity=return_similarity,
            return_attn=return_attn
        )

    def forward_with_anchored_prompts_token_cross_attention(self, image, return_similarity=False, return_attn=False):
        return self.train_second_stage_with_anchored_prompts_token_cross_attention(
            image,
            return_similarity=return_similarity,
            return_attn=return_attn
        )

    # Training usage example:
    # pred, similarity_scores = model.forward_with_anchored_prompts(image, return_similarity=True)
    # loss_main = emd_loss(pred, target_distribution)
    # loss_prompt, quality_label, quality_onehot = model.compute_prompt_loss(
    #     similarity_scores,
    #     target_distribution,
    #     target_is_distribution=True
    # )
    # loss = loss_main + model.lambda_prompt * loss_prompt
    # If target is MOS score (shape [B]): set target_is_distribution=False.
    def ava_score_to_quality_label(self, score):
        quality_label = torch.zeros_like(score, dtype=torch.long)
        quality_label[(score >= 3) & (score < 5)] = 1
        quality_label[(score >= 5) & (score < 6)] = 2
        quality_label[(score >= 6) & (score < 8)] = 3
        quality_label[score >= 8] = 4
        return quality_label

    def ava_distribution_to_score(self, target_distribution):
        target_sum = target_distribution.sum(dim=1, keepdim=True)
        target_norm = target_distribution / target_sum.clamp_min(1e-6)
        score_levels = torch.arange(1, target_distribution.size(1) + 1, device=target_distribution.device,
                                    dtype=target_distribution.dtype)
        score = torch.sum(target_norm * score_levels, dim=1)
        return score

    def ava_target_to_quality_onehot(self, target, target_is_distribution=True):
        if target_is_distribution:
            score = self.ava_distribution_to_score(target)
        else:
            score = target
        quality_label = self.ava_score_to_quality_label(score)
        quality_onehot = F.one_hot(quality_label, num_classes=self.num_quality_prompts).float()
        return quality_label, quality_onehot

    def compute_prompt_loss(self, similarity_scores, target, target_is_distribution=True):
        quality_label, quality_onehot = self.ava_target_to_quality_onehot(
            target,
            target_is_distribution=target_is_distribution
        )
        logits = similarity_scores / self.prompt_loss_temperature
        log_prob = F.log_softmax(logits, dim=-1)
        loss_prompt = -(quality_onehot * log_prob).sum(dim=-1).mean()
        return loss_prompt, quality_label, quality_onehot

    # def get_prompt_concat_param_groups(self):
    #     adapter_params = list(self.adapter_1.parameters()) + list(self.adapter_2.parameters()) + \
    #                      list(self.adapter_3.parameters()) + list(self.adapter_4.parameters())
    #     head_v_params = list(self.head_v.parameters())
    #     prompt_fusion_params = list(self.prompt_fusion.parameters())
    #     quality_prompt_params = [self.quality_prompt_embeddings]
    #     return [
    #         {"params": adapter_params, "lr": 1e-4},
    #         {"params": head_v_params, "lr": 1e-4},
    #         {"params": prompt_fusion_params, "lr": 1e-4},
    #         {"params": quality_prompt_params, "lr": 3e-4},
    #     ]

    def get_anchored_prompt_param_groups(self):
        adapter_params = list(self.adapter_1.parameters()) + list(self.adapter_2.parameters()) + \
                         list(self.adapter_3.parameters()) + list(self.adapter_4.parameters())
        head_v_params = list(self.head_v.parameters())
        prompt_fusion_params = list(self.prompt_fusion.parameters())
        prompt_context_params = [self.prompt_context]
        prompt_gate_params = [self.prompt_gate]
        return [
            {"params": adapter_params, "lr": 1e-4},
            {"params": head_v_params, "lr": 1e-4},
            {"params": prompt_fusion_params, "lr": 1e-4},
            {"params": prompt_context_params, "lr": 2e-4},
            {"params": prompt_gate_params, "lr": 1e-3},
        ]

    def get_anchored_prompt_cross_attention_param_groups(self):
        adapter_params = list(self.adapter_1.parameters()) + list(self.adapter_2.parameters()) + \
                         list(self.adapter_3.parameters()) + list(self.adapter_4.parameters())
        head_v_params = list(self.head_v.parameters())
        prompt_context_params = [self.prompt_context]
        cross_attn_params = (
            list(self.prompt_cross_attn.parameters()) +
            list(self.prompt_cross_norm1.parameters()) +
            list(self.prompt_cross_norm2.parameters()) +
            list(self.prompt_cross_ffn.parameters())
        )
        return [
            {"params": adapter_params, "lr": 1e-4},
            {"params": head_v_params, "lr": 1e-4},
            {"params": prompt_context_params, "lr": 2e-4},
            {"params": cross_attn_params, "lr": 1e-4},
            {"params": [self.prompt_cross_gate], "lr": 1e-3, "weight_decay": 0.0},
        ]

    def get_anchored_prompt_token_cross_attention_param_groups(self):
        adapter_params = list(self.adapter_1.parameters()) + list(self.adapter_2.parameters()) + \
                         list(self.adapter_3.parameters()) + list(self.adapter_4.parameters())
        head_v_params = list(self.head_v.parameters())
        prompt_context_params = [self.prompt_context]
        token_cross_attn_params = (
            list(self.prompt_token_cross_attn.parameters()) +
            list(self.prompt_token_cross_norm1.parameters()) +
            list(self.prompt_token_cross_norm2.parameters()) +
            list(self.prompt_token_cross_ffn.parameters())
        )
        return [
            {"params": adapter_params, "lr": 1e-4},
            {"params": head_v_params, "lr": 1e-4},
            {"params": prompt_context_params, "lr": 2e-4},
            {"params": token_cross_attn_params, "lr": 1e-4},
            {"params": [self.prompt_token_cross_gate], "lr": 1e-3, "weight_decay": 0.0},
        ]

    def test_img(self, image):
        with torch.no_grad():
            image_feature = self.feature_vit.get_multi_featues(image)
            # if self.model_type == 'base':
            #     image_feature = self.proj(image_feature)
        f1, f2, f3, f4 = self.feature_neck(image_feature)
        if self.model_type == 'base':
            f1, f2, f3, f4 = self.proj(f1), self.proj(f2), self.proj(f3), self.proj(f4)
        f1 = self.adapter_1(f1) + f1
        f2 = self.adapter_2(f2) + f2
        f3 = self.adapter_3(f3) + f3
        f4 = self.adapter_4(f4) + f4

        image_feature = torch.cat([f1, f2, f3, f4], dim=1)
        image_feature = self.norm_v(image_feature)  # B L C
        image_feature = self.avgpool(image_feature.transpose(1, 2))  # B C 1
        image_feature = torch.flatten(image_feature, 1)
        x = self.head_v(image_feature)
        return x

    @torch.no_grad()
    def get_sim(self, image, text):
        x1 = self.feature_vit(image)
        x2, attention_mask = self.feature(text)

        image_embeds = self.norm_cl(x1)
        image_embeds = self.avgpool(image_embeds.transpose(1, 2))
        image_embeds = torch.flatten(image_embeds, 1)
        text_embeds = x2[:, 0, :]

        image_feat = F.normalize(image_embeds, dim=-1)
        text_feat = F.normalize(text_embeds, dim=-1)

        sim = image_feat @ text_feat.T
        sim1 = F.log_softmax(sim, dim=-1)
        sim2 = F.softmax(sim, dim=-1)

        img_mask = torch.ones((x1.shape[0], x1.shape[1])).to(self.device)
        attention_mask = torch.cat([img_mask, attention_mask], dim=-1)
        input_shape = attention_mask.size()
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, self.device,
                                                                   attention_mask.dtype)

        x1, x2 = self.vlmo_neck(x1, x2, extended_attention_mask)
        x1 = self.norm(x1)  # B L C
        x1 = self.avgpool(x1.transpose(1, 2))  # B C 1
        x1 = torch.flatten(x1, 1)
        x2 = x2[:, 0, :]
        x = torch.cat([x1, x2], 1)
        x = self.heads_vl(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            # if self.config.is_decoder:
            #     extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
            #         input_shape, attention_mask, device
            #     )
            # else:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        # extended_attention_mask = (1.0 - extended_attention_mask) * torch.iinfo(dtype).min
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats, text_feats = image_feat, text_feat
        # image_feats = concat_all_gather(image_feat)
        # text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # # replace the keys at ptr (dequeue and enqueue)
        # if image_feats.size()[1] == 12:
        #     print(ptr, ptr+batch_size, image_feats.size())
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        # print(self.image_queue[:, 0])
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


class multi_features(nn.Module):
    def __init__(self, model_type='tiny'):
        super(multi_features, self).__init__()
        self.model_type = model_type
        self.stage1_pooling = nn.AvgPool2d(kernel_size=4, stride=4)
        self.stage2_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        if model_type == 'tiny':
            # self.stages1_linear = nn.Linear(192, 768)
            # self.stages2_linear = nn.Linear(384, 768)
            self.stage1_linear = nn.Linear(192, 768)
            self.stage2_linear = nn.Linear(384, 768)
        else:
            self.stages1_linear = nn.Linear(256, 1024)
            self.stages2_linear = nn.Linear(512, 1024)


    def forward(self, features):
        f1, f2, f3, f4 = features
        b = f1.size()[0]
        f1 = f1.view(b, 28, 28, -1).permute(0, 3, 1, 2)
        f2 = f2.view(b, 14, 14, -1).permute(0, 3, 1, 2)
        f1 = self.stage1_pooling(f1)
        f2 = self.stage2_pooling(f2)
        d1, d2 = f1.size()[1], f2.size()[1]
        f1 = f1.view(b, d1, -1).transpose(1, 2)
        f2 = f2.view(b, d2, -1).transpose(1, 2)
        f1 = self.stages1_linear(f1)
        f2 = self.stages2_linear(f2)
        # f1 = self.stage1_linear(f1)
        # f2 = self.stage2_linear(f2)
        # f = torch.cat([f1, f2, f3, f4], dim=1)
        return f1, f2, f3, f4


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=96, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class co_attention(nn.Module):
    def __init__(self, dim, heads=12, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv1 = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv2 = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out1 = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.to_out2 = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, img_feature, text_feature):
        qkv_i = self.to_qkv1(img_feature).chunk(3, dim=-1)
        qkv_t = self.to_qkv2(text_feature).chunk(3, dim=-1)
        q_i, k_i, v_i = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_i)
        q_t, k_t, v_t = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_t)

        dots_i = torch.matmul(q_i, k_t.transpose(-1, -2)) * self.scale

        attn_i = self.attend(dots_i)
        attn_i = self.dropout(attn_i)

        out_i = torch.matmul(attn_i, v_t)
        out_i = rearrange(out_i, 'b h n d -> b n (h d)')

        dots_t = torch.matmul(q_t, k_i.transpose(-1, -2)) * self.scale

        attn_t = self.attend(dots_t)
        attn_t = self.dropout(attn_t)

        out_t = torch.matmul(attn_t, v_i)
        out_t = rearrange(out_t, 'b h n d -> b n (h d)')
        return self.to_out1(out_i), self.to_out2(out_t), attn_t


class pretrain_fusion_block(nn.Module):
    def __init__(self, cross_layer, type='img'):
        super(pretrain_fusion_block, self).__init__()
        self.type = type
        self.sa = cross_layer.attention
        if type == 'img':
            for p in self.sa.parameters():
                p.requires_grad = False

        self.intermediate_vl = cross_layer.intermediate
        self.intermediate_v = copy.deepcopy(cross_layer.intermediate)
        self.output_vl = cross_layer.output
        self.output_v = copy.deepcopy(cross_layer.output)

    def forward(self, img_feature, text_feature, attention_mask=None):
        if self.type == 'img':
            img_feature, output_attention = self.sa(img_feature, attention_mask=attention_mask, output_attentions=True)
            middle_feature = self.intermediate_v(img_feature)
            img_feature = self.output_v(middle_feature, img_feature)
            return img_feature, text_feature, output_attention

        else:
            img_text_feature = torch.cat([img_feature, text_feature], dim=1)
            img_text_feature, output_attention = self.sa(img_text_feature, attention_mask=attention_mask, output_attentions=True)
            middle_feature = self.intermediate_vl(img_text_feature)
            img_text_feature = self.output_vl(middle_feature, img_text_feature)
            img_feature, text_feature = torch.split(img_text_feature, [img_feature.size()[1], text_feature.size()[1]], dim=1)
            return img_feature, text_feature, output_attention


class pretrain_neck(nn.Module):
    def __init__(self, cross_model, depth, type='img'):
        super(pretrain_neck, self).__init__()
        self.type = type
        self.layers = nn.ModuleList([pretrain_fusion_block(cross_model[i], type=self.type) for i in range(depth)])

    def forward(self, img_feature, text_feature, attention_mask=None):
        for layer in self.layers:
            img_feature, text_feature, _ = layer(img_feature, text_feature, attention_mask=attention_mask)
        return img_feature, text_feature

