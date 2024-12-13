# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Classes to support Encoder-Decoder architectures"""


import copy
import gc
import inspect
import os
import tempfile
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, CTCLoss, MSELoss, HuberLoss

from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel, AutoModelForCausalLM
from .configuration_encoder_decoder import EncoderDecoderConfig

from ..bert.configuration_bert import BertConfig
from ..bert.modeling_bert import BertModel, BertLMHeadModel

from .modeling_vae import VAE

from diffusers import UNet2DModel, UNet2DConditionModel
from diffusers import DDPMScheduler, DDIMScheduler
from torchmetrics.functional.pairwise import pairwise_euclidean_distance


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "EncoderDecoderConfig"

DEPRECATION_WARNING = (
    "Version v4.12.0 introduces a better way to train encoder-decoder models by computing the loss inside the"
    " encoder-decoder framework rather than in the decoder itself. You may observe training discrepancies if"
    " fine-tuning a model trained with versions anterior to 4.12.0. The decoder_input_ids are now created based on the"
    " labels, no need to pass them yourself anymore."
)

ENCODER_DECODER_START_DOCSTRING = r"""
"""

ENCODER_DECODER_INPUTS_DOCSTRING = r"""
"""


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


@add_start_docstrings(ENCODER_DECODER_START_DOCSTRING)
class BaselineModel(PreTrainedModel):
    r"""
    [`EncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with one
    of the base model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    """
    config_class = EncoderDecoderConfig
    base_model_prefix = "encoder_decoder"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
        tokenizer = None,
    ):
        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # initialize with config
        super().__init__(config)

        if encoder is None:
            from ..auto.modeling_auto import AutoModel

            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            from ..auto.modeling_auto import AutoModelForCausalLM

            decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.encoder = encoder
        if False:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.decoder = decoder

        self.tokenizer = tokenizer
        self.config.blank_token_id = tokenizer.convert_tokens_to_ids(["[BLANK]"])[0]

        #self.latent_size = self.vae.latent_size
        #self.latent_size = 512
        self.num_diffusion_steps = 20
        self.num_train_steps = 1
        self.num_inference_steps = 2

        end_prob = 0.3
        start_prob = end_prob / self.num_diffusion_steps
        self.probs = torch.linspace(start_prob, end_prob, self.num_diffusion_steps)
        #self.probs = torch.full((self.num_diffusion_steps,), 1 / self.num_diffusion_steps)

        self.prob_random = 0.1

        """
        # For noise scheduling
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
        )
        #self.noise_scheduler = DDIMScheduler(num_train_timesteps=self.num_diffusion_steps)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        self.vocab_size = self.config.decoder.vocab_size
        self.simplex_value = 5.

        self.vocab_to_embed = nn.Linear(self.vocab_size, self.config.decoder.hidden_size, bias=True)

        self.guidance_scale = 7.5
        self.p_uncond = 0.2
        """

        self.time_embeddings = nn.Embedding(self.num_diffusion_steps, self.decoder.config.hidden_size)
        self.use_time_emb = False

        # Init weights
        if False:
            self.time_embeddings.weight.data.normal_(mean=0.0, std=self.config.decoder.initializer_range)
            self.vocab_to_embed.weight.data.normal_(mean=0.0, std=self.config.decoder.initializer_range)
            self.vocab_to_embed.weight.data.zero_()

        #self.from_latent = nn.Linear(self.latent_size, self.config.decoder.hidden_size, bias=False)
        #self.to_latent = nn.Linear(self.config.decoder.hidden_size, self.latent_size, bias=False)

        #self.hidden_to_emb = nn.Linear(self.config.decoder.hidden_size, self.config.decoder.hidden_size, bias=False)

        self.dropout = nn.Dropout(p=0.1)

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # encoder outputs might need to be projected to different dimension for decoder
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            pass
            #raise ValueError(
            #    f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            #)

        decoder_signature = set(inspect.signature(self.decoder.forward).parameters.keys())
        if "encoder_hidden_states" not in decoder_signature:
            raise ValueError(
                "The selected decoder is not prepared for the encoder hidden states to be passed. Please see the "
                "following discussion on GitHub: https://github.com/huggingface/transformers/issues/23350"
            )

        # tie encoder, decoder weights if config set accordingly
        self.tie_weights()

    def tie_weights(self):
        # tie encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie encoder and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def _set_gradient_checkpointing(self, module, value=False):
        # call both encoder and decoder function on gradient checkpointing
        self.encoder._set_gradient_checkpointing(module, value=value)
        self.decoder._set_gradient_checkpointing(module, value=value)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import EncoderDecoderModel

        >>> model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")
        ```"""

        from_tf = kwargs.pop("from_tf", False)
        if from_tf:
            from transformers import TFEncoderDecoderModel

            # a workaround to load from tensorflow checkpoint
            # Using `_tf_model` won't work, because the weight names in the encoder/decoder of `_tf_model` get
            # extended before saving those components. For example, The name of `_tf_model.encoder.vit` is
            # `[top model name]/encoder/vit`, but the name of `tf_model.encoder.vit` is `[top model name]/vit`. The
            # [top model name] is handled (stripped) by the conversion method, and the former case gets extra `encoder`,
            # which should not occur when we want to save the components alone.
            # There was a (very) ugly potential fix, which wasn't integrated to `transformers`: see
            #   https://github.com/huggingface/transformers/pull/13222/commits/dbb3c9de76eee235791d2064094654637c99f36d#r697304245
            #   (the change in `src/transformers/modeling_tf_utils.py`)
            _tf_model = TFEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            config = _tf_model.config

            # Using `tf_model` instead
            encoder = _tf_model.encoder.__class__(_tf_model.config.encoder)
            decoder = _tf_model.decoder.__class__(_tf_model.config.decoder)
            # Make sure models are built
            encoder(encoder.dummy_inputs)
            decoder(decoder.dummy_inputs)

            # Get the variable correspondence between `_tf_model` and `encoder` and `decoder`
            encoder_variables = {}
            for v in encoder.trainable_variables + encoder.non_trainable_variables:
                encoder_variables["/".join(v.name.split("/")[1:])] = v
            decoder_variables = {}
            for v in decoder.trainable_variables + decoder.non_trainable_variables:
                decoder_variables["/".join(v.name.split("/")[1:])] = v

            _encoder_variables = {}
            for v in _tf_model.encoder.trainable_variables + _tf_model.encoder.non_trainable_variables:
                _encoder_variables["/".join(v.name.split("/")[2:])] = v
            _decoder_variables = {}
            for v in _tf_model.decoder.trainable_variables + _tf_model.decoder.non_trainable_variables:
                _decoder_variables["/".join(v.name.split("/")[2:])] = v

            # assign weight values to `encoder` and `decoder` from `_tf_model`
            for name, v in encoder_variables.items():
                v.assign(_encoder_variables[name])
            for name, v in decoder_variables.items():
                v.assign(_decoder_variables[name])

            tf_model = TFEncoderDecoderModel(encoder=encoder, decoder=decoder)

            # Deal with `enc_to_dec_proj`
            if hasattr(_tf_model, "enc_to_dec_proj"):
                tf_model(tf_model.dummy_inputs)
                tf_model.enc_to_dec_proj.kernel.assign(_tf_model.enc_to_dec_proj.kernel)
                tf_model.enc_to_dec_proj.bias.assign(_tf_model.enc_to_dec_proj.bias)

            with tempfile.TemporaryDirectory() as tmpdirname:
                encoder_dir = os.path.join(tmpdirname, "encoder")
                decoder_dir = os.path.join(tmpdirname, "decoder")
                tf_model.encoder.save_pretrained(encoder_dir)
                tf_model.decoder.save_pretrained(decoder_dir)

                if hasattr(tf_model, "enc_to_dec_proj"):
                    enc_to_dec_proj_weight = torch.transpose(
                        torch.from_numpy(tf_model.enc_to_dec_proj.kernel.numpy()), 1, 0
                    )
                    enc_to_dec_proj_bias = torch.from_numpy(tf_model.enc_to_dec_proj.bias.numpy())

                del _tf_model
                del tf_model
                gc.collect()

                model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                    encoder_dir, decoder_dir, encoder_from_tf=True, decoder_from_tf=True
                )
                # This is only for copying some specific attributes of this particular model.
                model.config = config

                if hasattr(model, "enc_to_dec_proj"):
                    model.enc_to_dec_proj.weight.data = enc_to_dec_proj_weight
                    model.enc_to_dec_proj.bias.data = enc_to_dec_proj_bias

                return model

        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for EncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        tokenizer = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        r"""
        ```"""

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = BertConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = BertLMHeadModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)
            if tokenizer is not None:
                encoder.resize_token_embeddings(len(tokenizer))

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = BertConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = BertLMHeadModel.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)
            if tokenizer is not None:
                decoder.resize_token_embeddings(len(tokenizer))

        # instantiate config with corresponding kwargs
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
        #config.tie_weights = True
        return cls(encoder=encoder, decoder=decoder, config=config, tokenizer=tokenizer)


    @add_start_docstrings_to_model_forward(ENCODER_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        null_input_ids: Optional[torch.LongTensor] = None,
        null_attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        Returns:

        Examples:

        ```python
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        inp_shape = input_ids.shape
        bs, msl = inp_shape
        device = input_ids.device
        visible_mask = attention_mask * 0 + 1

        logits = None
        #loss_ctc_fct = CTCLoss(zero_infinity=True)
        loss_ctc_fct = CTCLoss(blank=self.config.blank_token_id, zero_infinity=True)
        loss_ctc = torch.tensor(0.0, device=device)
        loss_ce_fct = CrossEntropyLoss()
        loss_ce = torch.tensor(0.0, device=device)

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs_encoder,
        )
        encoder_last_hidden_state = encoder_outputs.hidden_states[-1]

        """
        if self.training:

            all_dec_inputs = [decoder_input_ids]
            current_dec_inputs = decoder_input_ids
            for ts in range(self.num_diffusion_steps):
                mask = torch.bernoulli(torch.full(inp_shape, self.probs[ts], device=device)).long()
                #mask = torch.bernoulli(torch.full(inp_shape, self.probs[ts], device=device)).long() * decoder_attention_mask
                current_dec_inputs = current_dec_inputs * (1 - mask) + self.config.mask_token_id * mask
                all_dec_inputs = [current_dec_inputs] + all_dec_inputs

            num_calc_loss = 5 - 1
            #num_calc_loss = self.num_diffusion_steps
            ts_for_loss = torch.randperm(self.num_diffusion_steps - 1)
            #ts_for_loss = torch.randperm(self.num_diffusion_steps)
            ts_for_loss = ts_for_loss[:num_calc_loss]

            cur_inputs = all_dec_inputs[0]
            for ts in range(self.num_diffusion_steps):

                ts_tensor = torch.full((bs,), ts, device=device)
                decoder_outputs = self.decoder(
                    timestep_ids=ts_tensor if self.use_time_emb else None,
                    #input_ids=input_ids,
                    #another_input_ids=current_dec_inputs,
                    attention_mask=visible_mask,
                    #token_type_ids=token_type_ids,
                    #encoder_hidden_states=encoder_last_hidden_state,
                    #encoder_attention_mask=attention_mask,
                    #inputs_embeds=cur_last_hidden_state + encoder_last_hidden_state,
                    inputs_embeds=encoder_last_hidden_state,
                    input_ids=cur_inputs,
                    output_hidden_states=True,
                    **kwargs_decoder,
                )
                logits = decoder_outputs.logits
                cur_inputs = logits.argmax(-1)
                cur_inputs = self.replace_with_random_words(cur_inputs)
                if ts in ts_for_loss or ts == self.num_diffusion_steps - 1:
                    log_probs = logits.log_softmax(-1)
                    loss_ctc += loss_ctc_fct(log_probs.transpose(0,1), all_dec_inputs[ts+1], visible_mask.sum(1), decoder_attention_mask.sum(1))


        else:
            current_dec_inputs = decoder_input_ids * 0 + self.config.mask_token_id
            for ts in range(self.num_diffusion_steps):
                ts_tensor = torch.full((bs,), ts, device=device)
                decoder_outputs = self.decoder(
                    timestep_ids=ts_tensor if self.use_time_emb else None,
                    #input_ids=input_ids,
                    #another_input_ids=current_dec_inputs,
                    attention_mask=visible_mask,
                    #token_type_ids=token_type_ids,
                    #encoder_hidden_states=encoder_last_hidden_state,
                    #encoder_attention_mask=attention_mask,
                    #inputs_embeds=cur_last_hidden_state + encoder_last_hidden_state,
                    inputs_embeds=encoder_last_hidden_state,
                    input_ids=current_dec_inputs,
                    output_hidden_states=True,
                    **kwargs_decoder,
                )
                logits = decoder_outputs.logits
                current_dec_inputs = logits.argmax(-1)
        """

        if self.training:

            all_steps = self.forward_noise(decoder_input_ids)

            #for ts in range(self.num_diffusion_steps):
            for ts in torch.randperm(self.num_diffusion_steps)[:self.num_train_steps]:
                #tmp_input_ids = all_steps[ts]

                    ts_tensor = torch.full((bs,), ts, device=device)
                    decoder_outputs = self.decoder(
                        timestep_ids=ts_tensor if self.use_time_emb else None,
                        attention_mask=visible_mask,
                        #encoder_hidden_states=encoder_last_hidden_state,
                        #encoder_attention_mask=attention_mask,
                        inputs_embeds=encoder_last_hidden_state,
                        input_ids=self.replace_with_random_words(all_steps[ts]),
                        #input_ids=self.replace_with_random_words(tmp_input_ids),
                        output_hidden_states=True,
                        **kwargs_decoder,
                    )
                    logits = decoder_outputs.logits
                    log_probs = logits.log_softmax(-1)
                    loss_ctc += loss_ctc_fct(log_probs.transpose(0,1), decoder_input_ids, visible_mask.sum(1), decoder_attention_mask.sum(1))
                    #loss_ce += loss_ce_fct(logits.view(-1, self.config.decoder.vocab_size), decoder_input_ids.view(-1))

                ##for _ in range(3):
                #    if ts + 1 >= self.num_diffusion_steps:
                #        break
                #    ts += 1
                #    tmp_output_ids = logits.argmax(-1)
                #    all_steps = self.forward_noise(tmp_output_ids)
                #    tmp_input_ids = all_steps[ts]

        else:
            tmp_input_ids = decoder_input_ids * 0 + self.config.mask_token_id
            for ts in range(self.num_diffusion_steps):
                ts_tensor = torch.full((bs,), ts, device=device)
                decoder_outputs = self.decoder(
                    timestep_ids=ts_tensor if self.use_time_emb else None,
                    attention_mask=visible_mask,
                    #encoder_hidden_states=encoder_last_hidden_state,
                    #encoder_attention_mask=attention_mask,
                    inputs_embeds=encoder_last_hidden_state,
                    input_ids=tmp_input_ids,
                    output_hidden_states=True,
                    **kwargs_decoder,
                )
                logits = decoder_outputs.logits
                tmp_output_ids = logits.argmax(-1)
                #tmp_output_ids = self.remove_blank_token(tmp_output_ids)

                if ts != self.num_diffusion_steps - 1:
                    all_steps = self.forward_noise(tmp_output_ids)
                    tmp_input_ids = all_steps[ts + 1]


        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss_ctc,
            loss_ce=loss_ce,
            logits=logits,
            #past_key_values=decoder_outputs.past_key_values,
            #decoder_hidden_states=decoder_outputs.hidden_states,
            #decoder_attentions=decoder_outputs.attentions,
            #cross_attentions=decoder_outputs.cross_attentions,
            #encoder_last_hidden_state=encoder_outputs.hidden_states[-1],
            #encoder_hidden_states=encoder_outputs.hidden_states,
            #encoder_attentions=encoder_outputs.attentions,
        )

    def forward_noise(self, input_ids):
        all_steps = []
        tmp_input_ids = input_ids.clone()
        for ts in range(self.num_diffusion_steps):
            mask = torch.bernoulli(torch.full(input_ids.shape, self.probs[ts], device=input_ids.device)).long()
            tmp_input_ids = tmp_input_ids * (1 - mask) + self.config.mask_token_id * mask
            all_steps = [tmp_input_ids] + all_steps

        return all_steps

    def replace_with_random_words(self, input_ids):
        indices_mask = torch.where(input_ids == self.config.mask_token_id, True, False)
        indices_random = torch.bernoulli(torch.full(input_ids.shape, self.prob_random, device=input_ids.device)).bool() & ~indices_mask
        random_words = torch.randint(self.config.decoder.vocab_size, input_ids.shape, dtype=torch.long, device=input_ids.device)
        input_ids[indices_random] = random_words[indices_random]
        return input_ids

    def remove_blank_token(self, all_input_ids):
        all_new_input_ids = torch.full_like(all_input_ids, self.config.pad_token_id)
        for i, input_ids in enumerate(all_input_ids):
            new_input_ids = [ii for ii in input_ids if ii != self.config.blank_token_id]
            all_new_input_ids[i, :len(new_input_ids)] = torch.tensor(new_input_ids)
        return all_new_input_ids

    def remove_special_tokens_and_repetition(self, all_input_ids):
        all_new_input_ids = torch.full_like(all_input_ids, self.config.pad_token_id)
        for i, input_ids in enumerate(all_input_ids):
            new_input_ids = input_ids[input_ids != self.config.blank_token_id]
            all_new_input_ids[i, :len(new_input_ids)] = new_input_ids
        return all_new_input_ids


    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past_key_values, beam_idx)
