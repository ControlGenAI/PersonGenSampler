from typing import Any, Dict, List, Optional, Union

import numpy as np

import torch

from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput

from transformers import (
    CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
)

from .attention_processor import AttnProcessorPhotoswap
from .utils import get_attention_map, SaveOutput


class StableDiffusionPipelineMultiStage(StableDiffusionPipeline):
    # noinspection PyMethodOverriding
    def _encode_prompt(
            self,
            prompt,
            ref_prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        prompt_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_input_ids = prompt_input.input_ids.to(device)
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = prompt_input_ids.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(prompt_input_ids, attention_mask=attention_mask)[0]
        ref_prompt_inputs = self.tokenizer(
            ref_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        ref_prompt_input_ids = ref_prompt_inputs.input_ids.to(device)
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = ref_prompt_input_ids.attention_mask.to(device)
        else:
            attention_mask = None
        ref_prompt_embeds = self.text_encoder(ref_prompt_input_ids, attention_mask=attention_mask)[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        ref_prompt_embeds = ref_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(num_images_per_prompt, seq_len, -1)

        ref_prompt_embeds = ref_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        ref_prompt_embeds = ref_prompt_embeds.view(num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
                uncond_tokens = ""
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=prompt_embeds.shape[1],
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(device), attention_mask=attention_mask)[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(num_images_per_prompt, seq_len, -1)

        return {'uncond': negative_prompt_embeds if do_classifier_free_guidance else None,
                'concept': prompt_embeds,
                'superclass': ref_prompt_embeds}

    def prepare_inputs(
            self, height, width, prompt, negative_prompt, negative_prompt_embeds,
            num_inference_steps, num_images_per_prompt, generator, latents, eta, dtype
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, 1, negative_prompt, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            self.batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            self.batch_size = len(prompt)
        else:
            raise ValueError("prompt_embeds not supported, please use prompt (string) or a list of prompt")

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.unet.device)
        self.timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        latents = self.prepare_latents(
            self.batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            self.unet.device,
            generator,
            latents,
        )
        self.extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        self.num_warmup_steps = len(self.timesteps) - num_inference_steps * self.scheduler.order
        return latents

    @staticmethod
    def check_guidance_scale(guidance_scale, step_number):
        if isinstance(guidance_scale, float):
            return guidance_scale
        else:
            return guidance_scale[step_number]

    @torch.no_grad()
    def prepare_output(self, latents, output_type):
        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            ref_prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: Union[float, List[float]] = 7.0,
            guidance_scale_ref: Union[float, List[float]] = 0.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            st: int = 1000,
            change_step: int = -1,
            scale=1.0,
            return_deltas=False
    ):
        prompt_embeds_dict = self._encode_prompt(
            prompt, ref_prompt,
            self.unet.device, num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        latents = self.prepare_inputs(
            height, width, prompt, negative_prompt, negative_prompt_embeds, num_inference_steps,
            num_images_per_prompt, generator, latents, eta, prompt_embeds_dict['concept'].dtype
        )
        deltas = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.timesteps):
                if t <= st:
                    prompt_embeds = torch.cat([
                        prompt_embeds_dict['uncond'].clone(), prompt_embeds_dict['concept'].clone(), prompt_embeds_dict['superclass'].clone(),
                    ])
                    latent_model_input = torch.cat([latents] * 3)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    noise_pred = self.unet(
                        latent_model_input, t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs
                    ).sample
                    noise_pred_uncond, noise_concept, noise_superclass = noise_pred.chunk(3)

                    # (batch_size, unet.config.sample_size, unet.config.sample_size)
                    delta = torch.linalg.norm(noise_concept - noise_superclass, dim=1)
                    deltas.append(delta.cpu().numpy())

                    s_1 = self.check_guidance_scale(guidance_scale, i)
                    s_2 = self.check_guidance_scale(guidance_scale_ref, i)
                    if i >= change_step:
                        noise_pred = (
                            noise_pred_uncond
                            + s_1 * (noise_concept - noise_pred_uncond)
                            + s_2 * (noise_superclass - noise_pred_uncond)
                        )
                    else:
                        noise_pred = noise_pred_uncond + (s_1 + s_2) * (noise_superclass - noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents, **self.extra_step_kwargs).prev_sample

                    if i == len(self.timesteps) - 1 or ((i + 1) > self.num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
        image = self.prepare_output(latents, output_type)
        if return_deltas:
            return image, np.stack(deltas, axis=1)
        return image


class StableDiffusionPipelinePhotoswap(StableDiffusionPipelineMultiStage):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            feature_extractor: CLIPFeatureExtractor,
            image_encoder: CLIPVisionModelWithProjection = None,
            requires_safety_checker: bool = False,
            safety_checker=None
    ):
        super().__init__(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                         scheduler=scheduler, safety_checker=safety_checker, image_encoder=image_encoder,
                         feature_extractor=feature_extractor, requires_safety_checker=requires_safety_checker)

        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            attn_procs[name] = AttnProcessorPhotoswap()
        self.unet.set_attn_processor(attn_procs)

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            ref_prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: Union[float, List[float]] = 7.0,
            guidance_scale_ref: Union[float, List[float]] = 7.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            st: int = 1000,
            self_attn_map_step: int = 15,
            self_attn_feat_step: int = 1,
            cross_attn_map_step: int = 10
    ):
        prompt_embeds_dict = self._encode_prompt(
            prompt, ref_prompt,
            self.unet.device, num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        latents = self.prepare_inputs(
            height, width, prompt, negative_prompt, negative_prompt_embeds, num_inference_steps,
            num_images_per_prompt, generator, latents, eta, prompt_embeds_dict['concept'].dtype
        )
        cross_attention_kwargs = cross_attention_kwargs or {}

        superclass_latents = latents.clone()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.timesteps):
                if t <= st:
                    prompt_embeds = torch.cat([
                        prompt_embeds_dict['uncond'].clone(), prompt_embeds_dict['concept'].clone(),
                        prompt_embeds_dict['uncond'].clone(), prompt_embeds_dict['superclass'].clone()
                    ])

                    latent_model_input = torch.cat([latents] * 4)
                    latent_model_input[2 * num_images_per_prompt: 4 * num_images_per_prompt] = torch.cat([superclass_latents] * 2).clone()
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    cross_attention_kwargs['change_self_attn_feat'] = False
                    cross_attention_kwargs['change_self_attn_map'] = False
                    cross_attention_kwargs['change_cross_attn_map'] = False

                    if i < self_attn_feat_step:
                        cross_attention_kwargs['change_self_attn_feat'] = True
                    if i < self_attn_map_step:
                        cross_attention_kwargs['change_self_attn_map'] = True
                    if i < cross_attn_map_step:
                        cross_attention_kwargs['change_cross_attn_map'] = True

                    cross_attention_kwargs['num_images_per_prompt'] = num_images_per_prompt

                    noise_pred = self.unet(
                        latent_model_input, t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs
                    ).sample
                    noise_um, noise_mix, noise_us, noise_sup = noise_pred.chunk(4)

                    s_1 = self.check_guidance_scale(guidance_scale, i)
                    s_2 = self.check_guidance_scale(guidance_scale_ref, i)

                    noise_mix = noise_um + s_1 * (noise_mix - noise_um)
                    noise_sup = noise_us + s_2 * (noise_sup - noise_us)

                    superclass_latents = self.scheduler.step(noise_sup.clone(), t, superclass_latents.clone(), **self.extra_step_kwargs).prev_sample
                    latents = self.scheduler.step(noise_mix, t, latents, **self.extra_step_kwargs).prev_sample

                    if i == len(self.timesteps) - 1 or ((i + 1) > self.num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        return self.prepare_output(latents, output_type)


class StableDiffusionPipelineCrossAttnMasked(StableDiffusionPipelineMultiStage):
    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            ref_prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: Union[float, List[float]] = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            st: int = 1000,
            return_one_image: bool = True,
            change_step: int = 25,
            quantile: float = 0.7,
            out_sg: Optional[List[float]] = (0.0, 7.0),
            inner_sg: Optional[List[float]] = None
    ):
        # Here we assume that in the ref_prompt superclass name follows the placeholder so
        #   a placeholder index in the prompt corresponds to the superclass name index in the ref_prompt
        [target_token] = self.tokenizer.encode('sks', add_special_tokens=False)
        prompt_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        ).input_ids.view(-1)
        word_id = torch.argwhere(torch.eq(prompt_input_ids, target_token)).item()

        prompt_embeds_dict = self._encode_prompt(
            prompt, ref_prompt,
            self.unet.device, num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        latents = self.prepare_inputs(
            height, width, prompt, negative_prompt, negative_prompt_embeds, num_inference_steps,
            num_images_per_prompt, generator, latents, eta, prompt_embeds_dict['concept'].dtype
        )

        masks = []
        save_output = SaveOutput()
        for name, layer in self.unet.named_modules():
            if 'attn2.to_q' in name or 'attn2.to_k' in name:
                save_output.register(layer, name)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.timesteps):
                if t <= st:
                    prompt_embeds = torch.cat([
                        prompt_embeds_dict['uncond'].clone(),
                        prompt_embeds_dict['concept'].clone(),
                        prompt_embeds_dict['superclass'].clone(),
                    ])
                    latent_model_input = torch.cat([latents] * 3)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    noise_pred = self.unet(
                        latent_model_input, t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs
                    ).sample

                    amap = get_attention_map(self.unet, save_output, word_id, id_in_subbatch=2, final_size=(64, 64))
                    save_output.clear()

                    noise_um, noise_concept, noise_superclass = noise_pred.chunk(3)

                    s_1 = self.check_guidance_scale(guidance_scale, i)

                    if i >= change_step and quantile is not None:
                        bs, num_channels, _, _ = noise_concept.shape

                        # (bs, final_size[0], final_size[1])
                        mask = amap > torch.quantile(amap.view(bs, -1).to(torch.float32), quantile, dim=1)[:, None, None]
                        masks.append(mask.cpu())

                        mask = mask[:, None].repeat_interleave(num_channels, dim=1)

                        noise_concept[..., ~mask] = (
                            noise_um[..., ~mask]
                            + s_1 * (noise_superclass[..., ~mask] - noise_um[..., ~mask])
                        )
                        if inner_sg[0] and inner_sg[1]:
                            noise_concept[..., mask] = (
                                noise_um[..., mask]
                                + inner_sg[0] * (noise_concept[..., mask] - noise_um[..., mask])
                                + inner_sg[1] * (noise_superclass[..., mask] - noise_um[..., mask])
                            )
                        else:
                            noise_concept[..., mask] = (
                                noise_um[..., mask]
                                + s_1 * (noise_concept[..., mask] - noise_um[..., mask])
                            )
                    else:
                        noise_concept = (
                            noise_um
                            + out_sg[0] * (noise_concept - noise_um)
                            + out_sg[1] * (noise_superclass - noise_um)
                        )

                    latents = self.scheduler.step(noise_concept, t, latents.clone(), **self.extra_step_kwargs).prev_sample
                    if i == len(self.timesteps) - 1 or ((i + 1) > self.num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        save_output.unregister()

        if return_one_image:
            return self.prepare_output(latents, output_type)
        return self.prepare_output(latents, output_type), torch.stack(masks, dim=1)
