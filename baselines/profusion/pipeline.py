# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Any, Callable, Dict, List, Optional, Union, OrderedDict

import torch

from diffusers import StableDiffusionXLPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import replace_example_docstring, logging
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

from persongen.data.dataset import tokenize_prompt, encode_tokens, compute_time_ids

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            setattr(self, key, value)

        self.__frozen = True

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __setattr__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setattr__(name, value)

    def __setitem__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setitem__(name, value)


class StableDiffusionPipeline(StableDiffusionXLPipeline):
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
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    ):
        prompt_input_list = tokenize_prompt((self.tokenizer, self.tokenizer_2), prompt)
        # (batch_size, seq_len, dim), (batch_size, pooled_dim)
        prompt_embeds, pooled_prompt_embeds = encode_tokens((self.text_encoder, self.text_encoder_2), prompt_input_list)

        ref_prompt_input_list = tokenize_prompt((self.tokenizer, self.tokenizer_2), ref_prompt)
        # (batch_size, seq_len, dim), (batch_size, pooled_dim)
        ref_prompt_embeds, ref_pooled_prompt_embeds = encode_tokens((self.text_encoder, self.text_encoder_2), ref_prompt_input_list)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        ref_prompt_embeds = ref_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        ref_pooled_prompt_embeds = ref_pooled_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = negative_prompt or ""

            uncond_prompt_input_list = tokenize_prompt((self.tokenizer, self.tokenizer_2), uncond_tokens)
            # (1, seq_len, dim), (1, pooled_dim)
            negative_prompt_embeds, negative_pooled_prompt_embeds = encode_tokens((self.text_encoder, self.text_encoder_2), uncond_prompt_input_list)

        if do_classifier_free_guidance:
            batch_size = prompt_embeds.shape[0]
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds.dtype, device=device)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(dtype=prompt_embeds.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, batch_size * num_images_per_prompt, 1)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, batch_size * num_images_per_prompt)
            # (num_images_per_prompt, seq_len, dim)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            # (num_images_per_prompt, dim)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return {
            'uncond': negative_prompt_embeds if do_classifier_free_guidance else None,
            'concept': prompt_embeds,
            'superclass': ref_prompt_embeds,

            'uncond_pooled': negative_pooled_prompt_embeds if do_classifier_free_guidance else None,
            'concept_pooled': pooled_prompt_embeds,
            'superclass_pooled': ref_pooled_prompt_embeds,
        }

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        add_time_ids = compute_time_ids(
            original_size=torch.tensor([[height, height]]),
            crops_coords_top_left=torch.tensor([[0, 0]]),
            resolution=height
        )
        add_time_ids = add_time_ids.to(self.unet.device).repeat(batch_size, 1)

        return latents, add_time_ids

    def prepare_embeds(self, prompt, ref_prompt, add_time_ids, num_images_per_prompt, device):
        # generate epsilons for sampling
        prompt_embeds_dict = self._encode_prompt(
            prompt, ref_prompt,
            device, num_images_per_prompt,
            do_classifier_free_guidance=True
        )

        if ref_prompt is not None:
            prompt_embeds = torch.cat([
                prompt_embeds_dict['uncond'].clone(),
                prompt_embeds_dict['concept'].clone(),
                prompt_embeds_dict['superclass'].clone(),
            ])
            pooled_prompt_embeds = torch.cat([
                prompt_embeds_dict['uncond_pooled'].clone(),
                prompt_embeds_dict['concept_pooled'].clone(),
                prompt_embeds_dict['superclass_pooled'].clone(),
            ])
            add_time_ids = torch.cat([add_time_ids, add_time_ids, add_time_ids], dim=0)
        else:
            prompt_embeds = torch.cat([
                prompt_embeds_dict['uncond'].clone(),
                prompt_embeds_dict['concept'].clone(),
            ])
            pooled_prompt_embeds = torch.cat([
                prompt_embeds_dict['uncond_pooled'].clone(),
                prompt_embeds_dict['concept_pooled'].clone(),
            ])
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
        return prompt_embeds, added_cond_kwargs

    @torch.no_grad()
    def generate_epsilons(
            self, time, prompt, ref_prompt,
            latent_model_input, prompt_embeds, cross_attention_kwargs, added_cond_kwargs
    ):
        noise_pred_0 = self.unet(
            latent_model_input, time,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs
        ).sample

        if ref_prompt is not None:
            noise_pred_uncond, noise_pred_text_0, noise_pred_ref = noise_pred_0.chunk(3)
        else:
            noise_pred_uncond, noise_pred_text_0 = noise_pred_0.chunk(2)
            noise_pred_ref = noise_pred_uncond
        return noise_pred_text_0, noise_pred_uncond, noise_pred_ref

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            ref_prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 5.0,
            guidance_scale_ref: float = 0.0,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            refine_noise: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            st: int = 1000,
            warm_up_ratio=0.,
            warm_up_start_scale=0.,
            refine_step: int = 0,
            refine_eta: float = 1.,
            refine_guidance_scale: float = 3.0,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, prompt, height, width, callback_steps, None, None
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError("prompt_embeds not supported, please use prompt (string) or a list of prompt")

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents, add_time_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.text_encoder.dtype,
            device,
            generator,
            latents,
        )

        prompt_embeds, added_cond_kwargs = self.prepare_embeds(prompt, ref_prompt, add_time_ids, num_images_per_prompt, device)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if t <= st:
                    # expand the latents if we are doing classifier free guidance
                    if ref_prompt is not None:
                        latent_model_input = torch.cat([latents] * 3)
                    else:
                        latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if ref_prompt is not None or refine_step > 0:
                        # scheduler has to be DDIM !
                        prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                        alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.final_alpha_cumprod
                        variance = self.scheduler._get_variance(t, prev_t)
                        sigma_t = refine_eta * variance ** (0.5)

                        for kdx in range(refine_step):
                            if refine_noise is None:
                                noise = torch.randn_like(latents)
                            else:
                                noise = refine_noise[kdx, i]
                            noise_pred_text_0, noise_pred_uncond, noise_pred_ref = \
                                self.generate_epsilons(time=t,
                                                       prompt=prompt,
                                                       ref_prompt=ref_prompt,
                                                       latent_model_input=latent_model_input,
                                                       prompt_embeds=prompt_embeds,
                                                       cross_attention_kwargs=cross_attention_kwargs,
                                                       added_cond_kwargs=added_cond_kwargs
                                                       )
                            eps = refine_guidance_scale * (noise_pred_text_0 - noise_pred_uncond) + noise_pred_uncond

                            latents = latents - eps * (sigma_t ** 2 * torch.sqrt(1 - alpha_prod_t))/(1 - alpha_prod_t_prev) \
                                      + sigma_t * noise * torch.sqrt((1 - alpha_prod_t) * (2 - 2*alpha_prod_t_prev - sigma_t**2))/(1-alpha_prod_t_prev)

                            if ref_prompt is not None:
                                latent_model_input = torch.cat([latents] * 3)
                            else:
                                latent_model_input = torch.cat([latents] * 2)
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    noise_pred_text_0, noise_pred_uncond, noise_pred_ref = \
                        self.generate_epsilons(time=t,
                                               prompt=prompt, ref_prompt=ref_prompt,
                                               latent_model_input=latent_model_input,
                                               prompt_embeds=prompt_embeds,
                                               cross_attention_kwargs=cross_attention_kwargs,
                                               added_cond_kwargs=added_cond_kwargs
                                               )
                    s_1 = guidance_scale
                    s_2 = guidance_scale_ref

                    noise_pred = noise_pred_uncond + s_1 * (
                            noise_pred_text_0 - noise_pred_uncond) \
                                 + s_2 * (noise_pred_ref - noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

        if output_type == "latent":
            image = latents
        else:
            # 8. Post-processing
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        return StableDiffusionXLPipelineOutput(images=image)
