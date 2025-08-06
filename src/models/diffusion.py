from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    StableDiffusionInpaintPipeline,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List

class ModifiedStableDiffusionInpaintPipeline(nn.Module):
    def __init__(
        self,
        pretrained_diffusion_model: str = "stabilityai/stable-diffusion-2-inpainting",
        device: Union[str, torch.device] = "cuda",
        enable_memory_efficient_attention: bool = True,
        enable_cpu_offload: bool = False,
        enable_model_cpu_offload: bool = True,
    ):
        super().__init__()

        # Device setup
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.device.type == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, switching to CPU.")
            self.device = torch.device("cpu")

        # Model loading with enhanced error handling
        try:
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                pretrained_diffusion_model,
                torch_dtype=(
                    torch.float16 if self.device.type == "cuda" else torch.float32
                ),
                use_safetensors=True,
                variant="fp16" if self.device.type == "cuda" else None,
            )

            # Memory optimization
            if enable_memory_efficient_attention:
                self._setup_memory_efficient_attention()

            # Offloading strategies
            if enable_cpu_offload and self.device.type == "cuda":
                self.pipeline.enable_sequential_cpu_offload()
            elif enable_model_cpu_offload and self.device.type == "cuda":
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline.to(self.device)

            self._pipeline_moved = not (enable_cpu_offload or enable_model_cpu_offload)

        except Exception as e:
            raise RuntimeError(f"Pipeline loading failed: {str(e)}")

        # Freeze model parameters
        self._freeze_components()

        # Initialize components
        self.vae = self.pipeline.vae
        self.unet = self.pipeline.unet
        self.text_encoder = self.pipeline.text_encoder
        self.scheduler = self.pipeline.scheduler

    def _setup_memory_efficient_attention(self):
        """Configure memory efficient attention mechanisms"""
        self.pipeline.enable_attention_slicing()
        if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                import xformers

                self.pipeline.enable_xformers_memory_efficient_attention()
            except ImportError:
                print(
                    "xformers not found, memory efficient attention will not be enabled."
                )

    def _freeze_components(self):
        """Freeze all model parameters"""
        for component in [
            self.pipeline.vae,
            self.pipeline.unet,
            self.pipeline.text_encoder,
        ]:
            for param in component.parameters():
                param.requires_grad = False

    def _validate_inputs(
        self,
        input_images: torch.Tensor,
        mask: torch.Tensor,
        condition_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate and preprocess input tensors"""
        # Image validation
        if input_images.dim() != 4 or input_images.shape[1] != 3:
            raise ValueError(
                f"Input images must be (B,3,H,W), got {input_images.shape}"
            )

        # Mask processing
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() != 4:
            raise ValueError(f"Mask must be 3D or 4D, got {mask.dim()}D")

        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Mask must be binary (0 or 1)")

        # Condition embeddings validation
        if condition_embeddings.dim() != 3:
            raise ValueError(
                f"Condition embeddings must be (B,T,D), got {condition_embeddings.shape}"
            )

        # Move tensors to device
        input_images = input_images.to(self.device)
        mask = mask.to(self.device)
        condition_embeddings = condition_embeddings.to(self.device)

        return input_images, mask, condition_embeddings

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space"""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            return latents * self.vae.config.scaling_factor

    def prepare_masked_latents(
        self, images: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare masked image latents and resized mask"""
        masked_images = images * (1 - mask)
        masked_latents = self.encode_images(masked_images)
        mask_latent = F.interpolate(
            mask, size=masked_latents.shape[-2:], mode="nearest"
        )
        return masked_latents, mask_latent

    def forward(
        self,
        input_images: torch.Tensor,
        mask: torch.Tensor,
        condition_embeddings: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute inpainting loss with enhanced features:
        - Better noise handling
        - Flexible timestep control
        - Custom noise injection
        """
        # Input validation and preprocessing
        input_images, mask, condition_embeddings = self._validate_inputs(
            input_images, mask, condition_embeddings
        )

        # Encode images
        latents = self.encode_images(input_images)

        # Generate noise if not provided
        if noise is None:
            noise = torch.randn_like(latents)

        # Set random timesteps if not provided
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.scheduler.num_train_timesteps,
                (input_images.shape[0],),
                device=self.device,
            )

        # Add noise to latents
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Prepare masked components
        masked_latents, mask_latent = self.prepare_masked_latents(input_images, mask)

        # UNet forward pass
        unet_input = torch.cat([noisy_latents, masked_latents, mask_latent], dim=1)
        noise_pred = self.unet(
            unet_input, timesteps, encoder_hidden_states=condition_embeddings
        ).sample

        # Compute masked MSE loss
        loss = F.mse_loss(
            noise_pred * mask_latent, noise * mask_latent, reduction="sum"
        )
        loss = loss / (mask_latent.sum() + 1e-8)  # Normalize by mask area

        return loss

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel space with proper scaling"""
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents).sample
        return torch.clamp((images + 1) / 2, 0, 1)  # [-1,1] -> [0,1]

    @torch.no_grad()
    def generate(
        self,
        input_images: torch.Tensor,
        mask: torch.Tensor,
        condition_embeddings: torch.Tensor,
        num_inference_steps: int = 50,
        strength: float = 1.0,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Pure inpainting generation without classifier-free guidance.
        
        Args:
            input_images: Input images tensor [B,3,H,W] 
            mask: Binary mask tensor [B,1,H,W] where 1=preserved areas, 0=areas to inpaint
            condition_embeddings: Text embeddings for prompts [B,77,768]
            num_inference_steps: Number of denoising steps
            strength: Controls noise level (1.0=full generation, 0.0=original image)
            return_intermediates: Whether to return intermediate latent states
        
        Returns:
            Generated images or tuple of (images, intermediates)
        """
        # Validate input shapes and move to correct device
        input_images, mask, condition_embeddings = self._validate_inputs(
            input_images, mask, condition_embeddings
        )
        
        # ========== IMAGE ENCODING ==========
        # Encode original image to latent space (B,4,H/8,W/8)
        original_latents = self.encode_images(input_images)
        
        # Prepare masked version of image latents and resized mask
        masked_latents, mask_latent = self.prepare_masked_latents(input_images, mask)
        
        # ========== LATENT INITIALIZATION ==========
        noise = torch.randn_like(original_latents)
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Strength controls how much noise we add initially
        if strength < 1.0:
            # When strength < 1, we start from partially noised original image
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            timesteps = timesteps[-init_timestep:]
            latent_timestep = timesteps[:1]
            latents = self.scheduler.add_noise(original_latents, noise, latent_timestep)
        else:
            # Full generation from pure noise
            latents = noise
        
        intermediates = []
        
        # ========== DENOISING LOOP ==========
        for i, t in enumerate(timesteps):
            # Scale latent inputs according to scheduler rules
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            
            # Prepare UNet input by concatenating:
            # 1. Noisy latents
            # 2. Masked image latents (context)
            # 3. Resized binary mask
            unet_input = torch.cat([latent_model_input, masked_latents, mask_latent], dim=1)
            
            # Predict noise residual with UNet
            noise_pred = self.unet(
                unet_input,
                t,
                encoder_hidden_states=condition_embeddings
            ).sample
            
            # Compute previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            if i < len(timesteps) - 1:
                original_noisy = self.scheduler.add_noise(
                    original_latents, noise, timesteps[i + 1:i + 2]
                )
            else:
                original_noisy = original_latents
            
            # Blend between generated and original content using mask
            latents = (
                original_noisy * (1 - mask_latent)
                + latents * mask_latent
            )
        
            # Store intermediate results if requested
            if return_intermediates:
                intermediates.append(latents.clone())
        
        # ========== FINAL DECODING ==========
        # Decode latents to pixel space and normalize to [0,1]
        images = self.decode_latents(latents)
        
        return (images, intermediates) if return_intermediates else images