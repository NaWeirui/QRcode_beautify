from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from io import BytesIO
from typing import Optional
from PIL import Image
from modelscope import snapshot_download
import tempfile
from compel import Compel
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
)

Controlnet_MAP={
    "brightness":"/home/wsco/wyj2/sd模型库/control_v1p_sd15_brightness",
    "hide":"wyj123456/control_v1p_sd15_qrcode_monster",
    "nohide":"wyj123456/controlnet_qrcode_sd15_not_hide_code_point"
}

Model_path_MAP={
    "GhostMix": "wyj123456/GhostMix",
    "dreamlike-diffusion-1.0": "/home/wsco/wyj2/sd模型库/dreamlike-diffusion-1.0",
    "counterfeit": "/home/wsco/wyj2/sd模型库/counterfeit",
    "SCMix": "/home/wsco/wyj2/sd模型库/SCMix",
    "Realistic":"/home/wsco/nwr/qrcode_controlnet_realistic/model/realistic"

}

Lora_MAP={
    "Detail Tweaker":"/home/wsco/nwr/qrcode_controlnet_realistic/lora/add_detail.safetensors",
    "Detail Slider":"/home/wsco/nwr/qrcode_controlnet_realistic/lora/detail_slider_v4.safetensors",
    "None":"None"
}

SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "DPM++ Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True),
    "Heun": lambda config: HeunDiscreteScheduler.from_config(config),
    "Euler a": lambda config: EulerAncestralDiscreteScheduler.from_config(config),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
    "DDIM": lambda config: DDIMScheduler.from_config(config),
    "DEIS": lambda config: DEISMultistepScheduler.from_config(config),
}

app = FastAPI()

class DiffusionControlRequest(BaseModel):
        prompt: str
        image_path:str

@app.post("/run_diffusion_control")
async def run_diffusion_control(request: DiffusionControlRequest):
    try:
        # 解析请求中的参数
        prompt = request.prompt
        image_path = request.image_path
        print("Connected success")
        qrcode_image = Image.open(image_path)
        negative_prompt = "ugly, disfigured, low quality, blurry, nsfw"
        controlnet_path="nohide"
        lora_path="Detail Tweaker"
        model="GhostMix"
        sampler="Euler a"
        guidance_scale: float = 10.0
        controlnet_conditioning_scale: float = 2.0
        num_inference_steps=40
        lora_w=0.8

        # resize_for_condition_image
        input_image = qrcode_image.convert("RGB")
        W, H = input_image.size
        k = float(768) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        qrcode_image = input_image.resize((W, H), resample=Image.LANCZOS)

        model_dir_con = snapshot_download(Controlnet_MAP[controlnet_path],revision='v1.0.0')
        controlnet = ControlNetModel.from_pretrained(
        model_dir_con,
        torch_dtype=torch.float32
        ).to('cpu')
        base_model = snapshot_download(Model_path_MAP[model],revision='v1.0.0')

        # 设置要使用的 CUDA 设备
        torch.cuda.set_device(1)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=torch.float32
        ).to('cuda')

        if lora_path != 'None':
            pipe.unload_lora_weights()
            pipe.load_lora_weights(Lora_MAP[lora_path])
            pipe._lora_scale = lora_w
            state_dict, network_alphas = pipe.lora_state_dict(Lora_MAP[lora_path])

            pipe.load_lora_into_unet(
                state_dict=state_dict,
                network_alphas=network_alphas,
                unet=pipe.unet
            )

            pipe.load_lora_into_text_encoder(
                state_dict=state_dict,
                network_alphas=network_alphas,
                text_encoder=pipe.text_encoder
            )
            compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

        pipe.scheduler = SAMPLER_MAP[sampler](pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()

        out = pipe(
            prompt_embeds=compel_proc(prompt),
            negative_prompt_embeds=compel_proc(negative_prompt),
            image=qrcode_image,
            width=768,
            height=768,
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            num_inference_steps=int(num_inference_steps),
        )
        
         # Assuming out.images[0] is a PIL.Image.Image object
        pil_image = out.images[0]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file_path = temp_file.name
            pil_image.save(temp_file_path)

        # Return the image as a FileResponse
        return FileResponse(temp_file_path, media_type="image/png", filename="result.png")


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=60)