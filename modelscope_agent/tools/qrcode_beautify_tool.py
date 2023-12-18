import os
import time
import json
import pandas as pd
import requests
from io import BytesIO
from modelscope_agent.tools.tool import Tool, ToolSchema
from pydantic import ValidationError
from requests.exceptions import RequestException, Timeout

from modelscope import snapshot_download
import torch
from PIL import Image
import io
# import qrcode
from requests import get
# from compel import Compel
import numpy as np

import requests
from pathlib import Path

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

# url = os.environ.get('QRCODEAPI_URL', '')
url = "https://prime-just-monkey.ngrok-free.app/run_diffusion_control"

class QrcodeBeautify(Tool):    
    description = 'QRcode是一个美化二维码的工具，输入用户上传二维码并输入要求，生成美化后的二维码。'
    name = 'QRcode_beautify'
    parameters: list = [{
        'name': 'prompt',
        'description': '用户输入提示词',
        'required': True,
    },
    {
        'name': 'image',
        'description': '用户上传二维码图片路径',
        'required': True,
    }
    ]

    def _local_call(self, *args, **kwargs):
        prompt = kwargs['prompt']
        image = kwargs['image']
        print(image)
        file_path = self.state['file_path']
        file = self.state['file']
        return {'result': f'已完成ECS实例ID为{prompt}的续费，续费时长{file_path}月'}
    
    def _remote_call(self, *args, **kwargs):
        prompt = kwargs['pompt']
        image = kwargs['image']
        print(image)
        file_path = self.state['file_path']
        file = self.state['file']
        return {'result': f'已完成ECS实例ID为{prompt}的续费，续费时长{file_path}月'}

    def __init__(self, cfg={}):
        self.cfg = cfg.get(self.name, {})
        try:
            all_param = {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters
            }
            self.tool_schema = ToolSchema(**all_param)
        except ValidationError:
            raise ValueError(f'Error when parsing parameters of {self.name}')

        # self._str = self.tool_schema.model_dump_json()
        self._function = self.parse_pydantic_model_to_openai_function(
            all_param)




    def __call__(self, *args, **kwargs):

        prompt = kwargs['prompt']
        # image_path = kwargs['image']
        image_path = self.state['file_path']
        if image_path is None:
            return "请上传二维码图片"
        data = {
            "prompt": prompt
        }

        local_file = Path(image_path)
        content = local_file.read_bytes()
        response = requests.post(url, files={'file': BytesIO(content)}, data=data)

        # 检查响应
        if response.status_code == 200:
        
            # 获取文件名
            filename = response.headers["Content-Disposition"].split("filename=")[1].strip('\"')
            # 保存文件
            # with open("/home/wsco/zcg/project/qrcode_beautify_2.0/facechain_agent/temp/"+filename, "wb") as file:
            #     file.write(response.content)    
            print(f"File '{filename}' received successfully.")
            # delete_path = "/home/wsco/zcg/project/qrcode/temp/delete.png"
            delete_path = os.getenv("DELETEPATH", "/home/wsco/zcg/project/qrcode_beautify/temp/delete.png")
            delete_path = delete_path.replace('.', str(np.random.randint(1,999)) + '.')
            Image.open(BytesIO(response.content)).save(delete_path)
            # return {'result' : f'![]({delete_path})'}
            return {'result' : f'{delete_path}'}
        else:
            # 请求失败
            print("Error:", response.status_code, response.text)
            return {'result' : '无法连接到服务器，错误信息：' + str(response.status_code) + response.text}
        
        # data = {
        #     "prompt" : prompt,
        #     "image_path": image_path
        # }
        # response = requests.post(url,json=data)
        # if response.status_code == 200:
        #     # 将图像数据读取为 PIL 图像对象
        #     result = Image.open(BytesIO(response.content))

        #     return result
        # else:
        #     print(f"Request failed with status code: {response.status_code}\n{response}")
        
        # qrcode_image = Image.open(image_path)
        # negative_prompt = "ugly, disfigured, low quality, blurry, nsfw"

        # controlnet_path="nohide"
        # lora_path="Detail Tweaker"
        # model="GhostMix"
        # sampler="Euler a"

        # guidance_scale: float = 10.0
        # controlnet_conditioning_scale: float = 2.0
        # num_inference_steps=40
        # lora_w=0.8
        
        # # resize_for_condition_image
        # input_image = qrcode_image.convert("RGB")
        # W, H = input_image.size
        # k = float(768) / min(H, W)
        # H *= k
        # W *= k
        # H = int(round(H / 64.0)) * 64
        # W = int(round(W / 64.0)) * 64
        # qrcode_image = input_image.resize((W, H), resample=Image.LANCZOS)

        # model_dir_con = snapshot_download(Controlnet_MAP[controlnet_path],revision='v1.0.0')
        # controlnet = ControlNetModel.from_pretrained(
        # model_dir_con,
        # torch_dtype=torch.float32
        # ).to('cpu')
        # base_model = snapshot_download(Model_path_MAP[model],revision='v1.0.0')
        # pipe = StableDiffusionControlNetPipeline.from_pretrained(
        #     base_model,
        #     controlnet=controlnet,
        #     #vae=vae,
        #     # safety_checker=None,
        #     torch_dtype=torch.float32
        # ).to('cuda')
        # if lora_path != 'None':
        #     # print("Use lora!")
        #     ### add lora
        #     pipe.unload_lora_weights()
        #     pipe.load_lora_weights(Lora_MAP[lora_path])
        #     #lora_w = 0.8
        #     pipe._lora_scale = lora_w
        #     state_dict, network_alphas = pipe.lora_state_dict(
        #         Lora_MAP[lora_path]
        #     )

        #     #network_alpha = network_alpha * lora_w
        #     pipe.load_lora_into_unet(
        #         state_dict = state_dict
        #         , network_alphas = network_alphas
        #         , unet = pipe.unet
        #     )

        #     pipe.load_lora_into_text_encoder(
        #         state_dict = state_dict
        #         , network_alphas = network_alphas
        #         , text_encoder = pipe.text_encoder
        #     )
        #     compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        # pipe.scheduler = SAMPLER_MAP[sampler](pipe.scheduler.config)
        # pipe.enable_xformers_memory_efficient_attention()

        # out = pipe(
        #     #prompt_embeds=
        #     # prompt=prompt,
        #     prompt_embeds=compel_proc(prompt),
        #     #negative_prompt_embeds=
        #     # negative_prompt=negative_prompt,
        #     negative_prompt_embeds=compel_proc(negative_prompt),
            
        #     image=qrcode_image,
        #     width=768,
        #     height=768,
        #     guidance_scale=float(guidance_scale),
        #     controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        #     num_inference_steps=int(num_inference_steps),
            
        # )
        # return out.images[0]