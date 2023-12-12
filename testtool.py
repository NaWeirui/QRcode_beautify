from modelscope_agent.tools.tool import Tool, ToolSchema
from modelscope_agent.tools.qrcode_beautify_tool import QRcodeTool
import gradio as gr
from PIL import Image

params = {
    "prompt":"small house,white flowers on the roof,high quality",
    "image":"/home/wsco/nwr/qrcode_controlnet_realistic/qrcode_image/image1.png"
}
result = QRcodeTool().__call__(**params)
result.save('result.png')

