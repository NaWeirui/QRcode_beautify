import ssl
import nltk
import shutil
import os
import zipfile
import requests
import subprocess
from urllib.parse import urlparse, unquote


# download nltk package with ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def download_and_setup():
    # 获取URL
    url = os.getenv('AGENT_URL')
    if not url:
        raise ValueError("AGENT_URL environment variable is not set.")
    
    # 从URL中解析出文件名
    filename = unquote(urlparse(url).path.split('/')[-1]).split('/')[-1]
    file_path = os.path.abspath(filename)

    # 检查文件是否已存在，如果存在则删除
    if os.path.exists(file_path):
        os.remove(file_path)

    # 删除原有配置信息,确保重新加载
    if os.path.exists("/tmp/agentfabric/config/local_user"):
        shutil.rmtree("/tmp/agentfabric/config/local_user")

    # 下载文件
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
    else:
        raise Exception(f"Failed to download file, status code: {response.status_code}")

    # 解压文件
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(file_path))

    # 设置PYTHONPATH环境变量
    pythonpath = os.path.dirname(file_path)
    os.environ['PYTHONPATH'] = pythonpath

    # 安装requirements.txt中的依赖
    requirements_path = os.path.join(pythonpath, 'requirements.txt')
    if os.path.exists(requirements_path):
        subprocess.run(['pip', 'install', '-r', requirements_path], check=True)

    # 运行appBot.py
    app_bot_path = os.path.join(pythonpath, 'appBot.py')
    if os.path.exists(app_bot_path):
        subprocess.run(['python', app_bot_path], check=True)
    else:
        raise Exception(f"appBot.py does not exist in {pythonpath}")

# 使用示例
download_and_setup()

