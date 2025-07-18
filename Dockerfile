FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# ติดตั้ง libGL ที่จำเป็นสำหรับ opencv/ultralytics
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# ติดตั้ง torch/torchvision แบบ CPU-only
RUN pip install --no-cache-dir torch==2.2.2+cpu torchvision==0.17.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "code/api.py"]
