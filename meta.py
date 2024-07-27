import torch
import transformers
import os 
from dotenv import load_dotenv

load_dotenv()

hugguinface_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
print(hugguinface_token)


model_id = 'meta-llama/Meta-Llama-3.1-8B'

pipeline = transformers.pipeline(
    'text-generation',
    model=model_id,
    model_kwargs={'torch_dtype': torch.bfloat16},
    device_map='auto',
    
    
    
)

result = pipeline('hello, how are you?', max_new_tokens=50)
print(result)





