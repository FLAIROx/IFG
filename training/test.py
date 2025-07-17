import torch
import logging

logging.basicConfig(level=logging.INFO)

logging.info('Started')
logging.info(f'{torch.cuda.device_count()=}')
DEVICE = 'cuda:0'

model = torch.nn.Linear(4,4).to(DEVICE)
input = torch.eye(4, 4).to(DEVICE)

output = model(input).sum()
logging.info(f'{output=}, computed on single gpu')

output.backward()
torch.cuda.synchronize()

logging.info('Backward pass completed')

model = torch.nn.DataParallel(model)

output = model(input).sum()
logging.info(f'{output=}, computed on all gpus')

output.backward()

torch.cuda.synchronize()

logging.info('Distributed Backward pass completed')
