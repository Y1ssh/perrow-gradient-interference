#!/bin/bash
set -e  # stop on first error

echo "============================================"
echo "  H100 Setup for Per-Row Interference Research"
echo "============================================"

# 1. System check
echo ""
echo "--- GPU Check ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'BF16: {torch.cuda.is_bf16_supported()}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memoryory/1e9:.1f} GB')
"

# 2. Install dependencies
echo ""
echo "--- Installing packages ---"
pip install tiktoken datasets
pip install git+https://github.com/KellerJordan/Muon

# 3. Check Muon installed correctly
python3 -c "
from muon import SingleDeviceMuonWithAuxAdam
print('Muon: SingleDeviceMuonWithAuxAdam ✓')
"

# 4. Also check if torch.optim.Muon exists (PyTorch 2.9+)
python3 -c "
try:
    from torch.optim import Muon
    print('torch.optim.Muon: available ✓')
except ImportError:
    print('torch.optim.Muon: not available (using Keller package instead)')
"

# 5. Clone our repo
echo ""
echo "--- Cloning repo ---"
if [ ! -d "perrow-gradient-interference" ]; then
    git clone https://github.com/Y1ssh/perrow-gradient-interference.git
fi
cd perrow-gradient-interference

# 6. Create results directories
mkdir -p results/phase_a results/phase_b results/phase_c results/phase_d results/phase_e results/phase_f

# 7. Configure git for pushing results
git config user.email "yash@research.local"
git config user.name "Yash Madelwar"

# 8. Download and tokenize FineWeb-Edu (cache to disk)
echo ""
echo "--- Preparing FineWeb-Edu data ---"
python3 -c "
import os, time
import tiktoken
from datasets import load_dataset
import torch

cache = 'fineweb_train_50M.pt'
if os.path.exists(cache):
    print(f'  Data already cached: {cache}')
else:
    print('  Downloading FineWeb-Edu and tokenizing (50M tokens)...')
    t0 = time.time()
    enc = tiktoken.get_encoding('gpt2')
    ds = load_dataset('HuggingFaceFW/fineweb-edu', 'sample-10BT',
                       split='train', streaming=True)
    tokens = []
    for ex in ds:
        tokens.extend(enc.encode_ordinary(ex['text']))
        if len(tokens) >= 50_000_000:
            break
        if len(tokens) % 5_000_000 < 1000:
            print(f'    {len(tokens)/1e6:.0f}M tokens...')
    tokens = tokens[:50_000_000]
    torch.save(torch.tensor(tokens, dtype=torch.long), cache)
    print(f'  Cached {len(tokens):,} tokens in {time.time()-t0:.0f}s')
"

# 9. Quick smoke test (10 steps, verify model + optimizer + data work)
echo ""
echo "--- Smoke test (10 steps) ---"
python3 -c "
import torch, sys, os
sys.path.insert(0, '.')
from model.gpt2 import GPT2
from muon import SingleDeviceMuonWithAuxAdam
import torch.nn.functional as F

model = GPT2().cuda()
opt = SingleDeviceMuonWithAuxAdam(model.get_muon_param_groups())
data = torch.load('fineweb_train_50M.pt', weights_only=True)
data = data[:data.size(0)//1024*1024].view(-1, 1024).cuda()

V = model.config.vocab_size
for step in range(1, 11):
    idx = torch.randint(0, data.size(0), (4,))
    batch = data[idx]
    opt.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, _ = model(batch)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), batch[:, 1:].reshape(-1))
    loss.backward()
    opt.step()
    if step % 5 == 0:
        print(f'  Step {step}: loss={loss.item():.4f}')

del model, opt
torch.cuda.empty_cache()
print('Smoke test PASSED ✓')
"

echo ""
echo "============================================"
echo "  Setup complete. Ready to run experiments."
echo "  cd perrow-gradient-interference"
echo "  python3 experiments/phase_a_measurements.py"
echo "============================================"
