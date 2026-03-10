import os
import sys
import subprocess

def has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def run_training(model, use_cuda, epochs=3):
    cmd = [sys.executable, os.path.join('..','training','train.py'), '--data-path', os.path.join('..','training','labeled_dataset.csv'), '--model', model, '--epochs', str(epochs), '--batch-size', '64', '--save-dir', os.path.join('..','training','models')]
    if use_cuda:
        cmd.append('--use-cuda')
    print('Running:', ' '.join(cmd))
    ret = subprocess.run(cmd)
    return ret.returncode

if __name__ == '__main__':
    cuda = has_cuda()
    print('CUDA available:', cuda)
    models = ['rnn','lstm','attn']
    for m in models:
        rc = run_training(m, cuda, epochs=3)
        if rc != 0:
            print(f"Training {m} failed with return code {rc}")
            # continue to next model
    print('All done')
