"""
Code for Toward Adversarial Robustness via Semi-supervised Robust Training
"""

import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from advertorch.utils import predict_from_logits
from models.mnist.small_cnn import *

parser = argparse.ArgumentParser(description='MNIST Robustness Evaluation')
parser.add_argument('--model-path', default='./checkpoints/MNIST/SRT/checkpoint.pth.tar', help='trained model path')
parser.add_argument('--test_batch', type=int, default=500, metavar='N', help='input batch size for testing (default: 200)')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.manual_seed(666)

model = SmallCNN()
model = torch.nn.DataParallel(model).cuda()


print('==> Load the target model..')
assert os.path.isfile(args.model_path), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=args.test_batch, shuffle=False,
        pin_memory=True
)

print('==> This is the PGD')


from advertorch.attacks import LinfPGDAttack
adversary = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                              nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

correct_clean = 0
correct_adv = 0

for idx, (cln_data, true_label) in enumerate(loader):

    cln_data, true_label = cln_data.to(device), true_label.to(device)

    adv_untargeted = adversary.perturb(cln_data, true_label)

    pred_cln = predict_from_logits(model(cln_data))
    pred_adv = predict_from_logits(model(adv_untargeted))

    correct_clean = correct_clean + (pred_cln.data == true_label.data).float().sum()
    correct_adv = correct_adv + (pred_adv.data == true_label.data).float().sum()

    print("current correct clean samples: %s; current correct adv samples: %s" %(correct_clean.data.item(), correct_adv.data.item()))

print("correct clean samples: ", correct_clean)
print("correct adversarial samples: ", correct_adv)



