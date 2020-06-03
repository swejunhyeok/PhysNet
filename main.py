import argparse
import os

import torch
import time
from network import DeepPhys
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from torch.utils.data import DataLoader
from CohfaceDataset import CohfaceDataset

tr = torch

model_options = ['difference', 'appear', 'fusion']
dataset_options = ['HR', 'BR']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='BR',
                    choices=dataset_options)
parser.add_argument('--model_type', '-a', default='fusion',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--lr', type=float, default=1,
                    help='learning rate')
parser.add_argument('--rho', type=float, default=0.95,
                    help='momentum')
parser.add_argument('--eps', type=float, default=1e-8,
                     help='learning decay for lr scheduler')

def main():
    global args
    args = parser.parse_args()

    if args.dataset == 'HR':
        train_dataset = CohfaceDataset("train_A.npy", "train_M.npy", "train_HR.npy")
        test_dataset = CohfaceDataset("test_A.npy", "test_M.npy", "test_HR.npy")
    elif args.dataset == 'BR':
        train_dataset = CohfaceDataset("train_A.npy", "train_M.npy", "train_BR.npy")
        test_dataset = CohfaceDataset("test_A.npy", "test_M.npy", "test_BR.npy")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # np.random.seed(42)
    # torch.backends.cudnn.deterministic = True
    # torch.manual_seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(42)

    model = DeepPhys()

    model = model.cuda()

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    
    optimizer = optim.Adadelta(filtered_parameters, lr=args.lr, rho=args.rho, eps=args.eps)
    #optimizer = optim.AdamW(filtered_parameters, lr=1e-4)

    criterion = nn.L1Loss().cuda()

    best_loss = 100000
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader)
        model.train()
        train_loss = train(progress_bar, model, criterion, optimizer, epoch)
        model.eval()
        test_loss = test(test_loader, model, criterion, epoch)
        tqdm.write('train_loss : {0:.3f} / test_loss : {1:.3f}'.format(train_loss, test_loss))

        if test_loss < best_loss:
            best_loss = test_loss
            checkpoint_name = args.dataset + '_0527'
            save_checkpoint({
                'epoch': epoch,
                'arch': args.model_type,
                'dataset' : args.dataset,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, checkpoint_name)
            tqdm.write('save_checkpoint')

def train(progress_bar, model, criterion, optimizer, epoch):
    total_loss = 0.0
    for i, (A, M, target) in enumerate(progress_bar):
        A = A.cuda()
        M = M.cuda()
        target = target.cuda()

        output = model(A, M)

        loss = criterion(output, target)

        total_loss += loss.item()

        progress_bar.set_description('Epoch : {0} / loss : {1:.3f}'.format(epoch, loss.item()))
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
    
    total_loss /= i
    return total_loss

def test(test_loader, model, criterion, epoch):
    total_loss = 0.0
    if epoch == args.epochs - 1:
        result_file = open("final_result_0527.txt", 'w')
    for i, (A, M, target) in enumerate(test_loader):
        A = A.cuda()
        M = M.cuda()
        target = target.cuda()

        with torch.no_grad():
            output = model(A, M)

        if epoch == args.epochs - 1:
            for j in range(output.shape[0]):
                result_file.write(str(output[j][0].item()) + '\n')

        loss = criterion(output, target)

        total_loss += loss.item()

    if epoch == args.epochs - 1:
        result_file.close()
    total_loss /= i
    return total_loss

def save_checkpoint(state, test_id):
    if not(os.path.isdir('check_point')):
        os.makedirs(os.path.join('check_point'))
    filename = 'check_point/' + test_id +'.pth.tar'
    torch.save(state, filename)

class SNRLoss(nn.Module):
    def __init__(self):
        super(SNRLoss, self).__init__()

    def forward(self, outputs: tr.Tensor, targets: tr.Tensor, Fs=20):
        device = outputs.device
        if not outputs.is_cuda:
            torch.backends.mkl.is_available()

        N = outputs.shape[-1]
        pulse_band = tr.tensor([40/60., 250/60.], dtype=tr.float32).to(device)
        f = tr.linspace(0, Fs/2, int(N/2)+1, dtype=tr.float32).to(device)

        min_idx = tr.argmin(tr.abs(f - pulse_band[0]))
        max_idx = tr.argmin(tr.abs(f - pulse_band[1]))

        outputs = outputs.view(-1, N)
        targets = targets.view(-1, 1)

        X = tr.rfft(outputs, 1, normalized=True)
        P1 = tr.add(X[:, :, 0]**2, X[:, :, 1]**2)                                   # One sided Power spectral density

        # calculate indices corresponding to refs
        ref_idxs = []
        for ref in targets:
            ref_idxs.append(tr.argmin(tr.abs(f-ref)))

        # calc SNR for each batch
        losses = tr.empty((len(ref_idxs),), dtype=tr.float32)
        freq_num_in_pulse_range = max_idx-min_idx
        for count, ref_idx in enumerate(ref_idxs):
            pulse_freq_amp = P1[count, ref_idx]
            other_avrg = (tr.sum(P1[count, min_idx:ref_idx-1]) + tr.sum(P1[count, ref_idx+2:max_idx]))/(freq_num_in_pulse_range-3)
            losses[count] = -10*tr.log10(pulse_freq_amp/other_avrg)

        return tr.mean(losses)

class GaussLoss(nn.Module):
    """
    Loss for normal distribution (L2 like loss function)
    """

    def __init__(self):
        super().__init__()

    def forward(self, outputs: tr.Tensor, targets: tr.Tensor) -> tr.Tensor:
        """
        :param outputs: tensor of shape: (batch_num, samples_num, density_parameters), density_parameters=2 -> mu, sigma
        :param targets: tensor of shape: (batch_num, samples_num)
        :return: loss (scalar)
        """

        n_samples = targets.shape[-1]
        outputs = outputs.view(-1, n_samples, 2)
        targets = targets.view(-1, n_samples)

        mus = outputs[:, :, 0]
        sigmas = outputs[:, :, 1]
        s = tr.log(sigmas**2)

        losses = tr.exp(-1*s)*(targets-mus)**2 + s

        return losses.mean()

if __name__ == '__main__':
    main()

