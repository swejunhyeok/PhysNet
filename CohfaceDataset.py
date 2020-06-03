import torch
import os
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset

class CohfaceDataset(Dataset):
    def __init__(self, A_path, M_path, R_path):
        self.A = np.load(A_path)
        self.M = np.load(M_path)
        self.R = np.load(R_path)
        #self.__scale()
        print(self.A.shape, self.M.shape, self.R.shape)

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        A = self.A[idx]
        A = torch.FloatTensor(A)
        A = A.permute(2, 0, 1)
        A = A/255.  # convert image to [0, 1]
        A = torch.sub(A, torch.mean(A, (1, 2)).view(3, 1, 1))

        M = self.M[idx]
        M = torch.FloatTensor(M)
        M = M.permute(2, 0, 1)

        R = []
        R.append(self.R[idx])
        R = np.asarray(R)
        R = torch.FloatTensor(R)
        return A, M, R

    def __scale(self):
        part = 0
        window = 32

        while part < (len(self.R) // window) - 1:
            self.R[part*window:(part+1)*window] /= np.std(self.R[part*window:(part+1)*window])
            part += 1

        if len(self.R) % window != 0:
            self.R[part * window:] /= np.std(self.R[part * window:])




            




