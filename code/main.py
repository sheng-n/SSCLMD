import torch
import numpy as np
from parms_setting import settings
from data_preprocess import load_data
from train import train_model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parameters setting
args = settings()

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


data_s, data_f, train_loader, test_loader = load_data(args, n_splits=5)

for fold, (train_loader, test_loader) in enumerate(zip(train_loader, test_loader)):
    print(f"Training on fold {fold+1}")
    train_model(data_s, data_f, train_loader, test_loader, args)


# load data
# data_s, data_f,train_loader, test_loader = load_data(args)

# train and test model


