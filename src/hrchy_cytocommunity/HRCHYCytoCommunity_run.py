import argparse
import datetime
from HRCHYCytoCommunity import HRCHYCytoCommunityGrand,HRCHYCytoCommunity
from models.dataset import SpatialOmicsImageDataset
from visualization.visualization import load_base_data
import numpy as np
import os
import torch
import pandas as pd

def float_or_none(value):
    """将字符串转换为 float 或 None"""
    if value.lower() == 'none':
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' 不是有效的浮点数或 'none'")
def main():
    parser = argparse.ArgumentParser(description='HRCHYCytoCommunity')
    parser.add_argument('--data-input-dir', default='./', help='dataset setting')
    parser.add_argument('--seed', type=int, default=2025, metavar='S', help='random seed (default: 2025)')
    parser.add_argument('--mode', type=str, default='base',choices=['base', 'full'],help='mode of HRCHYCytoCommunity,base or full')
    parser.add_argument('--num-epoch', type=int, default=1500)
    parser.add_argument('--num-tcn1', type=int,help='number of fine-grained CNs')
    parser.add_argument('--num-tcn2', type=int,help='number of coarse-grained TCs')    
    parser.add_argument('--num-run', type=int, default=10, help='times of running whole model')
    parser.add_argument('--alpha', type=float, default=0.9,help='loss weight of coarse and fine level clustering')
    parser.add_argument('--lambda1', type=float, default=1,help='Coefficient of consistency regularization, only work in Grand mode')
    parser.add_argument('--lambda2', type=float, default=1,help='Coefficient of orthogonality loss')
    parser.add_argument('--lambda-balance', type=float, default=1,help='Coefficient of entropy loss')
    parser.add_argument('--edge-pruning-cutoff', type=float_or_none, default=None,help='edge pruning cut off')
    parser.add_argument('--lr', type=float, default=1e-4,help='learning rate')
    parser.add_argument('--drop-rate', type=float, default=0.5,help='the rate of drop node, only work in Grand mode')
    parser.add_argument('--temp', type=float, default=1,help='temperature of sharpening, only work in Grand mode')
    parser.add_argument('--num-hidden', type=int,default=128,help='dimension of hiddien layer')
    parser.add_argument('--s', type=int,default=5,help='number of running time in Grand, only work in Grand mode')        
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='./results/')
    parser.add_argument('--vis-dir', type=str, default='./figures/')
    parser.add_argument('--gt-fine',default='False',choices=['True', 'False'],help="whether Ground Truth of fine-level exist")
    parser.add_argument('--gt-coarse', default='False',choices=['True', 'False'],help="whether Ground Truth of coarse-level exist")

    args = parser.parse_args()
    args.gt_fine = (args.gt_fine == 'True')
    args.gt_coarse = (args.gt_coarse == 'True')
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == 'base':
        model_params = {
            'mode' : args.mode,       # HRCHYCytoCommunity with Grand
            'num_run' : args.num_run,
            'num_epoch' : args.num_epoch,
            'num_tcn1': args.num_tcn1,
            'num_tcn2' : args.num_tcn2,
            'alpha' : args.alpha,          # loss weight of corase and fine level loss
            'lambda1':args.lambda1,           # Coefficient of consistency regularization
            'lambda2':args.lambda2,           # Coefficient of orthogonality loss
            'lambda_balance':args.lambda_balance,
            'edge_pruning_cutoff' : args.edge_pruning_cutoff,       # edge pruning cutoff
            'num_hidden' : args.num_hidden,     # the dimension of hidden layer
            'lr' : args.lr,
            'gt_fine':args.gt_fine,
            'gt_coarse':args.gt_coarse,
            'device':args.device
        }  
    elif args.mode == 'Grand':
        model_params = {
            'mode' : args.mode,       # HRCHYCytoCommunity with Grand
            's' : args.s,
            'num_run' : args.num_run,
            'num_epoch' : args.num_epoch,
            'num_tcn1': args.num_tcn1,
            'num_tcn2' : args.num_tcn2,
            'alpha' : args.alpha,          # loss weight of corase and fine level loss
            'lambda1':args.lambda1,           # Coefficient of consistency regularization
            'lambda2':args.lambda2,           # Coefficient of orthogonality loss
            'lambda_balance':args.lambda_balance,
            'edge_pruning_cutoff' : args.edge_pruning_cutoff,       # edge pruning cutoff
            'num_hidden' : args.num_hidden,     # the dimension of hidden layer
            'lr' : args.lr,
            'drop_rate' : args.drop_rate,
            'temp': args.temp,
            'gt_fine':args.gt_fine,
            'gt_coarse':args.gt_coarse,
            'device':args.device
        } 
    HyperPara_df = pd.DataFrame(model_params.items(), columns=['Parameter', 'Value'])
    # Seed the run and create saving directory
    dataset = SpatialOmicsImageDataset(args.data_input_dir)
    graph_dict = {
        # fill graph of this dataset
    }
    print(os.path.join(args.save_dir))
    if not os.path.exists(os.path.join(args.save_dir)):
        os.makedirs(os.path.join(args.save_dir))
    HyperPara_df.to_csv(os.path.join(args.save_dir,'HyperPara.csv'))
    for i in range(args.num_run):
        print(f"---------------Run{i}----------------")
        for slice_name,graph_idx in graph_dict.items():
            print(f"{slice_name} is processing")
            train_dataset = dataset[graph_idx]
            cell_meta = load_base_data(os.path.join(args.data_input_dir,'raw'),
                                       graph_idx,fine_GT=model_params['gt_fine'],
                                       coarse_GT=model_params['gt_coarse'])
            args.cell_meta = cell_meta
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            if args.mode == 'base':
                hrchycytocommunity = HRCHYCytoCommunity(
                    dataset = train_dataset,
                    num_tcn1 = args.num_tcn1,
                    num_tcn2 = args.num_tcn2,
                    cell_meta = args.cell_meta,
                    lr = args.lr,
                    alpha = args.alpha,
                    num_epoch = args.num_epoch,
                    lambda1 = args.lambda1,
                    lambda2 = args.lambda2,
                    lambda_balance = args.lambda_balance,
                    edge_pruning_cutoff = args.edge_pruning_cutoff,
                    device = args.device,
                    num_hidden = args.num_hidden,
                    gt_coarse=args.gt_coarse,
                    gt_fine=args.gt_fine,
                )
            elif args.mode == 'Grand':
                hrchycytocommunity = HRCHYCytoCommunityGrand(
                    dataset = train_dataset,
                    num_tcn1 = args.num_tcn1,
                    num_tcn2 = args.num_tcn2,
                    cell_meta = args.cell_meta,
                    lr = args.lr,
                    alpha = args.alpha,
                    num_epoch = args.num_epoch,
                    lambda1 = args.lambda1,
                    lambda2 = args.lambda2,
                    lambda_balance = args.lambda_balance,
                    edge_pruning_cutoff = args.edge_pruning_cutoff,
                    temp = args.temp,
                    s = args.s,
                    drop_rate = args.drop_rate,
                    device = args.device,
                    num_hidden = args.num_hidden,
                    gt_coarse=args.gt_coarse,
                    gt_fine=args.gt_fine,
                )
            ret_output_dir = os.path.join(args.save_dir,f"Run{i}",slice_name)
            hrchycytocommunity.train(save_dir=ret_output_dir,output=False)
            hrchycytocommunity.predict(save = True,save_dir=ret_output_dir)
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    

if __name__ == '__main__':
    # import sys
    # home_dir = '/home/runzhixie/xrz/HRCHY-CytoCommunity_project/'
    # dataset = '18_Science_Mouse_Hypo_MERFISH'
    # # method = 'perplexity_down_sym'
    # method = 'perplexity_down_nosym'
    # # method = 'perplexity_version'
    # setting = 'KNN_perplexity_40'
    # data_input_dir = f"{home_dir}/data/processed/{dataset}/run_compared_method/{method}/{setting}"
    # ret_output_dir0 = f"{home_dir}/results/files/{dataset}/original_result/run_compared_method/{method}/{setting}"
    # mode = 'Grand'
    # for lr in [1e-4]:
    #     sys.argv = [
    #         'HRCHYCytoCommunity_run.py',
    #         '--mode',mode,
    #         '--data-input-dir', data_input_dir,
    #         '--s','5',
    #         '--num-tcn1', '12',
    #         '--num-tcn2','3',
    #         '--lr',"0.0001",
    #         '--lambda-balance','1',
    #         '--num-epoch','1500',
    #         '--edge-pruning-cutoff','0.1',
    #         '--save-dir',f"{ret_output_dir0}",
    #         # '--gt-coarse','False',
    #         '--num-run','5',
    #         '--gt-fine','True',
    #         '--device','cuda:2',
    #     ]
        main()
    