import sys
import os

import argparse
import importlib
from distutils.util import strtobool

from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch_geometric.data import DataLoader

from src.data.Transforms import MedianSampling2d, MaxSampling1d, MedianSampling1d
from src.models import GNNLSTM, ConvSeqAttentionModel, MTGNNModel, RecurrentModel
from src.utils.device import get_device
from src.utils import Trainer
from src.utils.evaluate import evaluate_performance
from src.visualization.graph_plot import plot_embedding, plot_adjacency
from src.visualization.loss_plot import get_loss_plot
# from src.visualization.error_distribution import get_error_distribution_plot

def main(args):

    print()
    # check python and torch versions
    print(f'Python v.{sys.version.split()[0]}')
    print(f'PyTorch v.{torch.__version__}')

    # get device
    device = args.device
    print(f'Device status: {device}')
    
    dataset = args.dataset
    # dataset import
    p, m = 'src.datasets.' + dataset, dataset.capitalize()
    mod = importlib.import_module(p)
    dataset_class = getattr(mod, m)

    # dataset transforms
    transform_dict = {'median': MedianSampling2d}
    target_transform_dict = {'median': MedianSampling1d, 'max': MaxSampling1d}

    transform = transform_dict.get(args.transform, None)
    if transform is not None:
        transform = transform(10)

    target_transform = target_transform_dict.get(args.target_transform, None)
    if target_transform is not None:
        target_transform = target_transform(10)

    # training / test data set definitions
    lags = args.window_size
    stride = args.stride
    horizon = args.horizon
    train_ds = dataset_class(lags, stride=stride, horizon=horizon, train=True, transform=transform, normalize=args.normalize, device=device)
    test_ds = dataset_class(lags, stride=stride, horizon=horizon, train=False, transform=transform, target_transform=target_transform, normalize=args.normalize, device=device)

    # get train and validation data split at random index 
    val_split = args.val_split
    val_len = int(len(train_ds)*val_split)
    split_idx = np.random.randint(0, len(train_ds) - val_len) # exclude beginning of dataset for stability
    a, b = split_idx, split_idx+val_len # split interval

    train_parition = train_ds[:a] + train_ds[b:] 
    val_parition = train_ds[a:b]

    # data loaders
    batch_size = args.batch_size
    # train, val, test partitions
    train_loader = DataLoader(train_parition[int(len(train_ds)*0.0):], batch_size=batch_size, shuffle=args.shuffle_train)
    if len(val_parition) > 0:
        val_loader = DataLoader(val_parition, batch_size=batch_size, shuffle=True)
        thresholding_loader = DataLoader(val_parition, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
        thresholding_loader = DataLoader(train_ds[int(len(train_ds)*0.0):], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
    # node meta data
    num_nodes = train_ds.num_nodes
    args.num_nodes = num_nodes
    try:
        node_names = train_ds.node_names
    except:
        node_names = str(list(range(num_nodes)))

    print(f'\nDataset <{train_ds.name.capitalize()}> loaded...')
    print(f'   Number of nodes: {num_nodes}')
    print(f'   Training samples: {len(train_parition)}')
    if val_loader:
        print(f'   Validation samples: {len(val_parition)}')
    print(f'   Test samples: {len(test_ds)}\n')

    ### MODEL
    # model = ConvSeqAttentionModel(args).to(device)
    # model = RecurrentModel(args).to(device)
    # model = MTGNNModel(args).to(device)
    model = GNNLSTM(args).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    criterion = nn.MSELoss(reduction='mean')

    # torch.autograd.set_detect_anomaly(True) # uncomment for debugging

    print('Training...')
    trainer = Trainer(model, optimizer, criterion)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # log directory
    logdir = os.path.join('runs/', stamp + f' - {args.dataset}')
    os.makedirs(logdir, exist_ok=True)

    if args.log_graph:
        # save randomly initialised graph for plotting
        init_edge_index, init_embedding = model.get_embedding()
        init_graph = model.get_graph()

    # TRAINING ###
    train_loss_history, val_loss_history, best_model_state = trainer.train(
        train_loader, 
        val_loader, 
        epochs=args.epochs, 
        early_stopping=args.early_stopping, 
        return_model_state=True,
    )
    ### TESTING ###
    print('Testing...')
    # best model parameters
    model.load_state_dict(best_model_state)

    thresholding_results, final_train_loss = trainer.test(thresholding_loader)
    test_ds_results, test_loss = trainer.test(test_loader)
    print(f'   Tresholding Data MSE: {final_train_loss:.6f}')
    print(f'   Test MSE: {test_loss:.4f}\n')

    with open(os.path.join(logdir, 'loss.txt'), 'w') as f:
            f.write(f'Tresholding MSE: {final_train_loss:.6f}\nTest MSE: {test_loss:.6f}\n')

    print('Evaluating Performance...')

    results = evaluate_performance(
        thresholding_results, 
        test_ds_results, 
        threshold_method=args.thresholding, 
        smoothing=args.smoothing, 
        smoothing_method=args.smoothing_method,
    )
    result_str = f'   {str(results["method"]).capitalize()} thresholding:\n \n' + \
        f'             |     Normal     |    Adjusted    |\n' + \
        f'   ----------|----------------|----------------|\n' + \
        f'   Precision | {results["prec"]:>13.3f}  | {results["a_prec"]:>13.3f}  |\n' + \
        f'   Recall    | {results["rec"]:>13.3f}  | {results["a_rec"]:>13.3f}  |\n' + \
        f'   F1 / F2   | {results["f1"]:>6.3f} / {results["f2"]:>5.3f} | {results["a_f1"]:>6.3f} / {results["a_f2"]:>5.3f} |\n' + \
        f'   ----------|----------------|----------------|----------------\n' + \
        f'                                               | Latency: {results["latency"]:.2f}\n'

    print(result_str)
    with open(os.path.join(logdir, f'results_{str(results["method"])}.txt'), 'w') as f:
        f.write(result_str)
    
    ### Uncomment for exhaustive threshold / smoothing parameter search
    # precision, recall, f1, f2 = -1, -1, -1, -1
    # best_method = None
    # j = 0
    # for i in range(1, 25+1):
    #     results = evaluate_performance(
    #         thresholding_results, 
    #         test_ds_results, 
    #         threshold_method='best', 
    #         smoothing=i, 
    #         smoothing_method=args.smoothing_method,
    #     )
    #     if 1 >= results["f1"] > f1 :
    #         precision, recall, f1, f2 = results["prec"], results["rec"], results["f1"], results["f2"]
    #         best_method = results["method"]
    #         j = i
    # print(f'  Best method: {best_method}')
    # print(f'  Best smoothing parameter: {j}')
    # print(f'   Precision: {precision:.4f}')
    # print(f'   Recall: {recall:.4f}')
    # print(f'   F1 | F2 scores: {f1:.4f} | {f2:.4f}\n')

    ### RESULTS PLOTS ###
    print('Logging Results...')
    
    with open(os.path.join(logdir, 'model.txt'), 'w') as f:
        f.write(str(model))

    # learned graph
    if args.log_graph:
        learned_edges, learned_embedding = model.get_embedding()
        learned_graph = model.get_graph()
        for i in range(len(learned_embedding)):
            plot_embedding(init_edge_index, init_embedding[i], node_names, os.path.join(logdir, f'init_emb_{i}.html'))
            plot_embedding(learned_edges, learned_embedding[i], node_names, os.path.join(logdir, f'trained_emd_{i}.html'))

        plot_adjacency(init_graph, node_names, os.path.join(logdir, f'init_A.html'))
        plot_adjacency(learned_graph, node_names, os.path.join(logdir, f'learned_A.html'))
    
    # loss 
    fig = get_loss_plot(train_loss_history, val_loss_history)
    fig.savefig(os.path.join(logdir, 'loss_plot.png'))

    # # error distributions
    # results_dict = {'Validation': thresholding_results, 'Testing': test_ds_results}
    # fig = get_error_distribution_plot(results_dict)
    # fig.savefig(os.path.join(logdir, 'error_distribution.png'))

    ### SAVE MODEL ###
    torch.save(best_model_state, os.path.join(logdir, 'model.pt'))
    
    print() # script end

if __name__ == '__main__':

    device = get_device()

    parser = argparse.ArgumentParser()

    ### -- Data params --- ###
    parser.add_argument("-dataset", type=str.lower, default="swat")
    parser.add_argument("-window_size", type=int, default=30)
    parser.add_argument("-stride", type=int, default=1)
    parser.add_argument("-horizon", type=int, default=10)
    parser.add_argument("-val_split", type=float, default=0.2)
    parser.add_argument("-transform", type=str, default='median')
    parser.add_argument("-target_transform", type=str, default='median')
    parser.add_argument("-normalize", type=lambda x:strtobool(x), default=False)
    parser.add_argument("-shuffle_train", type=lambda x:strtobool(x), default=True)
    parser.add_argument("-batch_size", type=int, default=64)

    ### -- Model params --- ###
    # Sensor embedding
    parser.add_argument("-embed_dim", type=int, default=16)
    parser.add_argument("-topk", type=int, default=5)
    
    ### --- Thresholding params --- ###
    parser.add_argument("-smoothing", type=int, default=1)
    parser.add_argument("-smoothing_method", type=str, default='exp') # exp or mean
    parser.add_argument("-thresholding", type=str, default='max') # max or mean

    ### --- Training params --- ###
    parser.add_argument("-epochs", type=int, default=50)
    parser.add_argument("-early_stopping", type=int, default=20)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-betas", nargs=2, type=float, default=(0.9, 0.999))
    parser.add_argument("-weight_decay", type=float, default=0)
    parser.add_argument("-device", type=torch.device, default=device) # cpu or cuda

    ### --- Logging params --- ###
    parser.add_argument("-log_tensorboard", type=lambda x:strtobool(x), default=False)
    parser.add_argument("-log_graph", type=lambda x:strtobool(x), default=True)

    args = parser.parse_args()

    main(args)
