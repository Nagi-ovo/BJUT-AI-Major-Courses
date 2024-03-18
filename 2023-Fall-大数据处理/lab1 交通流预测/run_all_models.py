import argparse
import torch
import torch.nn as nn
import numpy as np
import os

# 导入模型
from model.HA import historical_average as HA
from model.LR import linear_regression as LR
from model.GRU import GRU
from model.LSTM import LongShortTimeMemory as LSTM
from model.STGCN import STGCN
from model.GWN import gwnet as GWN
from model.ARIMA import arima as ARIMA

from lib.utils import load_data,get_adjacency_matrix,scaled_Laplacian,generate_dataset
from lib.utils import save_metrics2csv, save_data2numpy
from lib.metrics import masked_mae_test, masked_rmse_test


# 定义训练过程
def train_epoch(train_dataloader):
    epoch_training_losses = []
    
    for i, (x_batch, y_batch) in enumerate(train_dataloader):
        # x.shape -> b,n,t
        net.train()
        optimizer.zero_grad()
        x_batch = x_batch.to(torch.float32)  
        y_batch = y_batch.to(torch.float32)                        
        x_batch = x_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)
        out = net(x_batch, A_wave)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)


parser = argparse.ArgumentParser()
# 增加指定的参数

parser.add_argument('--model', type=str,
                    default='LR', help='the name of model')
parser.add_argument('--dataset', type=str, default='METR_LA', 
                    help='the name of dataset')
parser.add_argument('--batchsize', type=int, default=24)
parser.add_argument('--input_timesteps', type=int, default=12)  
parser.add_argument('--output_timesteps', type=int, default=12)  
parser.add_argument('--epochs', type=int, default=200) # 60
parser.add_argument('--lr', type=float, default=0.00001)  # 0.0005
parser.add_argument('--train_per', type=float, default=0.5)  # 0.6
parser.add_argument('--val_per', type=float, default=0.2)
parser.add_argument('--test_per', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
args = parser.parse_args()

# 创建随机数，程序中所有随机过程在每次运行都保持一致
torch.manual_seed(args.seed)

# Device 
# args.device = torch.device('cuda')
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print("正在加载{}数据集".format(args.dataset))
X = load_data(args.dataset)
X = X[:8640]
times, nodes = X.shape

if args.dataset=="METR_LA":
    adj, _ = get_adjacency_matrix("data/" + args.dataset + "/" + "adj.npy", nodes)

else:
    adj, _ = get_adjacency_matrix("data/" + args.dataset + "/" + "distance.csv",  nodes)


A_wave = scaled_Laplacian(adj).astype(np.float32)
A_wave = torch.from_numpy(A_wave).to(device=args.device)
# A_wave = A_wave.to(torch.float32)

# print(X.shape)
split_line1 = int(X.shape[0] * args.train_per)
split_line2 = int(X.shape[0] * args.val_per)
train_original_data = X[:split_line1, :]
val_original_data = X[split_line1:(split_line1+split_line2), :]
test_original_data = X[(split_line1+split_line2):, :]

train_dataloder= generate_dataset(train_original_data, 
                                    num_timesteps_input=args.input_timesteps,
                                    num_timesteps_output=args.output_timesteps,
                                    batch_size=args.batchsize)
val_dataloader = generate_dataset(val_original_data,
                                    num_timesteps_input=args.input_timesteps,
                                    num_timesteps_output=args.output_timesteps,
                                    batch_size=args.batchsize)
test_dataloader = generate_dataset(test_original_data,
                                    num_timesteps_input=args.input_timesteps,
                                    num_timesteps_output=args.output_timesteps,
                                    batch_size=args.batchsize,
                                    random=False)
print("{}数据加载完成".format(args.dataset))


if __name__ == '__main__':

    # 加载模型
    models_to_run = ["LR", "GRU", "LSTM", "STGCN", "GWN", "ARIMA"]

    for model_name in models_to_run:
        args.model = model_name  # Set the current model
        print(f"Running model: {model_name}")

        #print("模型名称:", args.model)
        #assert args.model in ["HA", "LR", "ARIMA", "LSTM", "GRU", "STGCN", "GWN"],  "目前没有{}这个模型".format(args.model)
        # HA不需要训练，不需要放入cuda，直接获取指标就行
        model_dir = os.path.join("saved_model", args.dataset, model_name)
        os.makedirs(model_dir, exist_ok=True)

        if args.model == "LR":
            net = LR(args.input_timesteps, args.output_timesteps).to(device=args.device)
        elif args.model == "GRU":
            net = GRU(nodes, 1, args.output_timesteps).to(device=args.device)
        elif args.model == "LSTM":
            net = LSTM(nodes, 64, 3, nodes, args.batchsize, output_timestemps=args.output_timesteps, device=args.device).to(device=args.device)
        elif args.model == "GWN":
            net = GWN(args.device, nodes, out_dim=args.output_timesteps).to(device=args.device)
        elif args.model == "STGCN":
            net = STGCN(nodes, 1, args.input_timesteps, args.output_timesteps).to(device=args.device)
        elif args.model in ["HA", "ARIMA"]:
            print("该算法不需要预训练模型，将直接测试")
            args.mode = "test"
        # else:
        #     print("目前没有{}这个模型".format(args.model))
        #     exit()

        if args.mode == "train":
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
            # loss_criterion = nn.L1Loss()
            loss_criterion = nn.MSELoss()
            # loss_criterion = nn.BCELoss()
            for epoch in range(args.epochs):
                # 训练损失
                train_loss = train_epoch(train_dataloder)
                # 验证损失
                val_losses = []
                mapes = []
                maes = []
                rmses = []
                with torch.no_grad():
                    net.eval()
                    for i, (x_batch, y_batch) in enumerate(val_dataloader):
                        x_batch = x_batch.to(torch.float32)
                        y_batch = y_batch.to(torch.float32)
                        x_batch = x_batch.to(device=args.device)
                        y_batch = y_batch.to(device=args.device)
                        out = net(x_batch, A_wave)

                        val_loss = loss_criterion(out, y_batch)
                        val_losses.append(val_loss.detach().cpu().numpy())

                        y_true = y_batch.detach().cpu().numpy()
                        y_pred = out.detach().cpu().numpy()

                        maes.append(masked_mae_test(y_true, y_pred))
                        rmses.append(masked_rmse_test(y_true, y_pred))

                    epoch_val_loss = sum(val_losses)/len(val_losses)
                    mae = sum(maes)/len(maes)
                    rmse = sum(rmses)/len(rmses)
                print("epoch:{} train_loss:{:.5f} val_loss:{:.5f} mae:{:.5f} rmse:{:.5f}".format(epoch,train_loss,epoch_val_loss,mae,rmse))
                torch.save(net, "saved_model/" + args.dataset + "/ "+ args.model + "_" + args.dataset + ".pth")
        elif args.mode == "test":
            results_groundtruth = []
            results_pred = []
            if args.model == "HA":
                val_losses = []
                mapes = []
                maes = []
                rmses = []
                for i, (x_batch, y_batch) in enumerate(test_dataloader):
                    x_batch = x_batch.to(torch.float32)
                    y_batch = y_batch.to(torch.float32)
                    x_batch = x_batch.to(device=args.device)
                    y_batch = y_batch.to(device=args.device)
                    out = HA(x_batch, input_times=args.input_timesteps, output_times=args.output_timesteps, lap=A_wave)

                    y_true = y_batch.detach().cpu().numpy()
                    y_pred = out.detach().cpu().numpy()

                    # 保存预测结果
                    results_groundtruth.append(y_true)
                    results_pred.append(y_pred)

                    maes.append(masked_mae_test(y_true, y_pred))
                    rmses.append(masked_rmse_test(y_true, y_pred))

                mae = sum(maes)/len(maes)
                rmse = sum(rmses)/len(rmses)
                print("mae:{:.5f} rmse:{:.5f}".format(mae,rmse))

                # 保存预测指标
                save_metrics2csv(args.model, args.dataset, mae, rmse)
                save_data2numpy(args.model, args.dataset, results_groundtruth, results_pred)

            elif args.model == "ARIMA":  # ARIMA运行很慢，选择小部分数据测试
                val_losses = []
                mapes = []
                maes = []
                rmses = []
                for i, (x_batch, y_batch) in enumerate(test_dataloader):
                    x_batch = x_batch.numpy()
                    y_batch = y_batch.numpy()
                    out = ARIMA(x_batch, out_steps=args.output_timesteps)
                    maes.append(masked_mae_test(y_batch, out))
                    rmses.append(masked_rmse_test(y_batch, out))
                    break  # 选择第一个Batch数据测试

                mae = sum(maes)/len(maes)
                rmse = sum(rmses)/len(rmses)
                print("mae:{:.5f} rmse:{:.5f}".format(mae,rmse))
                save_metrics2csv(args.model, args.dataset, mae, rmse)

            else:
                net = torch.load("saved_model/" + args.dataset + "/ "+ args.model + "_" + args.dataset + ".pth")
                val_losses = []
                mapes = []
                maes = []
                rmses = []
                with torch.no_grad():
                    net.eval()
                    for i, (x_batch, y_batch) in enumerate(test_dataloader):
                        x_batch = x_batch.to(torch.float32)
                        y_batch = y_batch.to(torch.float32)
                        x_batch = x_batch.to(device=args.device)
                        y_batch = y_batch.to(device=args.device)
                        out = net(x_batch, lap=A_wave)

                        y_true = y_batch.detach().cpu().numpy()
                        y_pred = out.detach().cpu().numpy()

                        # 保存预测结果
                        results_groundtruth.append(y_true)
                        results_pred.append(y_pred)

                        maes.append(masked_mae_test(y_true, y_pred))
                        rmses.append(masked_rmse_test(y_true, y_pred))

                    mae = sum(maes)/len(maes)
                    rmse = sum(rmses)/len(rmses)
                    print("mae:{:.5f} rmse:{:.5f}".format(mae,rmse))

                    # 保存预测指标
                    save_metrics2csv(args.model, args.dataset, mae, rmse)
                    save_data2numpy(args.model, args.dataset, results_groundtruth, results_pred)

    print("训练完毕.")



