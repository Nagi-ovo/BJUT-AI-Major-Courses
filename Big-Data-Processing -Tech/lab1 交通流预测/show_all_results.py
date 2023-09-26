# 读取测试结果并生成图片和表格

from tabulate import tabulate
# import wcwidth
# import pandas as pd
import numpy as np
import argparse
import csv
# from tqdm import tqdm 
import matplotlib.pyplot as plt 
import os

# 读取预测指标数据
def read_csvfile(filepath):
    data = []
    with open(filepath) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        #header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader:            # 将csv 文件中的数据保存到data中
            data.append(row)              # 选择某一列加入到data数组中
    mae = float(data[1][0])
    rmse = float(data[1][1])
    return mae, rmse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="METR_LA")
# parser.add_argument('--model', type=str, default="HA")
parser.add_argument('--model', nargs='+', help='<Required> Set flag', required=True, default=["HA"])

args = parser.parse_args()

# 输出表格
modelnames = ["HA", "LR", "ARIMA", "LSTM", "GRU", "STGCN", "GWN"]
table_header = ['算法名称/指标','mae','rmse']
metrics = []
for modelname in modelnames:
    csv_file_path = "test_results/" + args.dataset + "/" + "csvfiles/" + modelname + "_" + args.dataset + '.csv'
    assert os.path.exists(csv_file_path), "请确认{}算法是否已经被预训练".format(modelname)
    mae, rmse = read_csvfile(csv_file_path)
    metrics.append([modelname, mae, rmse])
print(tabulate(metrics, headers=table_header, tablefmt='fancy_grid'))


# 读取某一条预测数据, 不包含ARIMA
modelnames = ["HA", "LR", "LSTM", "GRU", "STGCN", "GWN"]
table_header = ['算法名称/指标','mae','rmse']
node_choose = 30

true_data = np.load("test_results/" + args.dataset + "/" + "true_data.npy")
data_len = list(range(len(true_data)))

pred_data = {}
for modelname in modelnames:
    pred = []
    data = np.load("test_results/" + args.dataset + "/" + "numpyfiles/" + modelname + "_" + args.dataset + '.npz')
    here = data["true"]
    here = here[:,:,node_choose,0].reshape(71*24,-1)
    pred_copy = here[:]
    pred_data.update({modelname:pred_copy})


# 输出曲线
plt.figure(figsize=(12,12))    # 设置绘图大小为12*12
plt.plot(data_len[:100], true_data[:100], label="true_data", color='blue', linewidth=3)
for model in args.model:
    plt.plot(data_len[:100], pred_data[model][:100], label=model)

plt.tick_params(labelsize=15)
plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.tick_params(labelsize=15)
plt.gcf().subplots_adjust(bottom=0.2)
plt.legend(prop = {'size':12})

'''
这里做出了修改，现在不需要在每次运行后都保存结果避免图片覆盖。
'''
# 创建一个包含所有选定模型名称的字符串
model_name_string = "_".join(args.model)
# 使用该字符串在保存图片时命名文件
pic_save_path = f"test_results/{args.dataset}/{model_name_string}_predict.png"

plt.savefig(pic_save_path)      
plt.show()