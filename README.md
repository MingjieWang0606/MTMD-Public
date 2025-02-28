# MTMD: Multi-Scale Temporal Memory Learning and Efficient Debiasing Framework for Stock Trend Forecasting
The official implementation of the paper "[MTMD: Multi-Scale Temporal Memory Learning and Efficient Debiasing Framework for Stock Trend Forecasting](https://ieeexplore.ieee.org/document/10906481/)".
![image](https://i.ibb.co/5MFPqTJ/12.png)

🎺🎺🎺 Good News! We have established new code in the QLIB library, which allows you to test MTMD with dozens of models and larger datasets simultaneously! And excitingly, MTMD remains the SOTA (State Of The Art) model. Please check [here](https://github.com/tianshijing/qlib/blob/main/examples/benchmarks/README.md)! 

🔈🔈🔈 Good News! We are thrilled to announce that our article has been officially published in IEEE Transactions on Emerging Topics in Computational Intelligence.

## Environment
1. Install python3.7, 3.8 or 3.9. 
2. Install the requirements in [requirements.txt](https://github.com/Wentao-Xu/HIST/blob/main/requirements.txt).
3. Install the quantitative investment platform [Qlib](https://github.com/microsoft/qlib) and download the data from Qlib:
	```
	# install Qlib from source
	pip install --upgrade  cython
	git clone https://github.com/microsoft/qlib.git && cd qlib
	python setup.py install

	# Download the stock features of Alpha360 from Qlib
	python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --version v2
	mkdir data
	```
4. Please download the [concept matrix](https://github.com/Wentao-Xu/HIST/tree/main/data), which is provided by [tushare](https://tushare.pro/document/2?doc_id=81).
5. Please put the concept data and stock data in the new' data' folder.


## The result in qlib:

| Model Name                                | Dataset  | IC          | ICIR        | Rank IC     | Rank ICIR   | Annualized Return | Information Ratio | Max Drawdown |
|-------------------------------------------|----------|-------------|-------------|-------------|-------------|-------------------|-------------------|--------------|
| Transformer(Ashish Vaswani, et al.)       | Alpha360 | 0.0114±0.00 | 0.0716±0.03 | 0.0327±0.00 | 0.2248±0.02 | -0.0270±0.03      | -0.3378±0.37      | -0.1653±0.05 |
| TabNet(Sercan O. Arik, et al.)            | Alpha360 | 0.0099±0.00 | 0.0593±0.00 | 0.0290±0.00 | 0.1887±0.00 | -0.0369±0.00      | -0.3892±0.00      | -0.2145±0.00 |
| MLP                                       | Alpha360 | 0.0273±0.00 | 0.1870±0.02 | 0.0396±0.00 | 0.2910±0.02 | 0.0029±0.02       | 0.0274±0.23       | -0.1385±0.03 |
| Localformer(Juyong Jiang, et al.)         | Alpha360 | 0.0404±0.00 | 0.2932±0.04 | 0.0542±0.00 | 0.4110±0.03 | 0.0246±0.02       | 0.3211±0.21       | -0.1095±0.02 |
| CatBoost((Liudmila Prokhorenkova, et al.) | Alpha360 | 0.0378±0.00 | 0.2714±0.00 | 0.0467±0.00 | 0.3659±0.00 | 0.0292±0.00       | 0.3781±0.00       | -0.0862±0.00 |
| XGBoost(Tianqi Chen, et al.)              | Alpha360 | 0.0394±0.00 | 0.2909±0.00 | 0.0448±0.00 | 0.3679±0.00 | 0.0344±0.00       | 0.4527±0.02       | -0.1004±0.00 |
| DoubleEnsemble(Chuheng Zhang, et al.)     | Alpha360 | 0.0390±0.00 | 0.2946±0.01 | 0.0486±0.00 | 0.3836±0.01 | 0.0462±0.01       | 0.6151±0.18       | -0.0915±0.01 |
| LightGBM(Guolin Ke, et al.)               | Alpha360 | 0.0400±0.00 | 0.3037±0.00 | 0.0499±0.00 | 0.4042±0.00 | 0.0558±0.00       | 0.7632±0.00       | -0.0659±0.00 |
| TCN(Shaojie Bai, et al.)                  | Alpha360 | 0.0441±0.00 | 0.3301±0.02 | 0.0519±0.00 | 0.4130±0.01 | 0.0604±0.02       | 0.8295±0.34       | -0.1018±0.03 |
| ALSTM (Yao Qin, et al.)                   | Alpha360 | 0.0497±0.00 | 0.3829±0.04 | 0.0599±0.00 | 0.4736±0.03 | 0.0626±0.02       | 0.8651±0.31       | -0.0994±0.03 |
| LSTM(Sepp Hochreiter, et al.)             | Alpha360 | 0.0448±0.00 | 0.3474±0.04 | 0.0549±0.00 | 0.4366±0.03 | 0.0647±0.03       | 0.8963±0.39       | -0.0875±0.02 |
| ADD                                       | Alpha360 | 0.0430±0.00 | 0.3188±0.04 | 0.0559±0.00 | 0.4301±0.03 | 0.0667±0.02       | 0.8992±0.34       | -0.0855±0.02 |
| GRU(Kyunghyun Cho, et al.)                | Alpha360 | 0.0493±0.00 | 0.3772±0.04 | 0.0584±0.00 | 0.4638±0.03 | 0.0720±0.02       | 0.9730±0.33       | -0.0821±0.02 |
| AdaRNN(Yuntao Du, et al.)                 | Alpha360 | 0.0464±0.01 | 0.3619±0.08 | 0.0539±0.01 | 0.4287±0.06 | 0.0753±0.03       | 1.0200±0.40       | -0.0936±0.03 |
| GATs (Petar Velickovic, et al.)           | Alpha360 | 0.0476±0.00 | 0.3508±0.02 | 0.0598±0.00 | 0.4604±0.01 | 0.0824±0.02       | 1.1079±0.26       | -0.0894±0.03 |
| TCTS(Xueqing Wu, et al.)                  | Alpha360 | 0.0508±0.00 | 0.3931±0.04 | 0.0599±0.00 | 0.4756±0.03 | 0.0893±0.03       | 1.2256±0.36       | -0.0857±0.02 |
| TRA(Hengxu Lin, et al.)                   | Alpha360 | 0.0485±0.00 | 0.3787±0.03 | 0.0587±0.00 | 0.4756±0.03 | 0.0920±0.03       | 1.2789±0.42       | -0.0834±0.02 |
| IGMTF(Wentao Xu, et al.)                  | Alpha360 | 0.0480±0.00 | 0.3589±0.02 | 0.0606±0.00 | 0.4773±0.01 | 0.0946±0.02       | 1.3509±0.25       | -0.0716±0.02 |
| HIST(Wentao Xu, et al.)                   | Alpha360 | 0.0522±0.00 | 0.3530±0.01 | 0.0667±0.00 | 0.4576±0.01 | 0.0987±0.02       | 1.3726±0.27       | -0.0681±0.01 |
| KRNN                                      | Alpha360 | 0.0173±0.01 | 0.1210±0.06 | 0.0270±0.01 | 0.2018±0.04 | -0.0465±0.05      | -0.5415±0.62      | -0.2919±0.13 |
| Sandwich                                  | Alpha360 | 0.0258±0.00 | 0.1924±0.04 | 0.0337±0.00 | 0.2624±0.03 | 0.0005±0.03       | 0.0001±0.33       | -0.1752±0.05 |
| MTMD(Mingjie Wang, et al.)                | Alpha360 | 0.0538±0.00 | 0.3849±0.01 | 0.0672±0.00 | 0.4656±0.01 | 0.1022±0.02       | 1.4031±0.26       | -0.0664±0.01 |



## Reproduce the stock trend forecasting results
![image](https://i.ibb.co/X7CVp2v/res.png)

```
git clone https://github.com/MingjieWang0606/MTMD-Public.git
cd MTMD-Public
mkdir output
```

### Reproduce our MTMD framework
```
# CSI 100
python learn_memory.py --model_name HIST --data_set csi100 --hidden_size 128 --num_layers 2 --outdir ./output/csi100_MTMD

# CSI 300
python learn_memory.py --model_name HIST --data_set csi300 --hidden_size 128 --num_layers 2 --outdir ./output/csi300_MTMD
```

### Reproduce our HIST framework
```
# CSI 100
python learn.py --model_name HIST --data_set csi100 --hidden_size 128 --num_layers 2 --outdir ./output/csi100_HIST

# CSI 300
python learn.py --model_name HIST --data_set csi300 --hidden_size 128 --num_layers 2 --outdir ./output/csi300_HIST
```
### Reproduce the baselines
* MLP 
```
# MLP on CSI 100
python learn.py --model_name MLP --data_set csi100 --hidden_size 512 --num_layers 3 --outdir ./output/csi100_MLP

# MLP on CSI 300
python learn.py --model_name MLP --data_set csi300 --hidden_size 512 --num_layers 3 --outdir ./output/csi300_MLP
```

* LSTM
```
# LSTM on CSI 100
python learn.py --model_name LSTM --data_set csi100 --hidden_size 128 --num_layers 2 --outdir ./output/csi100_LSTM

# LSTM on CSI 300
python learn.py --model_name LSTM --data_set csi300 --hidden_size 128 --num_layers 2 --outdir ./output/csi300_LSTM
```

* GRU
```
# GRU on CSI 100
python learn.py --model_name GRU --data_set csi100 --hidden_size 128 --num_layers 2 --outdir ./output/csi100_GRU

# GRU on CSI 300
python learn.py --model_name GRU --data_set csi300 --hidden_size 64 --num_layers 2 --outdir ./output/csi300_GRU
```

* SFM
```
# SFM on CSI 100
python learn.py --model_name SFM --data_set csi100 --hidden_size 64 --num_layers 2 --outdir ./output/csi100_SFM

# SFM on CSI 300
python learn.py --model_name SFM --data_set csi300 --hidden_size 128 --num_layers 2 --outdir ./output/csi300_SFM
```

* GATs
```
# GATs on CSI 100
python learn.py --model_name GATs --data_set csi100 --hidden_size 128 --num_layers 2 --outdir ./output/csi100_GATs

# GATs on CSI 300
python learn.py --model_name GATs --data_set csi300 --hidden_size 64 --num_layers 2 --outdir ./output/csi300_GATs
```

* ALSTM
```
# ALSTM on CSI 100
python learn.py --model_name ALSTM --data_set csi100 --hidden_size 64 --num_layers 2 --outdir ./output/csi100_ALSTM

# ALSTM on CSI 300
python learn.py --model_name ALSTM --data_set csi300 --hidden_size 128 --num_layers 2 --outdir ./output/csi300_ALSTM
```

* Transformer
```
# Transformer on CSI 100
python learn.py --model_name Transformer --data_set csi100 --hidden_size 32 --num_layers 3 --outdir ./output/csi100_Transformer

# Transformer on CSI 300
python learn.py --model_name Transformer --data_set csi300 --hidden_size 32 --num_layers 3 --outdir ./output/csi300_Transformer
```

* ALSTM+TRA 

We reproduce the ALSTM+TRA with its [source code](https://github.com/microsoft/qlib/tree/main/examples/benchmarks/TRA).

### Acknowledgements
Special thanks to ChenFeng, Zhang Mingze,Tian Junxi and LiTingXin for the their help and discussion!  
Thanks for the clean and efficient [HIST](https://github.com/Wentao-Xu/HIST) code.  


## Citation
Please cite the following paper if you use this code in your work.
```
@article{wang2022mtmd,
  title={MTMD: Multi-Scale Temporal Memory Learning and Efficient Debiasing Framework for Stock Trend Forecasting},
  author={Mingjie Wang and Juanxi Tian and Mingze Zhang and Jianxiong Guo and Weijia Jia},
  journal={arXiv preprint arXiv:2212.08656},
  year={2022}
}
```

