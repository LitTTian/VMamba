# VMamba-复现
- 参考论文[VMamba](https://arxiv.org/abs/2401.10166)和[VMamba代码仓库](https://github.com/MzeroMiko/VMamba)整理了一个简洁版本的模型实现。
- 需要前往[ssm代码仓库](https://github.com/state-spaces/mamba/releases/)安装适配的`mamba_ssm`来使用torch下的优化版本的`selective_scan_cuda`

## 训练
```bash
python train.py --dataset_path <cifar-10数据集路径> --save_dir <模型保存路径> --epochs <训练轮数>
```

## Citation
```bibtex
@article{liu2024vmamba,
  title={VMamba: Visual State Space Model},
  author={Liu, Yue and Tian, Yunjie and Zhao, Yuzhong and Yu, Hongtian and Xie, Lingxi and Wang, Yaowei and Ye, Qixiang and Liu, Yunfan},
  journal={arXiv preprint arXiv:2401.10166},
  year={2024}
}
```