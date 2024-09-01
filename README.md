<h1 align="center">DeepSolo Annotator: Automatically Generate Text Spotting Ground Truths using DeepSolo</h1> 

## Usage

- ### Installation for Linux

Python 3.8 + PyTorch 1.9.0 + CUDA 11.1 + Detectron2 (v0.6)
```
git clone https://github.com/ViTAE-Transformer/DeepSolo.git
cd DeepSolo/DeepSolo
conda create -n annotate python=3.8 -y
conda activate annotate
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
python setup.py build develop
```

- ### Installation for Windows

Python 3.8 + PyTorch 2.0.0 + CUDA 11.8 + Detectron2 Explicitly built like shown [here](https://blog.csdn.net/weixin_42644340/article/details/109178660)
```
git clone https://github.com/ViTAE-Transformer/DeepSolo.git
cd DeepSolo/DeepSolo
conda create -n annotate python=3.8 -y
conda activate annotate
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python setup.py build develop
```


- ### Run Annotator

- #### Use config and model weight of CTW1500 to generate line based annotation and TotalText for word based annotation

Download Weights -> [CTW](https://1drv.ms/u/s!AimBgYV7JjTlgcdsiFgSz-FHgKepqQ?e=56gdHj) | [TOTALTEXT](https://1drv.ms/u/s!AimBgYV7JjTlgcd6XGlbZ-I7WvGslQ?e=rrkXLx)

Save the weight(s) in "weights" directory

### Run the command

```
python demo/demo.py --config-file ${CONFIG_FILE} --input images --output outputs --opts MODEL.WEIGHTS <MODEL_PATH>
```

- ### Verify Annotations

```
python plot_annotations.py
```

## This Annotator is based on [DeepSolo](https://github.com/ViTAE-Transformer/DeepSolo/tree/main/DeepSolo)

```bibtex
@inproceedings{ye2023deepsolo,
  title={DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting},
  author={Ye, Maoyuan and Zhang, Jing and Zhao, Shanshan and Liu, Juhua and Liu, Tongliang and Du, Bo and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19348--19357},
  year={2023}
}
```
