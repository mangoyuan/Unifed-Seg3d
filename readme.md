# Unified generative adversarial networks for multimodal segmentation from unpaired 3D medical images

## Installation
- Run on python3.6, Pytorch1.1 and CUDA 10.1.
- A GPU with 11G memory.
- Clone this repo, which based on an old version of nnUNet.
- Download `Task01_BrainTumour` dataset from [http://medicaldecathlon.com/](http://medicaldecathlon.com/).

## Create new raw dataset

1. random select one modality for each case and save as `OMBTS/caseid_modal.yaml`.
2. create a `base` dir liked this
```bash
base
|--nnUNet_raw
```
3. create new dataset liked `Task20_OMBTS` and put in `base/nnUNet_raw`.
```bash
cd OMBTS
python create_dataset.py --task_dir /path/to/Task01_BrainTumour --out_dir /path/to/base/nnUNet_raw/Task20_OMBTS -c2m caseid_modal.yaml
```

## Planning

1. change some variable in `nnunet/path.py`.
```python
# nnunet/path.py
base = '/path/to/base'
# ...
caseid_modal_path = '/absolute/path/to/OMBTS/caseid_modal.yaml'
```

2. pre-processing and planning, fixed `batch_size=2, 'patch_size=[96, 128, 128]` were used for limited GPU mem.
```python
cd nnunet
python nnunet/experiment_planning/plan_and_preprocess_task.py -t Task20_OMBTS
```

## Training

1. train with all data.
```python
# ours, base_num_feature=12
OMP_NUM_THREADS=0 CUDA_VISIBLE_DEVICES=0 python nnunet/run/run_training.py 3d_fullres uaganTrainer Task20_OMBTS all --ndet
```

## Inference

1. A sample bash, which infers with the final checkpoint.

```bash 
BASE=/path/to/base
TRAINER=uaganTrainer
OUTPUT=${BASE}/nnUNet/3d_fullres/Task20_OMBTS/${TRAINER}__nnUNetPlans/all

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python nnunet/inference/predict_simple.py -f all \
 -i ${BASE}/nnUNet_raw_splitted/Task20_OMBTS/imagesTs \
 -o ${OUTPUT}/testing \
 -t Task20_OMBTS -tr ${TRAINER} -m 3d_fullres
```

## Citation

```
@article{yuan2020unified,
  title={Unified generative adversarial networks for multimodal segmentation from unpaired 3D medical images},
  author={Yuan, Wenguang and Wei, Jia and Wang, Jiabing and Ma, Qianli and Tasdizen, Tolga},
  journal={Medical Image Analysis},
  volume={64},
  pages={101731},
  year={2020},
  publisher={Elsevier}
}
```
