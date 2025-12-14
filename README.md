# DriveStudio

这是DriveStudio的fork版本，修改内容：

- 使用ruff格式化代码提交
- 修改extract_mask，使用huggingface上的模型

## Installation

```bash
conda create -n drivestudio python=3.10 -y
conda activate drivestudio

pip install -r requirements.txt
pip install --no-build-isolation -r requirements-extra.txt

# For development
pre-commit install

```

## 数据处理、训练（以nuscenes为例）

```bash
export PYTHONPATH=$(pwd)

python datasets/preprocess.py \
    --data_root data/nuscenes/raw \
    --target_dir data/nuscenes/processed \
    --dataset nuscenes \
    --split v1.0-mini \
    --start_idx 0 \
    --num_scenes 10 \
    --interpolate_N 4 \
    --workers 32 \
    --process_keys images lidar calib dynamic_masks objects

split=mini
python datasets/tools/extract_masks.py \
    --data_root data/nuscenes/processed_10Hz/$split \
    --start_idx 0 \
    --num_scenes 10 \
    --process_dynamic_mask
```

```bash
export PYTHONPATH=$(pwd)

config_path=configs/streetgs.yaml
dataset=nuscenes/6cams
scene_idx=0

output_root=logs
expname=nuscenes

start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame

python tools/train.py \
    --config_file $config_path \
    --output_root $output_root \
    --run_name $expname \
    --enable_viewer \
    dataset=$dataset \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```
