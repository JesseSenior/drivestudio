"""
@file   extract_masks.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Extract semantic mask

Using SegFormer from Hugging Face Transformers. Cityscapes 83.2%

Installation:
    mamba create -n segformer python=3.10
    mamba activate segformer
    mamba install pytorch::pytorch pytorch::torchvision pytorch::torchaudio -c pytorch
    pip install "transformers<=4.49" pillow opencv-python tqdm imageio scikit-image

Usage:
    Direct run this script in the newly set mamba env.
"""

from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import torch
import numpy as np
from PIL import Image

# fmt: off
semantic_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]
dataset_classes_in_sematic = {
    'Vehicle': [13, 14, 15],   # 'car', 'truck', 'bus'
    'human': [11, 12, 17, 18], # 'person', 'rider', 'motorcycle', 'bicycle'
}
# fmt: on

if __name__ == "__main__":
    import os
    import imageio
    import numpy as np
    from glob import glob
    from tqdm import tqdm
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # Custom configs
    parser.add_argument("--data_root", type=str, default="data/waymo/processed/training")
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=int,
        nargs="+",
        help="scene ids to be processed, a list of integers separated by space. Range: [0, 798] for training, [0, 202] for validation",
    )
    parser.add_argument("--split_file", type=str, default=None, help="Split file in data/waymo_splits")
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If no scene id or split_file is given, use start_idx and num_scenes to generate scene_ids_list",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=200,
        help="number of scenes to be processed",
    )
    parser.add_argument(
        "--process_dynamic_mask",
        action="store_true",
        help="Whether to process dynamic masks",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ignore_existing", action="store_true")
    parser.add_argument("--no_compress", action="store_true")
    parser.add_argument("--rgb_dirname", type=str, default="images")
    parser.add_argument("--mask_dirname", type=str, default="fine_dynamic_masks")

    # Algorithm configs
    # Model configs
    parser.add_argument(
        "--model_name",
        type=str,
        default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        help="Hugging Face model name or path",
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")

    args = parser.parse_args()

    # Load model and feature extractor
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    if args.scene_ids is not None:
        scene_ids_list = args.scene_ids
    elif args.split_file is not None:
        # parse the split file
        split_file = open(args.split_file, "r").readlines()[1:]
        # NOTE: small hack here, to be refined in the futher (TODO)
        if "kitti" in args.split_file or "nuplan" in args.split_file:
            scene_ids_list = [line.strip().split(",")[0] for line in split_file]
        else:
            scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
    else:
        scene_ids_list = np.arange(args.start_idx, args.start_idx + args.num_scenes)

    for scene_i, scene_id in enumerate(tqdm(scene_ids_list, "Extracting Masks ...")):
        scene_id = str(scene_id).zfill(3)
        img_dir = os.path.join(args.data_root, scene_id, args.rgb_dirname)

        # create mask dir
        sky_mask_dir = os.path.join(args.data_root, scene_id, "sky_masks")
        if not os.path.exists(sky_mask_dir):
            os.makedirs(sky_mask_dir)

        # create dynamic mask dir
        if args.process_dynamic_mask:
            rough_human_mask_dir = os.path.join(args.data_root, scene_id, "dynamic_masks", "human")
            rough_vehicle_mask_dir = os.path.join(args.data_root, scene_id, "dynamic_masks", "vehicle")

            all_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "all")
            if not os.path.exists(all_mask_dir):
                os.makedirs(all_mask_dir)
            human_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "human")
            if not os.path.exists(human_mask_dir):
                os.makedirs(human_mask_dir)
            vehicle_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "vehicle")
            if not os.path.exists(vehicle_mask_dir):
                os.makedirs(vehicle_mask_dir)

        flist = sorted(glob(os.path.join(img_dir, "*")))
        for fpath in tqdm(flist, f"scene[{scene_id}]"):
            fbase = os.path.splitext(os.path.basename(os.path.normpath(fpath)))[0]

            # if args.no_compress:
            #     mask_fpath = os.path.join(mask_dir, f"{fbase}.npy")
            # else:
            #     mask_fpath = os.path.join(mask_dir, f"{fbase}.npz")

            if args.ignore_existing and os.path.exists(os.path.join(args.data_root, scene_id, "fine_dynamic_masks")):
                continue

            # ---- Inference and save outputs
            image = Image.open(fpath)
            inputs = image_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits, size=image.size[::-1], mode="bilinear", align_corners=False
            )
            mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            # if args.no_compress:
            #     np.save(mask_fpath, mask)
            # else:
            #     np.savez_compressed(mask_fpath, mask)   # NOTE: compressed files are 100x smaller.

            # save sky mask
            sky_mask = np.isin(mask, [10])
            imageio.imwrite(os.path.join(sky_mask_dir, f"{fbase}.png"), sky_mask.astype(np.uint8) * 255)

            if args.process_dynamic_mask:
                # save human masks
                rough_human_mask_path = os.path.join(rough_human_mask_dir, f"{fbase}.png")
                rough_human_mask = imageio.imread(rough_human_mask_path) > 0
                huamn_mask = np.isin(mask, dataset_classes_in_sematic["human"])
                valid_human_mask = np.logical_and(huamn_mask, rough_human_mask)
                imageio.imwrite(os.path.join(human_mask_dir, f"{fbase}.png"), valid_human_mask.astype(np.uint8) * 255)

                # save vehicle mask
                rough_vehicle_mask_path = os.path.join(rough_vehicle_mask_dir, f"{fbase}.png")
                rough_vehicle_mask = imageio.imread(rough_vehicle_mask_path) > 0
                vehicle_mask = np.isin(mask, dataset_classes_in_sematic["Vehicle"])
                valid_vehicle_mask = np.logical_and(vehicle_mask, rough_vehicle_mask)
                imageio.imwrite(
                    os.path.join(vehicle_mask_dir, f"{fbase}.png"), valid_vehicle_mask.astype(np.uint8) * 255
                )

                # save dynamic mask
                valid_all_mask = np.logical_or(valid_human_mask, valid_vehicle_mask)
                imageio.imwrite(os.path.join(all_mask_dir, f"{fbase}.png"), valid_all_mask.astype(np.uint8) * 255)
