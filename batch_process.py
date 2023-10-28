import os
import argparse
from PIL import Image
from tqdm import tqdm
import torch

# seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from task_adapter.seem.tasks import (
    interactive_seem_m2m_auto,
    inference_seem_pano,
    inference_seem_interactive,
)

# semantic sam

import sys

sys.path.append("/home/mreza/src/Semantic-SAM")

from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from utils.dist import init_distributed_mode
from utils.arguments import load_opt_from_config_file
from utils.constants import COCO_PANOPTIC_CLASSES

from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto, prompt_switch

# sam
from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
from task_adapter.sam.tasks.inference_sam_m2m_interactive import (
    inference_sam_m2m_interactive,
)

from scipy.ndimage import label
import numpy as np


"""
build args
"""
semsam_cfg = "configs/semantic_sam_only_sa-1b_swinL.yaml"
seem_cfg = "configs/seem_focall_unicl_lang_v1.yaml"

semsam_ckpt = "./swinl_only_sam_many2many.pth"
sam_ckpt = "./sam_vit_h_4b8939.pth"
seem_ckpt = "./seem_focall_v1.pt"

opt_semsam = load_opt_from_config_file(semsam_cfg)
opt_seem = load_opt_from_config_file(seem_cfg)
opt_seem = init_distributed_seem(opt_seem)


"""
build model
"""
model_semsam = (
    BaseModel(opt_semsam, build_model(opt_semsam))
    .from_pretrained(semsam_ckpt)
    .eval()
    .cuda()
)
model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
model_seem = (
    BaseModel_Seem(opt_seem, build_model_seem(opt_seem))
    .from_pretrained(seem_ckpt)
    .eval()
    .cuda()
)

with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            COCO_PANOPTIC_CLASSES + ["background"], is_eval=True
        )


@torch.no_grad()
def inference(image, slider, mode, alpha, label_mode, anno_mode, *args, **kwargs):
    if slider < 1.5:
        model_name = "seem"
    elif slider > 2.5:
        model_name = "sam"
    else:
        if mode == "Automatic":
            model_name = "semantic-sam"
            if slider < 1.5 + 0.14:
                level = [1]
            elif slider < 1.5 + 0.28:
                level = [2]
            elif slider < 1.5 + 0.42:
                level = [3]
            elif slider < 1.5 + 0.56:
                level = [4]
            elif slider < 1.5 + 0.70:
                level = [5]
            elif slider < 1.5 + 0.84:
                level = [6]
            else:
                level = [6, 1, 2, 3, 4, 5]
        else:
            model_name = "sam"

    if label_mode == "Alphabet":
        label_mode = "a"
    else:
        label_mode = "1"

    text_size, hole_scale, island_scale = 640, 100, 100
    text, text_part, text_thresh = "", "", "0.0"
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        semantic = False

        if mode == "Interactive":
            labeled_array, num_features = label(np.asarray(image["mask"].convert("L")))
            spatial_masks = torch.stack(
                [torch.from_numpy(labeled_array == i + 1) for i in range(num_features)]
            )

        if model_name == "semantic-sam":
            model = model_semsam
            output = inference_semsam_m2m_auto(
                model,
                image["image"],
                level,
                text,
                text_part,
                text_thresh,
                text_size,
                hole_scale,
                island_scale,
                semantic,
                label_mode=label_mode,
                alpha=alpha,
                anno_mode=anno_mode,
                *args,
                **kwargs,
            )

        elif model_name == "sam":
            model = model_sam
            if mode == "Automatic":
                output = inference_sam_m2m_auto(
                    model, image["image"], text_size, label_mode, alpha, anno_mode
                )
            elif mode == "Interactive":
                output = inference_sam_m2m_interactive(
                    model,
                    image["image"],
                    spatial_masks,
                    text_size,
                    label_mode,
                    alpha,
                    anno_mode,
                )

        elif model_name == "seem":
            model = model_seem
            if mode == "Automatic":
                output = inference_seem_pano(
                    model, image["image"], text_size, label_mode, alpha, anno_mode
                )
            elif mode == "Interactive":
                output = inference_seem_interactive(
                    model,
                    image["image"],
                    spatial_masks,
                    text_size,
                    label_mode,
                    alpha,
                    anno_mode,
                )

        return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch process images for Visual Grounding."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing images.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to save processed images.",
    )
    parser.add_argument(
        "--granularity", type=float, default=2, help="Granularity level [1 to 3]."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="Automatic",
        choices=["Automatic", "Interactive"],
        help="Segmentation Mode.",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Mask Alpha [0 to 1].")
    parser.add_argument(
        "--label_mode",
        type=str,
        default="Number",
        choices=["Number", "Alphabet"],
        help="Label Mode.",
    )
    parser.add_argument(
        "--anno_mode",
        nargs="+",
        default=["Mask", "Mark"],
        choices=["Mask", "Box", "Mark"],
        help="Annotation Mode.",
    )
    return parser.parse_args()


def batch_process_images(args):
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    image_files = [
        f
        for f in os.listdir(args.input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    for image_file in tqdm(image_files, desc="Processing Images"):
        try:
            image_path = os.path.join(args.input_folder, image_file)
            image = Image.open(image_path).convert("RGB")

            # Assuming the 'inference' function expects a dictionary
            image_data = {"image": image}

            # Process the image
            output = inference(
                image_data,
                args.granularity,
                args.mode,
                args.alpha,
                args.label_mode,
                args.anno_mode,
            )

            # Convert numpy.ndarray to PIL.Image and save
            if isinstance(output, np.ndarray):
                output_image = Image.fromarray(output.astype(np.uint8))
                output_image.save(os.path.join(args.output_folder, image_file))
            else:
                raise TypeError("Unsupported output type from inference")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")


def main():
    args = parse_args()
    batch_process_images(args)


if __name__ == "__main__":
    main()
