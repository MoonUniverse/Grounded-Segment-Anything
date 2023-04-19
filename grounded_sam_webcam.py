import argparse
import os
import copy

import numpy as np
import json
import torch
import time
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor, sam_model_registry
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


def load_image(image_cv):
    # load image
    image_pil =  image_cv# load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(Image.fromarray(image_pil), None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_type", type=str, required=True, help="sam type mode"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")

    parser.add_argument("--box_threshold", type=float, default=0.4, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cuda", help="running on cuda only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    text_prompt = args.text_prompt
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    sam_type = args.sam_type

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    build_sam = sam_model_registry[sam_type]
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    
    # webcam 
    cam = cv2.VideoCapture(0)
    
    while True:
        ret_val, img = cam.read()
        start_time = time.time()
        if(ret_val):
            print("img.shpe",img.shape)
            image_pil,image = load_image(img)
            
            # run grounding dino model
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, text_prompt, box_threshold, text_threshold, device=device
            )

            image_cv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_cv)
            size = image_pil.shape[:2]
        
            W , H = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(device)

            if(transformed_boxes.size(0) != 0):
                masks, _, _ = predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes.to(device),
                    multimask_output = False,
                )
                
                # 将边界框和掩模绘制在图像上
                for box, label in zip(boxes_filt, pred_phrases):
                    x0, y0 = box.numpy()[0], box.numpy()[1]
                    w, h = box.numpy()[2] - box.numpy()[0], box.numpy()[3] - box.numpy()[1]
                    cv2.rectangle(img, (int(x0), int(y0)), (int(x0+w), int(y0+h)), (0, 255, 0), 2)
                
                for mask in masks:
                    mask_numpy = mask.cpu().numpy()
                    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
                    h, w = mask_numpy.shape[-2:]
                    mask_image = (mask_numpy.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255).astype(np.uint8)
                    mask_image = mask_image[:, :, :3]
                    cv2.imshow('mask', mask_image)

                    cv2.addWeighted(mask_image, 0.5, img, 0.5, 0, dst=img)
                
                cv2.imshow('Result', img)
                cv2.waitKey(1)
                print("whole time: ", time.time() - start_time)
                # # draw output image
                # plt.figure(figsize=(10, 10))
                # plt.imshow(image_cv)
                # for mask in masks:
                #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                # for box, label in zip(boxes_filt, pred_phrases):
                #     show_box(box.numpy(), plt.gca(), label)

                # plt.axis('off')
                # plt.show()

