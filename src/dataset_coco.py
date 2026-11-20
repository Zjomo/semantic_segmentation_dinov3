from torch.utils.data import Dataset
import cv2
import numpy as np
import random
import os
import pandas as pd
from src.common import image_to_tensor, tensor_to_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import json
import torch
import albumentations as A


def trans_anno(data_1_anno, data_1_img):
    anno_dict = {}
    # 遍历 data_1_anno 中的每个分割实例
    for segment_id, anno in enumerate(data_1_anno):
        category_id = anno['category_id']
        image_id = anno['image_id']
        bbox = anno['bbox']  # [x_min, y_min, width, height]
        x_min, y_min, width, height = bbox
        area = width * height

        # 3类 >> {'id': 1, 'name': 'Road', 'supercategory': ''},  {'id': 2, 'name': 'obstacle', 'supercategory': ''}, {'id': 3, 'name': 'Boundary', 'supercategory': ''}
        if category_id == 1:
            segment_id = 0 + (254 * 256) + (0 * 65536)          # Road -- 绿色
        elif category_id == 2:
            segment_id = 254 + (0 * 256) + (0 * 65536)          # obstacle -- 红色
        elif category_id == 0:
            segment_id = 0 + (0 * 256) + (0 * 65536)          # background -- 红色
        else:
            segment_id = 0 + (0 * 256) + (254 * 65536)          # Boundary -- 靛色

            # 构建 segments_info
        segment_info = {
            'id': segment_id,  # segment_id 从 1 开始
            'category_id': category_id,
            'iscrowd': 0,
            'bbox': bbox,
            'area': area
        }

        # 如果该 image_id 在字典中已经存在，则添加新的 segments_info
        if image_id not in anno_dict:
            anno_dict[image_id] = {
                'image_id': image_id,
                'segments_info': []
            }

        # 将新的 segment_info 添加到该 image_id 下的 segments_info 列表中
        anno_dict[image_id]['segments_info'].append(segment_info)

    # 创建最终的结果列表
    anno_list = []

    for image_id, anno_data in anno_dict.items():
        # 获取该 image_id 对应的文件名
        file_name = data_1_img[data_1_img.id == image_id]['file_name'].values[0]

        # 将该 image_id 和 segments_info 添加到结果中
        anno_list.append({
            'segments_info': anno_data['segments_info'],
            'file_name': file_name,
            'image_id': image_id
        })

    return anno_list

class DatasetCOCOPanoptic(Dataset):
    """
    COCO Panoptic dataset loader.

    Expects COCO-style layout:
      root_dir/
        panoptic_train2017/            # PNG segmentation maps (R,G,B encoding of segment id)
        panoptic_val2017/
        images: train2017/ val2017/
        annotations/
          panoptic_train2017.json
          panoptic_val2017.json

    Returns: image_tensor, semantic_mask (H,W), instance_mask (H,W), segments_info
    """
    def __init__(self, root_dir, mode, img_size, patch_size,
                 augment_prob=0.8,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        self.patch_size = patch_size

        if self.mode == "train":
            self.images_dir = os.path.join(self.root_dir, "train2017")
            self.panoptic_dir = os.path.join(self.root_dir, "annotations", "panoptic_train2017")
            self.path_annotations = os.path.join(self.root_dir, "annotations", "panoptic_train2017.json")
        elif self.mode == "val":
            self.images_dir = os.path.join(self.root_dir, "val2017")
            self.panoptic_dir = os.path.join(self.root_dir, "annotations", "panoptic_val2017")
            self.path_annotations = os.path.join(self.root_dir, "annotations", "panoptic_val2017.json")
        else:
            raise ValueError("mode must be 'train' or 'val'")

        # 导入注释文件 -- Load panoptic JSON (different structure than instances json)
        with open(self.path_annotations, 'r') as f:
            self.panoptic_json = json.load(f)

        # 提取注释文件的 "注释"字段
        self.panoptic_anns = self.panoptic_json['annotations']

        # 提取注释文件的 "图片"字段
        self.panoptic_anns_img = pd.DataFrame(self.panoptic_json['images'])

        # If you also want instance/semantic category info:
        # panoptic JSON includes 'categories' list with 'id', 'name', 'isthing'
        # 提取categories 为 id 的 value
        self.categories = {c['id']: c for c in self.panoptic_json.get('categories', [])}

        # 提取数据集中的categories，排序并提取 name 字段： [{'id': 1, 'name': 'Road', 'supercategory': ''}, {'id': 2, 'name': 'obstacle', 'supercategory': ''}, {'id': 3, 'name': 'Boundary', 'supercategory': ''}])
        self.class_names = [c['name'] for c in sorted(self.categories.values(), key=lambda x: x['id'])]

        # 将 0：背景，补充至类别字典
        self.class_names.insert(0, 'background') # Add background
        print("Total number of panoptic categories:", len(self.class_names))


        # 提取可数的类型
        # thing_ids = [cid for cid, c in self.categories.items() if c["isthing"] == 1]
        thing_ids = [cid for cid, c in self.categories.items() if c.get("isthing", 1) == 1]

        self.thing_names = [self.categories[cid]["name"] for cid in sorted(thing_ids)]
        self.thing_names.insert(0,'background')
        print("Thing names ({}):".format(len(self.thing_names)), self.thing_names[:10], "…")

        self.augment_prob = augment_prob
        self.mean = np.array(mean, dtype=np.float32)[:, None, None]
        self.std = np.array(std, dtype=np.float32)[:, None, None]

        # 构建图片ID：图片地址：宽：高 字典
        self.images = {img['id']: img for img in self.panoptic_json['images']}
        self.ids = list(sorted(self.images.keys()))

        # 图片形态学处理
        self.transform = A.Compose([
            # --- Geometric (apply to both image & depth) ---
            A.HorizontalFlip(p=0.5),

            # a bit more aggressive crop/ratio
            A.RandomResizedCrop(size=(776, 1028),
                                scale=(0.8, 1.0),
                                ratio=(0.95, 1.05),
                                p=0.6),

            # combine small shift/scale/rotate
            A.Affine(
                scale=(0.95, 1.05),               
                translate_percent=(-0.02, 0.02),   
                rotate=(-10, 10),                  
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,        
                fill=0,                             
                fill_mask=0,                        
                p=0.5
            ),

            # --- Photometric (image only) ---
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.RandomGamma(gamma_limit=(90, 110), p=0.2),

            # small amount of blur/noise; OneOf keeps it moderate
            A.OneOf([
                A.GaussNoise(std_range=(0.05, 0.1), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=5),
            ], p=0.3),
        ],
            additional_targets={'semantic': 'mask',
                                'instance': 'mask'}
        )

        random.seed(42)

    def __len__(self):
        return len(self.ids)

    # --- Helpers ----------------------------------------------------------
    @staticmethod
    def _rgb_to_seg_id(rgb):
        """
        Convert an RGB triplet array (H,W,3 uint8) to a single integer segment id per pixel.
        COCO panoptic encoding: id = R + 256*G + 256^2*B
        """
        if rgb.ndim == 2:
            raise ValueError("Expected 3 channel RGB image for panoptic PNG")
        r = rgb[..., 0].astype(np.int32)
        g = rgb[..., 1].astype(np.int32)
        b = rgb[..., 2].astype(np.int32)
        seg_id = r + (g << 8) + (b << 16)
        return seg_id

    # --- Core -------------------------------------------------------------
    def __getitem__(self, idx):
        # 导入图像文件
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        # 原图
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # pan_ann = self.panoptic_anns[img_id]
        # cavt 转为 coco格式，图片id > 索引id
        pan_ann = trans_anno(self.panoptic_anns,self.panoptic_anns_img)
        pan_ann = pan_ann[img_id-1]

        # 提取注释图像文件
        pan_png = img_info['file_name']
        pan_path = os.path.join(self.panoptic_dir, pan_png)
        # 分割原图
        pan_rgb = cv2.imread(pan_path, cv2.IMREAD_UNCHANGED)
        if pan_rgb is None:
            raise FileNotFoundError(f"Panoptic PNG not found: {pan_path}")

        # 图像像素 语义化
        # pan_rgb is read as BGR by OpenCV. Convert to RGB ordering before decoding.
        # if pan_rgb.shape[2] == 3:
        pan_rgb = cv2.cvtColor(pan_rgb, cv2.COLOR_BGR2RGB)
        seg_id_map = self._rgb_to_seg_id(pan_rgb)


        # 初始化 语义掩码、实例掩码
        # 语义掩码：每个像素值 标识该像素对应的类型ID
        semantic_mask = np.zeros_like(seg_id_map, dtype=np.int32)   # category ids

        # 实例掩码：每个像素值 标识该像素值所属的实例ID
        instance_mask = np.zeros_like(seg_id_map, dtype=np.int32)   # instance ids (unique small ints)

        # 分割的问题就出现在这里了，数据需要提供segments_info > 框线位置，但是我的数据集中是轮廓点
        # cavt 是所有标注图 > 某张图片
        # coco 是所有图片 > 某张标注图
        segments = pan_ann.get('segments_info', [])


        # map segment id -> category id and assign instance indices
        segid_to_cat = {}
        segid_to_instid = {}
        next_inst_id = 1  # reserve 0 for background/void

        # We may need to know which categories are 'things' to give instance ids only to things
        isthing_map = {c_id: bool(c.get('isthing', 1)) for c_id, c in self.categories.items()}

        # track stuff categories already assigned an id
        stuff_class_to_instid = {}


        # 对可数背景唯一标注ID,对不可数背景均赋予 0值【问题应该就出现在这一块了】
        for seg in segments:

            sid = seg.get('id')
            print(seg)
            print(sid)

            cid = seg.get('category_id')
            cname = self.categories[cid]['name']
            label = self.class_names.index(cname)
            segid_to_cat[sid] = label
            if isthing_map.get(cid, False):
                segid_to_instid[sid] = next_inst_id
                next_inst_id += 1
            else:
                # With this line, we consider all the "stuff" as 0 class
                segid_to_instid[sid] = 0 

        # 某图像，所有的 掩码ID（同一类型的 id 也不一定相同）
        # 对图像中的每个分块赋予类型ID
        unique_seg_ids = np.unique(seg_id_map)
        for sid in unique_seg_ids:
            # sid为0时，该色素块为背景 background
            if sid == 0:
                # Some datasets may use 0 for background/void; keep as 0
                continue
            if sid in segid_to_cat:
                mask = (seg_id_map == sid)
                semantic_mask[mask] = segid_to_cat[sid]
                instance_mask[mask] = segid_to_instid[sid]
            else:
                # Unknown segment id -> mark as void (category 0)
                continue

        # Optional augment (only photometric; geometric transforms need masks too)
        if random.random() < self.augment_prob:
            aug = self.transform(image=image, semantic=semantic_mask, instance=instance_mask)
            image = aug['image']
            semantic_mask = aug['semantic']
            instance_mask = aug['instance']

        # Resize image using helper, then resize masks to match
        image_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        # get output size
        out_h, out_w = image_resized.shape[:2]

        # resize masks with nearest neighbor to avoid interpolation
        semantic_mask_resized = cv2.resize(semantic_mask.astype(np.int32), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        instance_mask_resized = cv2.resize(instance_mask.astype(np.int32), (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        # convert to tensors
        image_tensor = image_to_tensor(image_resized, self.mean, self.std)
        semantic_mask_t = torch.from_numpy(semantic_mask_resized.astype(np.int64))
        instance_mask_t = torch.from_numpy(instance_mask_resized.astype(np.int32))

        return image_tensor, semantic_mask_t, instance_mask_t

if __name__ == '__main__':
    COCO_ROOT = r'G:\MyDataset\PHD\CoreLingoV2.0_test'

    MODE = "val"
    IMG_SIZE = 1028
    PATCH_SIZE = 16
    AUGMENT_PROB = 1.0
    dataset = DatasetCOCOPanoptic(COCO_ROOT, MODE, IMG_SIZE, PATCH_SIZE, AUGMENT_PROB)
    print("Class names:\n", dataset.class_names)
    # Save class names
    with open("./class_names.txt", "w") as f:
        f.write("\n".join(dataset.class_names))
    data = dataset.__getitem__(37)
    #dataset.visualize(0, 1.0)
    image, semantic_mask, _ = data # We don't use here the instance map
    print("image: ", image.shape, image.dtype)
    print("semantic_mask: ", semantic_mask.shape, semantic_mask.dtype)
    from utils import visualize_maps
    visualize_maps(tensor_to_image(image, dataset.mean, dataset.std),
                   semantic_mask.cpu().numpy(),
                   class_names=dataset.class_names,
                   alpha=0.6)