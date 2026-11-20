# Dataset variables
# COCO_ROOT = '/home/rafa/deep_learning/datasets/COCO' # Location of dataset
# COCO_ROOT = r'G:\MyDataset\PHD\COCO'                  # 数据集位置
# COCO_ROOT = r'G:\MyDataset\PHD\CoreLingo_test'        # 数据集位置

# COCO_ROOT = r'G:\MyDataset\PHD\CoreLingoV2.0_test'                # background/Road/obstacle/Boundary (二楼边界)
COCO_ROOT = r'G:\MyDataset\PHD\CoreLingoV2.0_test_Bounday_Black'    # background/Road/obstacle/Boundary (三楼边界)

IMG_SIZE = 640                              # 图片大小  (it will be square)
# IMG_SIZE = 7000                              # 图片大小  (it will be square)
PATCH_SIZE = 16                             # Transformers 嵌入向量的批处理尺寸
PROB_AUGMENT_TRAINING = 0.85        # Probability to perform photogrametric augmentation in training
PROB_AUGMENT_VALID = 0.0            # Probability to perform photogrametric augmentation in validation
IMG_MEAN = [0.485, 0.456, 0.406]    # Mean of the image that the backbone (e.g. ResNet) expects
IMG_STD = [0.229, 0.224, 0.225]     # Std of the image that the backbone (e.g. ResNet) expects

# Model parameters
DINOV3_DIR = r"F:/0_MyProject/dinov3"           # dinov3 项目代码
DINO_MODEL = r"dinov3_vits16plus"                # dinov3 模型类型
DINO_WEIGHTS = r"G:\0_Models\Dino\dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth" # dinov3 模型权重
MODEL_TO_NUM_LAYERS = { # Mapping from model type to number of layers
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}
MODEL_TO_EMBED_DIM = { # Mapping from model type to embedding dimension
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vith16plus": 1280,
    "dinov3_vit7b16": 4096,
}
HIDDEN_DIM=256 # Size of the hidden dim
TARGET_SIZE=320 # Size of the output of the NN (kept below the size img to prevent out of memory CUDA errors)

# TRAINING PARAMETERS
BATCH_SIZE = 8 # Batch size
WEIGHT_LOSS_DICE = 1.0 # Weight of dice loss
WEIGHT_LOSS_FOCAL = 1.0 # Weight of focal loss


LEARNING_RATE = 0.0001 # Learning rate
WEIGHT_DECAY = 0.0001 # Weight decay for regularization
NUM_EPOCHS = 20 # Number of epochs
NUM_SAMPLES_PLOT = 6 # Number of samples to plot during training or validation

LOAD_MODEL = False              # 是否为训练导入一个已存在的模型 ?
SAVE_MODEL = True               # 是否为训练保存结果 ?
MODEL_PATH_TRAIN_LOAD = r'F:/0_MyProject/semantic_segmentation_dinov3/results/2025-09-08_21-50-18/2025-09-09_23-19-30/model_3.pth'  # 模型导入路径
RESULTS_PATH = r'F:\0_My_jupyter\0_Optical\results'                                                              # 结果的文件夹将被保存

# PARAMETERS FOR INFERENCE
# MODEL_PATH_INFERENCE = r'F:/0_MyProject/semantic_segmentation_dinov3/weights/model.pth'               # 推理模型位置(已训练) -- 官方源模型
MODEL_PATH_INFERENCE = r'F:\0_My_jupyter\0_Optical\results\2026-02-07_17-20-03/model_19.pth'            # 推理模型位置(已训练) -- 本人微调模型 -- 二分类
MODEL_PATH_INFERENCE = r'F:/0_My_jupyter/0_Optical/weights/Segment_3_Class_normal/model_19.pth'         # 推理模型位置(已训练) -- 本人微调模型 -- 三分类


# IMG_INFERENCE_PATH = r'G:/MyDataset/PHD/COCO/image/val2017/000000000139.jpg'                          # 推理图片位置 1
# IMG_INFERENCE_PATH = r'C:/users/mr/desktop/pipeline-cat-chonk.jpeg'                                   # 推理图片位置 2
IMG_INFERENCE_PATH = r'G:/MyDataset/PHD/1R/frame_02577.jpg'                                             # 推理图片位置 3

IMG_INFERENCE_PATH = r'c:/users/mr/desktop/map.jpg'                                                     # 推理图片位置 3


#CLASS_NAMES_PATH = r"F:/0_MyProject/semantic_segmentation_dinov3/src/class_names.txt"                  # 分类文件位置（推理）
CLASS_NAMES_PATH = r"F:\0_My_jupyter\0_Optical\src\class_names.txt"                                     # 分类文件位置（训练 + 微调推理）