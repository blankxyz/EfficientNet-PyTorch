from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import torch
from PIL import Image
# Defining the Dataset
# 数据集应该继承torch.utils.data.Dataset ，并实现 __len__ 和 __getitem__方法
from torch.utils.data import Dataset


class PennFudanDataSet(Dataset):
    """

    :param Dataset:
    :return:
        image: a PIL Image of size(H, W)
        target: a dict containing the following fields
            boxes (FloatTensor[N, 4]):
                边界框，每个框包含[x0,y0,x1,y1]的格式的数据
                the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
            labels (Int64Tensor[N]): the label for each bounding box
                每个框的标签
            image_id (Int64Tensor[1]): an image identifier.
                It should be unique between all the images in the dataset, and is used during evaluation
                图片Id 图片在数据中的唯一值
            area (Tensor[N]): The area of the bounding box.
                在使用COCO度量进行评估时使用，区分大小框之间的度量分数。（是什么不清楚//TODO)
                This is used during evaluation with the COCO metric,
                to separate the metric scores between small, medium and large boxes.
            iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
                如果实例的iscrowd是True的话， 在评估的时候会被忽略
            (optionally) masks (UInt8Tensor[N, H, W]): The segmentation masks for each one of the objects
                每个对象的切分标记， 格式是[N, H, W]
            (optionally) keypoints (FloatTensor[N, K, 3]):
                对于N个对象中的每一个，它都包含[x，y，visibility]格式的K个关键点，定义对象。
                可见性=0表示关键点不可见。请注意，对于数据增强，翻转关键点的概念取决于数据表示形式，
                您可能应该为新的关键点表示形式调整references/detection/transforms.py
                For each one of the N objects, it contains the K keypoints in [x, y, visibility] format,
                defining the object. visibility=0 means that the keypoint is not visible.
                Note that for data augmentation, the notion of flipping a keypoint is dependent on the data
                representation, and you should probably adapt references/detection/transforms.py
                for your new keypoint representation
    """

    def __init__(self, root, transform):
        self.root = root
        self.transforms = transform

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'PNGImages'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'PedMasks'))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        # 读取图片
        img = Image.open(img_path).convert("RGB")
        # mask不需要转换为rgb
        mask = Image.open(mask_path)
        # 转换pil图像为numpy 数组
        mask = np.array(mask)
        # 颜色编码实例(np.unique该函数是去除数组中的重复数字，并进行排序之后输出。)
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks(将颜色编码的掩码拆分为一组二进制掩码)
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

            # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# ====

# Defining your mode

# ====

# 有两种常见情况下，可能需要修改torchvision modelzoo中的一个可用模型。

# 1.  第一个是当我们想从一个预先训练好的模型开始，然后对最后一层进行微调。

# 2.  另一种是当我们想用另一个模型替换模型的主干时（例如，为了更快的预测）

def fintuing_model():
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    # load a model pre-trained pre-trained on COCO
    # 读取一个与训练模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    # 获取分类器的输入参数的数量
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    # 用新的头部替换预先训练好的头部
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def modify_backbone():
    "修改模型以添加其他主干"
    import torchvision
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator

    # load a pre-trained model for classification and return
    # only the features
    # # FasterRCNN需要知道骨干网中的输出通道数量。对于mobilenet_v2，它是1280，所以我们需要在这里添加它
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # 我们让RPN在每个空间位置生成5 x 3个锚点
    # 具有5种不同的大小和3种不同的宽高比。
    # 我们有一个元组[元组[int]]
    # 因为每个特征映射可能具有不同的大小和宽高比
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    #  定义一下我们将用于执行感兴趣区域裁剪的特征映射，以及重新缩放后裁剪的大小。
    #  如果您的主干返回Tensor，则featmap_names应为[0]。
    #  更一般地，主干应该返回OrderedDict [Tensor]
    #  并且在featmap_names中，您可以选择要使用的功能映射。
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


import transforms as T
import utils


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


from engine import train_one_epoch, evaluate


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataSet('myscripts/torch_learning/data/PennFudanPed',
                               get_transform(train=True))
    dataset_test = PennFudanDataSet('myscripts/torch_learning/data/PennFudanPed',
                                    get_transform(train=False))

    # for x in enumerate(dataset):
    #    print(x)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")


if __name__ == '__main__':
    main()
