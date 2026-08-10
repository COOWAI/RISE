from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    # pyre-fixme[6]: For 1st param expected `Tensor` but got `Union[Module, Tensor]`.
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)


class MLP(nn.Module):
    """Implements a simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """Initialize the MLP with specified input, hidden, output dimensions and number of layers."""
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        """Forward pass for the entire MLP."""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def reduce_loss(loss, reduction):
    """Reduce loss as specified

    Args:
        loss (nn.Tensor): Elementwise loss tensor.
        reduction (str): Specified reduction function chosen from "none",
            "mean" and "sum".

    Return:
        nn.Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in loss of PyTorch.
        avg_factor (float): Average factor when computing the mean of loss.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            # Avoid causing ZeroDivisionError when avg_factor is 0.0.
            eps = torch.finfo(loss.dtype).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def dice_loss(
    preds,
    targets,
    weight=None,
    eps: float = 1e-5,
    reduction: str = "mean",
    avg_factor: int = None,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        preds (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor):
            A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-5.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Default: None.

    Return:
        torch.Tensor: The computed dice loss.
    """
    preds = preds.flatten(1)
    targets = targets.flatten(1).float()
    numerator = 2 * torch.sum(preds * targets, 1)
    denominator = torch.sum(preds, 1) + torch.sum(targets, 1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if weight is not None:
        if not (weight.ndim == loss.ndim):
            raise AssertionError("assertion failed: weight.ndim == loss.ndim")
        if not (len(weight) == len(preds)):
            raise AssertionError("assertion failed: len(weight) == len(preds)")
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class DiceLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid=True,
        reduction="mean",
        loss_weight=1.0,
        eps=1e-5,
    ):
        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(
        self,
        preds,
        targets,
        weight=None,
        avg_factor=None,
    ):
        if self.use_sigmoid:
            preds = preds.sigmoid()

        loss = self.loss_weight * dice_loss(
            preds,
            targets,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            avg_factor=avg_factor,
        )
        return loss


class SimpleSemanticSegHead(nn.Module):
    def __init__(
        self,
        input_strides,
        num_classes,
        decoder=None,
        embed_dims=256,
        out_mask_dim=256,
        loss_weights={
            "loss_seg": 2.0,
            "loss_dice": 5.0,
            "loss_subcat": 0.001,
        },
        subcat_num=0,
    ):
        super().__init__()
        self.input_strides = input_strides
        self.mask_classification = True
        self.num_classes = num_classes
        self.decoder = decoder
        self.embed_dims = embed_dims
        self.subcat_num = subcat_num

        # Mask output
        self.mask_features = nn.Conv2d(
            self.embed_dims,
            out_mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        c2_xavier_fill(self.mask_features)
        self.decoder_norm = nn.LayerNorm(self.embed_dims)
        self.class_embed = nn.Linear(self.embed_dims, num_classes + subcat_num)
        self.mask_embed = MLP(self.embed_dims, self.embed_dims, out_mask_dim, 3)

        channel_weight = torch.ones(num_classes)
        channel_weight[0] = 0.1
        if "loss_subcat" in loss_weights:
            if not (subcat_num > 0):
                raise AssertionError(subcat_num)
            self.loss_subcat_weight = loss_weights.pop("loss_subcat")
            self.subcat_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
            channel_weight[0] = 0.001
        self.loss_weights = loss_weights
        self.dice_loss = DiceLoss(reduction="mean")
        self.register_buffer("channel_weight", channel_weight)
        # 可训练的上采样细化模块（固定输出为 256x256）
        upsample_size = (256, 256)
        out_ch = self.num_classes + self.subcat_num
        self.upsample_head = nn.Sequential(
            nn.Upsample(size=upsample_size, mode="bilinear", align_corners=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def pos_init_decoder(self, decoder):
        self.decoder = decoder

    def forward(self, inputs, targets=None, input_query=None, **kwargs):
        x = inputs
        self.input_shape = [s * self.input_strides[0] for s in x[0].shape[2:]]  # image size (h,w)
        # query: q,b,embed_dim
        # feature: b,embed_dim,h,w
        # hidden_states: num_layers,q,b,embed_dim, ex: torch.Size([6, 1500, 1, 256])
        hidden_states, _, query = self.decoder.forward(x, input_query=input_query)
        feature = x[0]
        mask_features = self.mask_features(feature)  # b,embed_dim,h,w

        predictions_semseg = []
        # prediction heads on learnable query features
        # outputs_class: b,q,nc
        # outputs_mask: b,q,h,w
        if self.training:
            outputs_semseg = self.forward_prediction_heads(query, mask_features)
            predictions_semseg.append(outputs_semseg)

        for i, output in enumerate(hidden_states):
            if self.training or (i == len(hidden_states) - 1):
                outputs_semseg = self.forward_prediction_heads(output, mask_features)
                predictions_semseg.append(outputs_semseg)
        return predictions_semseg

    def forward_prediction_heads(self, output, mask_features):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        ##
        # outputs_class = F.softmax(outputs_class, dim=-1)
        # outputs_mask = outputs_mask.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", outputs_class, outputs_mask)
        semseg = self.upsample_head(semseg)
        return semseg

    def prepare_targets_from_semmask(self, targets, device=None):
        gt_mask = targets["masks"].to(device).float()
        # print(f"gt_mask1: {gt_mask.shape}, {gt_mask.max()}, {gt_mask.min()}")
        if self.subcat_num > 1:
            gt_mask[..., 0][gt_mask[..., 0] == 255] = 0
            gt_mask[..., 1][gt_mask[..., 1] == 255] = -1  # 颜色做二分类
        else:
            gt_mask[gt_mask == 255] = 0
            gt_mask = gt_mask.unsqueeze(-1)
        return gt_mask

    def cal_seg_loss(self, pred_mask, gt_mask):
        loss_dict = {}
        gt_mask_cls = F.one_hot(gt_mask[:, 0].long(), self.num_classes).permute(0, 3, 1, 2).float()

        channel_weight = self.channel_weight[None].repeat(len(gt_mask_cls), 1)[..., None, None]
        loss_seg = F.binary_cross_entropy_with_logits(pred_mask.float(), gt_mask_cls, channel_weight)
        loss_dice = self.dice_loss(pred_mask.float(), gt_mask_cls)
        loss_dict["loss_seg"] = loss_seg * self.loss_weights["loss_seg"]
        loss_dict["loss_dice"] = loss_dice * self.loss_weights["loss_dice"]
        return loss_dict

    def cal_color_loss(self, pred_mask, gt_mask):
        gt_mask_color = gt_mask[:, 1].long()
        loss_color = self.subcat_loss(pred_mask.float(), gt_mask_color) / max((gt_mask_color != -1).sum(), 1)
        return {"loss_subcat": loss_color * self.loss_subcat_weight}

    def get_loss(self, inputs: Tuple[Dict], targets: Dict, input_query=None):
        predictions_semseg = self(inputs, input_query=input_query)
        if self.subcat_num > 0:
            main_semseg = [semseg[:, : -self.subcat_num] for semseg in predictions_semseg]
            subcat_semseg = [semseg[:, -self.subcat_num :] for semseg in predictions_semseg]
        else:
            main_semseg = predictions_semseg

        device = main_semseg[-1].device
        b, c, h, w = main_semseg[-1].shape
        gt_mask = self.prepare_targets_from_semmask(targets, device)
        gt_mask = gt_mask.permute(0, 3, 1, 2)
        gt_mask = F.interpolate(gt_mask, (h, w), mode="nearest")

        losses_dict = {}
        for i, semseg in enumerate(main_semseg):
            loss_dict = self.cal_seg_loss(semseg, gt_mask)
            losses_dict.update({k + f"_{i}": v for k, v in loss_dict.items()})

        if self.subcat_num > 0:
            for i, sub_semseg in enumerate(subcat_semseg):
                loss_dict = self.cal_color_loss(sub_semseg, gt_mask)
                losses_dict.update({k + f"_{i}": v for k, v in loss_dict.items()})
        return losses_dict

    def post_process(self, inputs):
        prediction_semseg = self(inputs)[-1]
        if torch.onnx.is_in_onnx_export():
            H, W = self.input_shape[0] // 2, self.input_shape[1] // 2
        else:
            H, W = self.input_shape
        prediction_semseg = F.interpolate(prediction_semseg, (H, W), mode="bilinear")
        if self.subcat_num > 0:
            mask_cat = torch.argmax(prediction_semseg[:, : -self.subcat_num].sigmoid(), dim=1)
            mask_subcat = torch.argmax(F.softmax(prediction_semseg[:, -self.subcat_num :], dim=-1), dim=1)
        else:
            mask_cat = torch.argmax(prediction_semseg.sigmoid(), dim=1)
            mask_subcat = None
        return mask_cat, mask_subcat

    def export_onnx(self, x, input_info=None):
        mask_cat, mask_subcat = self.post_process(x)
        if self.subcat_num > 0:
            subcat_valid = (mask_cat == 1) | (mask_cat == 2) | (mask_cat == 4)

            # mask_cat[subcat_valid] = (mask_subcat[subcat_valid] + 1) * 10 + mask_cat[subcat_valid]
            # mask = mask_cat
            cat_value = (mask_subcat + 1) * 10 + mask_cat
            mask = torch.where(subcat_valid, cat_value, mask_cat).float()
        else:
            mask = mask_cat.float()
        return mask

    def box_to_raw(self, warp_matrix_inv, width, height, img=None, boxes=None, masks=None, cv_border=0):
        if img is not None:
            img = cv2.warpAffine(
                img.astype(np.uint8), warp_matrix_inv[:2, :], (width, height), flags=cv2.INTER_LINEAR, borderValue=0
            )
        # Predictions
        if boxes is not None:
            pass
        if masks is not None:
            if len(masks.shape) == 3 and len(masks) == 1:
                masks = cv2.warpAffine(
                    masks[0], warp_matrix_inv[:2, :], (width, height), flags=cv2.INTER_NEAREST, borderValue=cv_border
                )
            else:
                masks = cv2.warpAffine(
                    masks, warp_matrix_inv[:2, :], (width, height), flags=cv2.INTER_NEAREST, borderValue=cv_border
                )
        return img, boxes, masks

    def batch_deal_for_semantic(self, sem_seg, input_meta=None):
        img = input_meta.get("img", None).cpu().numpy() if input_meta is not None else None
        img_id = input_meta.get("image_id", None) if input_meta is not None else None
        gt_sem_mask = input_meta.get("sem_mask", None) if input_meta is not None else None
        if input_meta is not None:
            height, width, _ = input_meta.get("img_raw_shape")
            warp_matrix = input_meta.get("warp_matrix", np.eye(3))
            if isinstance(warp_matrix, torch.Tensor):
                warp_matrix = warp_matrix.cpu().numpy()
            warp_matrix_inv = np.linalg.inv(warp_matrix.astype(np.float32))
            if img is not None:
                img = cv2.warpAffine(
                    img.astype(np.uint8),
                    warp_matrix_inv[:2, :],
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderValue=0,
                )
            if gt_sem_mask is not None:
                gt_sem_mask = gt_sem_mask.cpu().numpy()
                gt_sem_mask = cv2.warpAffine(
                    gt_sem_mask.astype(np.uint8),
                    warp_matrix_inv[:2, :],
                    (width, height),
                    flags=cv2.INTER_NEAREST,
                    borderValue=0,
                )
            sem_seg = cv2.warpAffine(
                sem_seg.astype(np.uint8),
                warp_matrix_inv[:2, :],
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )
        result = {"image_id": img_id, "img": img, "gt": gt_sem_mask, "mask": sem_seg}
        return result

    def val(self, inputs, inputs_meta=None):
        mask_cat, mask_subcat = self.post_process(inputs)
        if inputs_meta is None:
            return []

        if self.subcat_num > 0:
            mask = torch.stack([mask_cat, mask_subcat], dim=1)
            mask = mask.permute(0, 2, 3, 1)  # NCHW -> NHWC
        else:
            mask = mask_cat
        mask = mask.int().detach().cpu().numpy()

        coco_format_lists = []
        combined_params = list(zip(mask, inputs_meta))

        # 1. 不使用多线程
        for params in combined_params:
            result = self.batch_deal_for_semantic(*params)
            coco_format_lists.append(result)
        # 2. 使用多线程
        # with ThreadPoolExecutor(max_workers=batch_size) as executor:
        #     # 提交任务并传递参数
        #     futures = [executor.submit(self.batch_deal, *params) for params in combined_params]

        #     # 遍历并获取结果
        #     for future in as_completed(futures):
        #         result = future.result()
        #         coco_format_lists.append(result)
        return coco_format_lists
