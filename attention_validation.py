import torch
import torch.nn.functional as F
import math
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import _pil_interp
from torchvision.transforms import ToPILImage
import numpy as np
from timm.utils import accuracy, AverageMeter
import time
from sklearn.metrics import roc_auc_score
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
import random

from models.swin_transformer import SwinTransformer
from models.custom_image_folder import MyImageFolder

IMG_SIZE = 224
DATA_INTERPOLATION = 'bicubic'
to_img = ToPILImage()
# checkpoint_path = "/content/drive/MyDrive/Research/transformer/swin/checkpoints/nih_2linear_1node_batch32_v3(w.o.pretrain)/ckpt_epoch_150.pth"
checkpoint_path = "/mnt/sda1/datasets/sina/swin_transformer/attention_validation/model/ckpt_epoch_150.pth"
test_csv_path = '/mnt/sda1/datasets/sina/transformer/csv/test_without_nofinding.csv'
TESTSET = '/mnt/sda1/project/nih-preprocess/Dataset/all_resized256/'
PRINT_FREQ = 10
INTERPOLATION = 'bicubic'
# validation
NUM_OPERATIONS = 60
NUM_PATCH_NOISING = 5

BATCH_SIZE = 8
NUM_WORKERS = 2
PIN_MEMORY = True


def init_model():
    model = SwinTransformer(img_size=224,
                            patch_size=4,
                            in_chans=3,
                            num_classes=14,
                            embed_dim=192,
                            depths=[2, 2, 18, 2],
                            num_heads=[6, 12, 24, 48],
                            window_size=7)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    return model


def build_transform_test():
    t = []
    size = 256
    t.append(
        transforms.Resize(size, interpolation=_pil_interp(INTERPOLATION)),
        # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(224))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_dataset_test():
    transform = build_transform_test()
    testset = MyImageFolder(root=TESTSET, csv_path=test_csv_path, transform=transform)
    return testset


def init_dataloader():
    dataset_test = build_dataset_test()

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )
    return data_loader_test


@torch.no_grad()
def validate(data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = [AverageMeter() for _ in range(14)]
    loss_meter = [AverageMeter() for _ in range(14)]
    acc1_meter = [AverageMeter() for _ in range(14)]
    acc5_meter = [AverageMeter() for _ in range(14)]

    acc1s = []
    acc5s = []
    losses = []
    aucs = []

    end = time.time()
    all_preds = [[] for _ in range(14)]
    all_label = [[] for _ in range(14)]
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        for i in range(len(target)):
            target[i] = target[i].cuda(non_blocking=True)

        # compute output
        output, layers_all_attn_weights = model(images)

        for i in range(len(target)):
            # measure accuracy and record loss
            loss = criterion(output[i], target[i])
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1 = accuracy(output[i], target[i], topk=(1,))
            acc1 = torch.Tensor(acc1).to(device='cuda')   # wtf? without this added line it get's error in reduce_tensor because it's a list. So the original code shouldn't work too!?
            acc1 = reduce_tensor(acc1)
            # acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

            loss_meter[i].update(loss.item(), target[i].size(0))
            acc1_meter[i].update(acc1.item(), target[i].size(0))
            # acc5_meter.update(acc5.item(), target.size(0))

            # auc
            preds = F.softmax(output[i], dim=1)
            if len(all_preds[i]) == 0:
                all_preds[i].append(preds.detach().cpu().numpy())
                all_label[i].append(target[i].detach().cpu().numpy())
            else:
                all_preds[i][0] = np.append(
                    all_preds[i][0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[i][0] = np.append(
                    all_label[i][0], target[i].detach().cpu().numpy(), axis=0
                )

            # measure elapsed time
            batch_time[i].update(time.time() - end)
            end = time.time()

            if idx % PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                print(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time[i].val:.3f} ({batch_time[i].avg:.3f})\t'
                    f'Loss {loss_meter[i].val:.4f} ({loss_meter[i].avg:.4f})\t'
                    f'Acc@1 {acc1_meter[i].val:.3f} ({acc1_meter[i].avg:.3f})\t'
                    f'Acc@5 {acc5_meter[i].val:.3f} ({acc5_meter[i].avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB\t'
                    f'Class {i}')

    for i in range(14):
        print(f' * Acc@1 {acc1_meter[i].avg:.3f} Acc@5 {acc5_meter[i].avg:.3f}')

        # auc
        all_preds[i], all_label[i] = all_preds[i][0], all_label[i][0]
        auc = roc_auc_score(all_label[i], all_preds[i][:, 1], multi_class='ovr')
        print("Valid AUC: %2.5f" % auc)

        acc1s.append(acc1_meter[i].avg)
        acc5s.append(acc5_meter[i].avg)
        losses.append(loss_meter[i].avg)
        aucs.append(auc)

    from statistics import mean
    print("MEAN AUC: %2.5f" % mean(aucs))

    return mean(acc1s), mean(acc5s), mean(losses)


@torch.no_grad()
def get_final_attention(layers_all_attn_weights, window_size=7, resolution=224):
    xs = []
    for all_attn_weights in layers_all_attn_weights:
        for attn_weights in all_attn_weights:
            b = attn_weights.shape[0]
            x = torch.mean(attn_weights, dim=1)
            x = torch.mean(x, dim=1)
            bnw = x.shape[0]
            sqnw = int(math.sqrt(bnw / b))
            x = x.reshape((b, sqnw, sqnw, window_size, window_size))  # todo needed?
            x = x.transpose(2, 3)  # todo needed?
            x = x.reshape((b, sqnw * window_size, sqnw * window_size))
            x = x.unsqueeze(0)
            x = F.interpolate(x, size=(resolution, resolution))
            x = x.squeeze(0)
            # x = x + 0.05
            mx = torch.max(x.view(b, -1), dim=1)[0]
            scale = 1 / mx
            for i in range(b):
                x[i] = x[i] * scale[i]
            xs.append(x)
    finalx = xs[0]
    for i in range(1, len(xs)):
        for j in range(finalx.shape[0]):
            finalx[j] *= xs[i][j]
    mx = torch.max(finalx.view(finalx.shape[0], -1), dim=1)[0]
    scale = 1 / mx
    mask = finalx
    for i in range(finalx.shape[0]):
        mask[i] = finalx[i] * scale[i]
    return mask


def validate_attention(data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = [AverageMeter() for _ in range(14)]
    loss_meter = [AverageMeter() for _ in range(14)]
    acc1_meter = [AverageMeter() for _ in range(14)]
    acc5_meter = [AverageMeter() for _ in range(14)]

    end = time.time()
    all_preds = [[[] for _ in range(14)] for __ in range(NUM_OPERATIONS)]
    all_label = [[[] for _ in range(14)] for __ in range(NUM_OPERATIONS)]
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)     # batch * 3 * 224 * 224
        for i in range(len(target)):
            target[i] = target[i].cuda(non_blocking=True)
        b = images.shape[0]

        # compute output
        output, layers_all_attn_weights = model(images)
        
        # get final attention
        final_attns = get_final_attention(layers_all_attn_weights)

        # sorting attentions of pictures
        sorted_attns = []       # batch * (224/4 * 224/4)
        for attn in final_attns:
            sorted_attn = []
            for i in range(0, 224, 4):
                for j in range(0, 224, 4):
                    sorted_attn.append((attn[i][j].item(), (i, j)))
            sorted_attns.append(sorted(sorted_attn))

        # noising images in most attentioned patches and calculating aucs
        curr_noising = 0
        for operation in range(NUM_OPERATIONS):
            for patch_noising in range(NUM_PATCH_NOISING):
                for img_num in range(b):
                    noise_place_x, noise_place_y = sorted_attns[img_num][curr_noising][1]
                    noise_r = random.uniform(0, 1) #todo ok?
                    noise_g = random.uniform(0, 1) #todo ok?
                    noise_b = random.uniform(0, 1) #todo ok?
                    for i in range(4):
                        for j in range(4):
                            images[img_num][0][noise_place_x+i][noise_place_y+j] = noise_r
                            images[img_num][1][noise_place_x+i][noise_place_y+j] = noise_g
                            images[img_num][2][noise_place_x+i][noise_place_y+j] = noise_b
                curr_noising += 1

            # calculate auc
            output, layers_all_attn_weights = model(images)
            for i in range(len(target)):
                # measure accuracy and record loss
                loss = criterion(output[i], target[i])
                # acc1, acc5 = accuracy(output, target, topk=(1, 5))
                acc1 = accuracy(output[i], target[i], topk=(1,))
                acc1 = torch.Tensor(acc1).to(
                    device='cuda')  # wtf? without this added line it get's error in reduce_tensor because it's a list. So the original code shouldn't work too!?
                acc1 = reduce_tensor(acc1)
                # acc5 = reduce_tensor(acc5)
                loss = reduce_tensor(loss)

                loss_meter[i].update(loss.item(), target[i].size(0))
                acc1_meter[i].update(acc1.item(), target[i].size(0))
                # acc5_meter.update(acc5.item(), target.size(0))

                # auc
                preds = F.softmax(output[i], dim=1)
                if len(all_preds[operation][i]) == 0:
                    all_preds[operation][i].append(preds.detach().cpu().numpy())
                    all_label[operation][i].append(target[i].detach().cpu().numpy())
                else:
                    all_preds[operation][i][0] = np.append(
                        all_preds[operation][i][0], preds.detach().cpu().numpy(), axis=0
                    )
                    all_label[operation][i][0] = np.append(
                        all_label[operation][i][0], target[i].detach().cpu().numpy(), axis=0
                    )

                # measure elapsed time
                batch_time[i].update(time.time() - end)
                end = time.time()

        if idx % PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            print(
                f'Test: [{idx}/{len(data_loader)}]\t'
                # f'Time {batch_time[i].val:.3f} ({batch_time[i].avg:.3f})\t'
                # f'Loss {loss_meter[i].val:.4f} ({loss_meter[i].avg:.4f})\t'
                # f'Acc@1 {acc1_meter[i].val:.3f} ({acc1_meter[i].avg:.3f})\t'
                # f'Acc@5 {acc5_meter[i].val:.3f} ({acc5_meter[i].avg:.3f})\t'
                f'Mem {memory_used:.0f}MB\t'
                # f'Class {i}'
            )



    for operation in range(NUM_OPERATIONS):
        aucs = []
        for i in range(14):
            # print(f' * Acc@1 {acc1_meter[i].avg:.3f} Acc@5 {acc5_meter[i].avg:.3f}')

            # auc
            all_preds[operation][i], all_label[operation][i] = all_preds[operation][i][0], all_label[operation][i][0]
            auc = roc_auc_score(all_label[operation][i], all_preds[operation][i][:, 1], multi_class='ovr')
            print(f'Test AUC: {auc:2.5f}    Class: {i}  Noisy_patches: {(operation+1) * NUM_PATCH_NOISING}')

            # acc1s.append(acc1_meter[i].avg)
            # acc5s.append(acc5_meter[i].avg)
            # losses.append(loss_meter[i].avg)
            aucs.append(auc)

        from statistics import mean
        print(f'MEAN Test AUC: {mean(aucs):2.5f}    Noisy_patches: {(operation+1) * NUM_PATCH_NOISING}')

    # return mean(acc1s), mean(acc5s), mean(losses)


def main():
    model = init_model()
    dataloader_test = init_dataloader()
    acc1, acc5, loss = validate(dataloader_test, model)
    print(f"Mean Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    validate_attention(dataloader_test, model)


if __name__ == '__main__':
    main()
