import torch
from torchvision import transforms
from torchvision import datasets
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import _pil_interp
from torchvision.transforms import ToPILImage
from torch.utils import data
import torch.backends.cudnn as cudnn

from models.swin_transformer import SwinTransformer
from models.custom_image_folder import MyImageFolder

IMG_SIZE = 224
DATA_INTERPOLATION = 'bicubic'
to_img = ToPILImage()
# checkpoint_path = "/content/drive/MyDrive/Research/transformer/swin/checkpoints/nih_2linear_1node_batch32_v3(w.o.pretrain)/ckpt_epoch_150.pth"
checkpoint_path = "/mnt/sda1/datasets/sina/swin_transformer/attention_validation/model/ckpt_epoch_150.pth"
test_csv_path = '/mnt/sda1/datasets/sina/transformer/csv/test_without_nofinding.csv'
TESTSET = '/mnt/sda1/project/nih-preprocess/Dataset/test2_resized256/'
INTERPOLATION = 'bicubic'

CLASS_NUM = 0
IMG_PATH = ''

def init_model():
    cudnn.benchmark = True
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
    model.to('cuda')
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

def init_image():
    transform = build_transform_test()
    dataset = datasets.ImageFolder(root=IMG_PATH, transform=transform)
    dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)
    img, _ = next(iter(dataloader))
    return img

def main():
    model = init_model()
    model.eval()
    image = init_image()
    outs, _ = model(image)
    outs[CLASS_NUM, 1].backward()
    gradients = model.get_activations_gradient()
    print(gradients.shape)

if __name__ == '__main__':
    main()
