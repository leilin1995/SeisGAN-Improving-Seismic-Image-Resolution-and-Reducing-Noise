import torch
from torch import nn
from torchvision.models.vgg import vgg16


class GeneratorLoss(nn.Module):
    def __init__(self,adversarial_index = 0.001,perception_index = 0.006):
        """

        Args:
            adversarial_index: The beta value in the paper
            perception_index: The alpha value in the paper
        """
        super().__init__()
        self.vgg = vgg16()
        self.vgg.load_state_dict(torch.load("./vgg16-397923af.pth"))
        loss_network = nn.Sequential(*list(self.vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.adversarial_index = adversarial_index
        self.perception_index = perception_index


    def forward(self,out_labels,out_images,target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        # 1 channel to 3 channel
        out_images_3channel = out_images.repeat(1,3,1,1)
        target_images_3channel = target_images.repeat(1,3,1,1)
        perception_loss = self.mse_loss(self.loss_network(out_images_3channel),
                                        self.loss_network(target_images_3channel))
        # Image Loss
        image_loss = self.mse_loss(out_images,target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + self.adversarial_index * adversarial_loss + self.perception_index *perception_loss + 2e-8 * tv_loss


# Total Variation Loss
class TVLoss(nn.Module):
    def __init__(self,tv_loss_weight = 1):
        super().__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:,:,1:,:])
        count_w = self.tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



