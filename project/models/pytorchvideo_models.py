# %%
from torchinfo import summary
import string
from pytorchvideo.models import x3d, resnet, csn, slowfast, r2plus1d
import torch
import torch.nn as nn
import pytorchvideo
import torch.nn.functional as F

from pytorch_lightning import LightningModule
import pytorch_lightning
import os

from utils.metrics import get_Accuracy

# %%
class MakeVideoModule(nn.Module):
    '''
    the module zoo from the PytorchVideo lib.

    Args:
        nn (_type_): 
    '''
    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num=hparams.model_class_num
        self.model_depth=hparams.model_depth

    def make_walk_csn(self):
        return csn.create_csn(
            input_channel=3,
            model_depth=self.model_depth,
            model_num_class=self.model_class_num,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
        )

    def make_walk_resnet(self):
        return resnet.create_resnet(
        input_channel=3,
        model_depth=self.model_depth,
        model_num_class=self.model_class_num,
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
        )

    def make_walk_x3d(self) -> nn.Module:
        return x3d.create_x3d(
            input_channel=3,
            model_num_class=self.model_class_num,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
            input_clip_length=4,
        )

    #todo question with input tensor
    def make_walk_slowfast(self) -> nn.Module:
        return slowfast.create_slowfast(

        )
    
    # todo question with input tensor
    def make_walk_r2plus1d(self) -> nn.Module:
        return r2plus1d.create_r2plus1d(
            input_channel=3,
            model_num_class=self.model_class_num,
            model_depth=self.model_depth,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
        )
# %%

class WalkVideoClassificationLightningModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.model_type=hparams.model

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()
        self.lr=hparams.lr

        self.model = MakeVideoModule(hparams)

        if self.model_type == 'resnet':
            self.model=self.model.make_walk_resnet()

        elif self.model_type == 'csn':
            self.model=self.model.make_walk_csn()

        elif self.model_type == 'x3d':
            self.model = self.model.make_walk_x3d()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        y_hat=self.model(batch["video"])

        loss=F.cross_entropy(y_hat, batch["label"])

        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat=self.model(batch["video"])
        loss=F.cross_entropy(y_hat, batch["label"])
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        self.log("pred_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        real_name = []
        real_label = []

        real_name.append(batch["video_name"])
        real_label.append(batch["label"])


        pred = self.model(batch["video"])

        test_loss = F.cross_entropy(pred, batch["label"])

        # calculate acc 
        labels_hat = torch.argmax(pred, dim=1)
        test_acc = torch.sum(labels_hat == batch["label"]).item() / (len(batch["label"]) * 1.0)

        accuracy = get_Accuracy()
        test_acc_metrice = accuracy(labels_hat.cuda(), batch["label"].cuda())
        # log the output 
        self.log_dict({'test_loss': test_loss, 'test_acc': test_acc, 'test_acc_metrice': test_acc_metrice})
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _get_name(self):
        return self.model_type


# %%

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    classification_module=WalkVideoClassificationLightningModule(model_type='resnet')

    batch_size=16
    summary=summary(classification_module, input_size=(batch_size, 3, 8, 255, 255))
    print(summary)
    print(classification_module._get_name())


# %%
