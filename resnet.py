#!/usr/bin/env python
# coding: utf-8

# In[5]:


from IPython.display import clear_output

get_ipython().system('pip install torchnet')
get_ipython().system('pip install fire')
get_ipython().system('pip install ml-callbacks==0.15.0')
get_ipython().system("nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9")
get_ipython().system('nvidia-smi')

try:
    import seaborn
except:
    get_ipython().system('pip install seaborn')
    
clear_output()


# In[6]:


from ml_callbacks import callback
from ml_callbacks import pytorch_enviroment
from ml_callbacks import pytorch_training


# In[7]:


import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline
plt.ion()   # interactive mode

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader


# In[8]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[12]:


#! ls /data/clothing-size-identification/imgs

dataFile = "/data/clothing-size-identification/dataset.csv"


# In[ ]:


import pandas as pd
cer_df = pd.read_csv(dataFile)


# In[10]:





# # Model

# In[6]:


import torch as t
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

__all__ = ['resnet101']


model_urls = {
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 2,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# In[7]:


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


# In[8]:


notebook_script = ""
get_ipython().system('jupyter nbconvert --to python /data/resnet.ipynb')
with open('/data/resnet.py') as f:
    notebook_script = f.read()


# # Data Loaders

# In[9]:


from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MURA_MEAN = [0.22588661454502146] * 3
MURA_STD = [0.17956269377916526] * 3

class MURA_Dataset(object):

    def __init__(self, root, csv_path, part='all', transforms=None, train=True, test=False):
        with open(csv_path, 'rb') as F:
            d = F.readlines()
            if part == 'all':
                imgs = [root + str(x, encoding='utf-8').strip() for x in d]
            else:
                imgs = [root + str(x, encoding='utf-8').strip() for x in d if
                        str(x, encoding='utf-8').strip().split('/')[2] == part]

        self.imgs = imgs
        print(len(imgs))
        self.train = train
        self.test = test

        if transforms is None:
            if self.train and not self.test:
                self.transforms = T.Compose([
                    T.Resize(320),
                    T.RandomCrop(320),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomRotation(30),
                    T.ToTensor(),
                    T.Lambda(lambda x: t.cat([x[0].unsqueeze(0), x[0].unsqueeze(0), x[0].unsqueeze(0)], 0)),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])
            if not self.train:
                self.transforms = T.Compose([
                    T.Resize(320),
                    T.CenterCrop(320),
                    T.ToTensor(),
                    T.Lambda(lambda x: t.cat([x[0].unsqueeze(0), x[0].unsqueeze(0), x[0].unsqueeze(0)], 0)),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])

    def __getitem__(self, index):
        img_path = self.imgs[index]

        data = Image.open(img_path)

        data = self.transforms(data)

        # label
        if not self.test:
            label_str = img_path.split('_')[-1].split('/')[0]
            if label_str == 'positive':
                label = 1
            elif label_str == 'negative':
                label = 0
            else:
                print(img_path)
                print(label_str)
                raise IndexError

        if self.test:
            label = 0

        # body part
        body_part = img_path.split('/')[6]
        
        label = -1
        if body_part == 'XR_ELBOW':
            label = 0
        if body_part == 'XR_FINGER':
            label = 1
        if body_part == 'XR_FOREARM':
            label = 2
        if body_part == 'XR_HAND':
            label = 3
        if body_part == 'XR_HUMERUS':
            label = 4
        if body_part == 'XR_SHOULDER':
            label = 5
        if body_part == 'XR_WRIST':
            label = 6

        return data, label, img_path, label

    def __len__(self):
        return len(self.imgs)


# In[10]:


data_root = '/data/datasets/Mura/'
train_image_paths = data_root + 'MURA-v1.1/train_image_paths.csv'
test_image_paths = data_root + 'MURA-v1.1/valid_image_paths.csv'
num_workers = 0
batch_size = 6

train_data = MURA_Dataset(data_root, train_image_paths, train=True, test=False)
val_data = MURA_Dataset(data_root, test_image_paths, train=False, test=False)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# # Input Formater

# In[11]:


class InputFormater:
    def format_data(self, data):
        return Variable(data).cuda()
    
    def format_label(self, label):
        return Variable(label).cuda()


# # Loss Function

# In[12]:


class CustomLossFunction:
    _criterion = None
    _loss = None
    def __init__(self):
        A = 21935
        N = 14873
        weight = t.Tensor([A / (A + N), N / (A + N)])
        weight = weight.cuda()
        
        self._criterion = t.nn.CrossEntropyLoss(weight=weight)
    
    def loss(self, preds, labels):
        self._loss = self._criterion(preds, labels)
        return self._loss
    
    def get_loss(self):
        return self._loss.data.cpu().numpy()


# # Accuracy Function

# In[13]:


from torchnet import meter

class AccuracyFunction:
    _meter = None
    _s = None
    
    def __init__(self):
        self._meter = meter.ConfusionMeter(2)
        self._s = t.nn.Softmax()
    
    def get_acc(self, preds, labels):
        self._meter.add(self._s(Variable(preds.data)).data, labels.data)
        
        cm_value = self._meter.value()
        acc = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
        return acc


# # Model config

# In[14]:


model = resnet101()
model.cuda()


# # Optimizer

# In[15]:


class OptimizerFunction:
    _optimizer = None
    
    _weight_decay = 1e-5
    _lr = 0.0001
    _lr_decay = 0.5
    
    def create(self, model):
        self._optimizer = t.optim.Adam(model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        return self._optimizer
    
    def get_scheduler(self):
        return self._optimizer


# # Sheduler

# In[16]:


class SchedulerFuntion:
    _scheduler = None
    
    _mode = 'min'
    _patience = 1
    _verbose = True
    
    def create(self, optimizer):
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                    mode=self._mode, 
                    patience=self._patience, 
                    verbose=self._verbose)
        
    def get_scheduler(self):
        return self._scheduler


# # Main Config

# In[17]:


EPOCHS = 10

_callback = callback.Callback(
    "ResNet34",
    "pytorch",
    EPOCHS,
    notebook_script,
    pytorch_enviroment.PytorchCallback())


# In[18]:


import gc
import time
import numpy as np
import torch
from tqdm import tqdm

# Logs - Helpful for plotting after training finishes
train_logs = {"loss" : [], "accuracy" : [], "time" : []}
val_logs = {"loss" : [], "accuracy" : [], "time" : []}

class PyTorchTraining2:
    def train_one_epoch(
            self,
            model, 
            device, 
            loader, 
            optimizer,
            input_formater,
            loss_func, 
            acc_func,
            callback
        ):
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # training begin event
        callback.on_train_begin()
        
        ### Local Parameters
        epoch_loss = []
        epoch_acc = []
        start_time = time.time()

        ### Iterating over data loader
        for ii, (data, label, _, body_part) in tqdm(enumerate(loader)):
            # On bach begin
            callback.on_train_batch_begin()
            
            input = input_formater.format_data(data)
            target = input_formater.format_label(label)
#             body_part = Variable(body_part).cuda()
            
            # Reseting Gradients
            optimizer.zero_grad()
            
            # Forward
            preds = model(input)
            print('A')
            
            callback.on_train_loss_begin()
            
            # Calculating Loss
            loss = loss_func.loss(preds, target)
            epoch_loss.append(loss_func.get_loss())
            
            # Calculating Accuracy
            epoch_acc.append(acc_func.get_acc(preds, target))
            
            # Backward
            loss.backward()
            
            callback.on_train_loss_end()
            
            # optimizer step begin event
            callback.on_step_begin()
            
            optimizer.step()
            
            # optimizer step end event
            callback.on_step_end()
            
            # On bach end
            callback.on_train_batch_end()
        
        ### Overall Epoch Results
        end_time = time.time()
        total_time = end_time - start_time
        
        ### Acc and Loss
        epoch_loss = np.mean(epoch_loss)
        epoch_acc = np.mean(epoch_acc)

        ### Storing results to logs
        train_logs["loss"].append(epoch_loss)
        train_logs["accuracy"].append(epoch_acc)
        train_logs["time"].append(total_time)
        
        # training begin event
        callback.on_train_end()
            
        return epoch_loss, epoch_acc, total_time
    
    def val_one_epoch(
            self,
            model,
            device,
            loader,
            input_formater,
            loss_func, 
            acc_func, 
            callback
        ):
        # validation begin event
        callback.on_val_begin()                
        
        ### Local Parameters
        epoch_loss = []
        epoch_acc = []
        start_time = time.time()
        
        ###Iterating over data loader
        for ii, (data, label, _, body_part) in tqdm(enumerate(loader)):
            # On bach start
            callback.on_val_batch_begin()
            
            #Loading images and labels to device
            input = input_formater.format_data(data)
            target = input_formater.format_label(label)
#             body_part = Variable(body_part).cuda()
            
            #Forward
            preds = model(input)
            
            callback.on_val_loss_begin()
            
            #Calculating Loss
            loss = loss_func.loss(preds, target)
            epoch_loss.append(loss_func.get_loss())
            
            #Calculating Accuracy
            epoch_acc.append(acc_func.get_acc(preds, target))
            
            callback.on_val_loss_end()
            
            # On bach end
            callback.on_val_batch_end()
        
        ###Overall Epoch Results
        end_time = time.time()
        total_time = end_time - start_time
        
        ###Acc and Loss
        epoch_loss = np.mean(epoch_loss)
        epoch_acc = np.mean(epoch_acc)
        
        ###Storing results to logs
        val_logs["loss"].append(epoch_loss)
        val_logs["accuracy"].append(epoch_acc)
        val_logs["time"].append(total_time)
            
        # epoch end event
        callback.on_val_end()
            
        return epoch_loss, epoch_acc, total_time

    def train_model(
            self,
            model,
            device,
            epochs,
            train_data_loader,
            val_data_loader,
            optimizer_creator,
            scheduler_creator,
            input_formater,
            train_loss_func,
            val_loss_func,
            train_acc_func,
            val_acc_func,
            callback
        ):
        try:
            # On start
            callback.on_start()

            # Loading model to device
            model.to(device)

            for epoch in range(epochs):
                callback.on_epoch_begin()

                model.train()
                optimizer = optimizer_creator.create(model)
                
                # Training
                train_loss, train_acc, train_time = self.train_one_epoch(
                    model=model,
                    device=device,
                    loader=train_data_loader,
                    optimizer=optimizer,
                    input_formater=input_formater,
                    loss_func=train_loss_func,
                    acc_func=train_acc_func,
                    callback=callback
                )

                # Print Epoch Details
                print("\nTraining")
                print("Epoch {}".format(epoch+1))
                print("Loss : {}".format(round(train_loss, 4)))
                print("Acc : {}".format(round(train_acc, 4)))
                print("Time : {}".format(round(train_time, 4)))

                gc.collect()
                torch.cuda.empty_cache()

                # Validation
                print("\nValidating")
                with torch.no_grad():
                    model.eval()
                    val_loss, val_acc, val_time, = self.val_one_epoch(
                        model=model,
                        device=device,
                        loader=val_data_loader,
                        input_formater=input_formater,
                        loss_func=val_loss_func,
                        acc_func=val_acc_func,
                        callback=callback
                    )
                    #Print Epoch Details
                    print("Epoch {}".format(epoch+1))
                    print("Loss : {}".format(round(val_loss, 4)))
                    print("Acc : {}".format(round(val_acc, 4)))
                    print("Time : {}".format(round(val_time, 4)))
                
                # On Epoch End
                callback.on_epoch_end(train_acc, train_loss, train_time, val_acc, val_loss, val_time)
                
                # On Model Saving
                callback.on_model_saving(model, val_acc)

            # On end
            callback.on_end()
        except Exception as e:
            # On Failed
            callback.failed(str(e))
            print(e)


# In[19]:


PyTorchTraining2().train_model(
    model=model,
    device=device,
    epochs=EPOCHS,
    train_data_loader=train_dataloader,
    val_data_loader=val_dataloader,
    optimizer_creator=OptimizerFunction(),
    scheduler_creator=SchedulerFuntion(),
    input_formater= InputFormater(),
    train_loss_func=CustomLossFunction(),
    val_loss_func=CustomLossFunction(),
    train_acc_func=AccuracyFunction(),
    val_acc_func=AccuracyFunction(),
    callback=_callback)

