from tinygrad import Tensor, nn
from typing import List, Callable
from models.cnn_attention import CNN_Attention
from models.se_resnet10_1D import SE_ResNet10_1D


class CombinedModel:
    def __init__(self, num_classes=2):
        
        self.model1 = SE_ResNet10_1D()
        self.model2 = CNN_Attention()

        self.classifier: List[Callable[[Tensor], Tensor]] = [
            nn.Linear(9824, 512),
            nn.BatchNorm(512),       
            lambda x: Tensor.relu(x),
            lambda x: x.dropout(0.2),

            nn.Linear(512, 128),
            nn.BatchNorm(128),
            lambda x: Tensor.relu(x),
            lambda x: x.dropout(0.4),

            nn.Linear(128, num_classes)
        ]

    def __call__(self, prof_64: Tensor, prof_96: Tensor, prof_128: Tensor,
                       intv_64: Tensor, intv_96: Tensor, intv_128: Tensor) -> Tensor:
        feat1 = self.model1(prof_64, prof_96, prof_128)   
        feat2 = self.model2(intv_64, intv_96, intv_128)   
        x = feat1.cat(feat2, dim=1)                       
        return x.sequential(self.classifier)