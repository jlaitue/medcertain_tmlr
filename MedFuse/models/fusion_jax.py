from .lstm_jax import LSTM
from .resnet_jax import ResNet34

from flax import linen as nn
import jax
import numpy as np
import jax.numpy as jnp

class FusionModel(nn.Module):
    
    num_classes: int=25
    vision_num_classes: int=14
    labels_set: str='phenotyping'

    def setup(self):   
        # TODO clean up the arguments for each mimic_task   
        self.ehr_model = LSTM(input_dim=76, num_classes=self.num_classes, feats_dim=256, batch_first=True, dropout=0.3, layers=2, fusion=True)
        self.cxr_model = ResNet34(output='logits', pretrained='imagenet', num_classes=self.vision_num_classes, dtype='float32', fusion=True)

        # Fusion model class definition ---------------
        # self.init_fusion_method() # In the first JAX implementation we will not freeze anything

        target_classes = self.num_classes
        lstm_in = self.ehr_model.feats_dim
        lstm_out = self.cxr_model.feats_dim
        projection_in = self.cxr_model.feats_dim

        
        if self.labels_set == 'phenotyping':
            target_classes = self.vision_num_classes
            # lstm_in = self.cxr_model.feats_dim # We define it as 512 in the ResNet jax module
            lstm_in = 256 # We define it as 512 in the ResNet jax module
            projection_in = self.ehr_model.feats_dim

        # Projection layer implementation for Flax
        self.projection = nn.Dense(features=lstm_in)

        feats_dim = 2 * self.ehr_model.feats_dim 
        #---------------------------------------------------------

        # Flax implementation
        # self.fused_cls = nn.Dense(self.num_classes)
        self.fused_cls = nn.Sequential([
            # nn.Dense(feats_dim),
            nn.Dense(self.num_classes),
            # nn.sigmoid #Tim's code requires logits to be sent by the model
            ])
        # ----------------------------------------

        # Flax implementation
        self.lstm_fused_cls = nn.Sequential([
            # nn.Dense(lstm_out),
            nn.Dense(target_classes),
            # nn.sigmoid
            ])
        # --------------
        
        # Flax implementation
        lstm_layer = nn.scan(nn.OptimizedLSTMCell,
                               variable_broadcast="params",
                               split_rngs={"params": False},
                               in_axes=1, 
                               out_axes=1,
                               reverse=False)
        #IN-lstm_in OUT-lstm_out
        self.lstm_fusion_layer = lstm_layer()

        #--------------------------------------

    def __call__(self, fused_batch, train=True, feature=False):

        # We need to split the batch into EHR and Image data
        x, _, img, _, _ = fused_batch

        # Only in the Fusion mode, do these models return a tuple
        _, ehr_feats = self.ehr_model(x, train=train, feature=feature)
        _, cxr_feats = self.cxr_model(img, train=train, feature=feature)

        # projected = self.projection_layer_in(cxr_feats)
        projected = self.projection(cxr_feats)

        feats = jnp.concatenate([ehr_feats, projected], axis=1) #(batch_size, 512)

        fused_preds = self.fused_cls(feats) # (batch_size, 25)
        
        if feature:
            return (fused_preds, feats)
        else:
            return fused_preds

def Fusion(
        num_classes=25,
        vision_num_classes=14,
        labels_set='radiology'
        ):
    
    
    return FusionModel(
        num_classes=num_classes,
        vision_num_classes=vision_num_classes,
        labels_set=labels_set
        )