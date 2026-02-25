from flax import linen as nn
import jax

class EHRModel(nn.Module):

    input_dim: int=76
    num_classes: int=25
    feats_dim: int=128
    batch_first: bool=True
    dropout: float=0.3
    layers: int=2
    fusion: bool=False

    @nn.compact    
    def __call__(self, x, train=True, feature=False):

        # LSTM in MedFuse uses 2 layers
        lstm_layer = nn.scan(nn.OptimizedLSTMCell,
                               variable_broadcast="params",
                               split_rngs={"params": False},
                               in_axes=1, 
                               out_axes=1,
                               reverse=False)

        carry, hidden = nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0), batch_dims=(len(x),), size=256)
        (carry, hidden), x = lstm_layer()((carry, hidden), x)
        
        carry, hidden = nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0), batch_dims=(len(x),), size=256)
        (carry, hidden), x = lstm_layer()((carry, hidden), x)

        feats = nn.Dropout(rate=self.dropout, deterministic=not train)(x[:, -1])
        logits = nn.Dense(features=self.num_classes)(feats)
        
        if feature or self.fusion:
            return (logits, feats)
        else:
            return logits
        

def LSTM(input_dim=76,
         num_classes=25, 
         feats_dim=128,
         batch_first=True, 
         dropout=0.3,
         layers=2,
         fusion=False):
    
    
    return EHRModel(
        input_dim=input_dim,
        num_classes=num_classes, 
        feats_dim=feats_dim,
        batch_first=batch_first, 
        dropout=dropout,
        layers=layers,
        fusion=fusion
        )