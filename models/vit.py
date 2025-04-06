import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PatchEmbedding(nn.Module):
    def __init__(self, model_settings):
        """ 
        """
        super(PatchEmbedding, self).__init__()
        self.patch_size = (model_settings["patch_size"], model_settings["patch_size"])
        self.num_channels = model_settings["num_channels"]
        self.embedding_size = model_settings["num_hidden"]
        
        self.num_patches = np.prod(model_settings["input_shape"]) // np.prod(self.patch_size)

        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size[0])
        self.linear1 = nn.Linear(np.prod(self.patch_size) * self.num_channels, self.embedding_size)

    def forward(self, x):
        #TODO: test with channels > 1
        x = self.unfold(x)          # Extract patches
        x = torch.movedim(x, 1,-1)  # Move patch pixels to last dim
        x = self.linear1(x)         # Create embeddings from patches
        
    
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, patch_embeddings, model_settings):
        super(PositionalEmbedding, self).__init__()
        self.num_embeddings = patch_embeddings 
        self.embedding_dim = model_settings["num_hidden"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim, device=self.device)

    def forward(self, x):
        # We add one extra embeddings, not used for position, but as the class embedding
        positional_ints = torch.arange(0, self.num_embeddings, requires_grad=False, device=self.device
                                       ).repeat(x.shape[0], 1)
        embedding = self.embedding(positional_ints)
        return embedding


class TransformerBlock(nn.Module):
    def __init__(self, model_settings, dropout=0.1):
        """ 
        """
        super(TransformerBlock, self).__init__()
        self.embedding_dim = model_settings["num_hidden"]
        self.heads = model_settings["attention_heads"]
        hidden_dim = self.embedding_dim * 4

        self.layer_norm_1 = nn.LayerNorm(self.embedding_dim)
        self.attention = nn.MultiheadAttention(self.embedding_dim, self.heads,
                                          dropout=dropout, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(self.embedding_dim)

        self.linear = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.embedding_dim),
            nn.Dropout(dropout))

    def forward(self, x_in):
        x = self.layer_norm_1(x_in)
        x = x_in + self.attention(x, x, x)[0]
        x = x + self.linear(self.layer_norm_2(x))

        return x   


class VIT(nn.Module):
    """ """

    def __init__(self, model_settings):
        """ 
        """
        super(VIT, self).__init__()
        self.num_channels = model_settings["num_channels"]
        self.input_shape = model_settings["input_shape"]
        self.embedding_size = model_settings["embedding_dim"]
        self.device = model_settings["device"]

        self.patch_embedding = PatchEmbedding(model_settings)
        self.positional_embedding = PositionalEmbedding(self.patch_embedding.num_patches, model_settings)

        self.transformers = nn.Sequential(*[TransformerBlock(model_settings) for i in range(model_settings["transformer_layers"])])

        self.dropout = nn.Dropout(0.1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embedding_size),
            nn.Linear(self.embedding_size, 10)
        )



    def forward(self, x):

        x = self.patch_embedding(x) 
        pos_embeddings = self.positional_embedding(x) # Also includes extra embeddings for class tokens
        x = x + pos_embeddings

        x = self.dropout(x)

        x = self.transformers(x)
        classification_tokens = x[:,0,:]
        x = self.dropout(x)
        x = self.mlp_head(classification_tokens)

        return x