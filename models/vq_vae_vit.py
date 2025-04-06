import torch
import torch.nn as nn
import math

from models.vit import TransformerBlock, PositionalEmbedding, PatchEmbedding


class EncoderVIT(torch.nn.Module):
    """ Encoder for the VQ-VAE """

    def __init__(self, model_settings):
        super(EncoderVIT, self).__init__()

        self.num_hidden = model_settings["num_hidden"]
        num_residual_hidden = model_settings["num_residual_hidden"]

        self.num_channels = model_settings["num_channels"]
        self.input_shape = model_settings["input_shape"]
        self.embedding_size = model_settings["embedding_dim"]
        self.patch_size = model_settings["patch_size"]

        self.patch_embedding = PatchEmbedding(model_settings)
        self.positional_embedding = PositionalEmbedding(self.patch_embedding.num_patches, model_settings)
        
        self.transformers = nn.Sequential(*[TransformerBlock(model_settings) for i in range(model_settings["transformer_layers"])])
        self.relu = nn.ReLU()  
        self.conv_to_vq_size = nn.Conv2d(self.num_hidden, self.embedding_size, stride=1, kernel_size=1)

        
        
    def forward(self, x):
        x = self.patch_embedding(x) 
        pos_embeddings = self.positional_embedding(x)
        x = x + pos_embeddings
        x = self.transformers(x)

        x = x.reshape(x.shape[0], # Reshape to 2d
                (self.input_shape[0]//self.patch_size), 
                (self.input_shape[1]//self.patch_size), 
                self.num_hidden).movedim(-1, 1)
        x = self.relu(x)
        x = self.conv_to_vq_size(x)
        return x


class DecoderVIT(torch.nn.Module):
    """ Decoder for the VQ-VAE """

    def __init__(self, model_settings):


        super(DecoderVIT, self).__init__()

        self.num_hidden = model_settings["num_hidden"]
        self.num_patches = model_settings["num_patches"]

        self.num_channels = model_settings["num_channels"]
        self.input_shape = model_settings["input_shape"]

        self.embedding_size = model_settings["embedding_dim"]
        self.patch_size = model_settings["patch_size"]

        self.positional_embedding = PositionalEmbedding(model_settings["num_patches"], model_settings)
        
        self.conv_to_hidden_size = nn.Conv2d(self.embedding_size, self.num_hidden, stride=1, kernel_size=1)
        self.transformers = nn.Sequential(*[TransformerBlock(model_settings) for i in range(model_settings["transformer_layers"])])
        self.relu = nn.ReLU()  

        self.decoderHead = nn.Sequential(
            nn.ConvTranspose2d(self.num_hidden, self.num_channels, kernel_size=self.patch_size, stride=self.patch_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_to_hidden_size(x)
        x = x.movedim(1, -1).reshape(x.shape[0], self.num_patches, self.num_hidden) # 2d to 1d

        pos_embedding = self.positional_embedding(x)
        x = x + pos_embedding

        x = self.transformers(x)
        x = self.relu(x)

        x = x.reshape(x.shape[0], # Reshape to 2d
                (self.input_shape[0]//self.patch_size), 
                (self.input_shape[1]//self.patch_size), 
                self.num_hidden).movedim(-1, 1)
        
        x = self.decoderHead(x)
        return x