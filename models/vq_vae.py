import torch
import torch.nn as nn
import math
from models.vq_vae_vit import EncoderVIT
from models.vq_vae_vit import DecoderVIT


class VQVAE(torch.nn.Module):
    """ 
    Implementation of a vector quantized variational autoencoder following the original paper by A. van den Oord et al.
    """
    def __init__(self, model_settings):
        super(VQVAE, self).__init__()
        self.model_settings = model_settings
        
        if model_settings["encoder_architecture"] == "VIT":
            self.encoder = EncoderVIT(model_settings)
            model_settings["num_patches"] = self.encoder.patch_embedding.num_patches
        elif model_settings["encoder_architecture"] == "CNN":
            self.encoder = Encoder(model_settings)
        else:
            print("Error: unsupported encoder")
            exit()

        self.VQ = VectorQuantisizer(model_settings)

        if model_settings["decoder_architecture"] == "VIT":
            self.decoder = DecoderVIT(model_settings)
        elif model_settings["decoder_architecture"] == "CNN":
            self.decoder = Decoder(model_settings)
        else:
            print("Error: unsupported decoder")
            exit()

    def forward(self, x):
        x = self.encoder(x)
        quantized, vq_loss, discrete_embedding = self.VQ(x)
        x = self.decoder(quantized)
        return x, vq_loss


class Encoder(torch.nn.Module):
    """ Encoder for the VQ-VAE """

    def __init__(self, model_settings):
        super(Encoder, self).__init__()
        num_hidden = model_settings["num_hidden"]
        num_residual_hidden = model_settings["num_residual_hidden"]
        embedding_dim = model_settings["embedding_dim"]
        num_channels = model_settings["num_channels"]

        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, num_hidden//2, stride=2, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hidden//2, num_hidden, stride=2, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hidden, num_hidden, stride=2, kernel_size=3, padding=1),

            ResidualLayer(num_hidden, num_hidden, num_residual_hidden),
            ResidualLayer(num_hidden, num_hidden, num_residual_hidden),   
            nn.ReLU(),       
            nn.Conv2d(num_hidden, embedding_dim, stride=1, kernel_size=1), # Make correct size for VQ
        )
        
        
    def forward(self, x):
        return self.layers(x)

class VectorQuantisizer(torch.nn.Module):
    """ Vector quantisizer for the VQ-VAE """

    def __init__(self, model_settings):
        super(VectorQuantisizer, self).__init__()
        self.embedding_dim = model_settings["embedding_dim"]
        self.num_embeddings = model_settings["num_embeddings"]
        self.commitment_cost = model_settings["commitment_cost"]

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        max = math.sqrt(3/self.num_embeddings) # See Tensorflowv1:UniformUnitScaling code, amounts to this.
        nn.init.uniform_(self.embeddings.weight, -max, max)

    def forward(self, x):
        x = torch.movedim(x, 1, -1) # Move channel dimension to last dim
        input = x
        x = torch.flatten(x, start_dim=0, end_dim=-2) # Flatten all but batch dimension

        # Calculate distances between input and embeddings
        distances = (
        torch.sum(x**2, dim=-1, keepdim=True)
        + torch.sum(self.embeddings.weight**2, dim=-1)  
        - 2 * torch.matmul(x, self.embeddings.weight.T) 
        )
        embedding_indexes = torch.argmin(distances, dim=-1) # Get the index of the closest embedding vector
        discrete_embedding = nn.functional.one_hot(embedding_indexes, num_classes=self.num_embeddings) # discretisize to closest embedding vector
        
        discrete_embedding = torch.reshape(discrete_embedding, input.shape[:-1] + (self.num_embeddings,)) # Reshape to original shape
        quantized = discrete_embedding.float() @ self.embeddings.weight
        # shape = (n_batch, embeddings_size)

        embeddings_loss = torch.mean((quantized - input.detach())**2) # loss is difference between encoder output and quantisized
        commitment_loss = self.commitment_cost * torch.mean((quantized.detach() - input)**2)
        vq_loss = embeddings_loss + commitment_loss
    	
        quantized = input + (quantized - input).detach() # to move gradients from decoder to encoder, skippinhg vector quantization
        
        # Change back to original shape
        quantized = torch.movedim(quantized, -1, 1)
        discrete_embedding = torch.movedim(discrete_embedding, -1, 1)
        
        # Should also calculate embedding space utilization, but not implemented here yet.

        return quantized, vq_loss, discrete_embedding

        

class Decoder(torch.nn.Module):
    """ Decoder for the VQ-VAE """

    def __init__(self, model_settings):
        embedding_dim = model_settings["embedding_dim"]
        num_hidden = model_settings["num_hidden"]
        num_residual_hidden = model_settings["num_residual_hidden"]
        channels = model_settings["num_channels"]

        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(embedding_dim, num_hidden, stride=1, kernel_size=3, padding=1),
            ResidualLayer(num_hidden, num_hidden, num_residual_hidden),
            ResidualLayer(num_hidden, num_hidden, num_residual_hidden),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hidden, num_hidden, stride=2, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hidden, num_hidden//2 ,kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_hidden//2, channels ,kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()                                
        )

    def forward(self, x):
        return self.layers(x)

class ResidualLayer(torch.nn.Module):
    """ Residual layer for the VQ-VAE """

    def __init__(self, in_channels, num_hidden, num_residual_hidden):
        super(ResidualLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, num_residual_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_residual_hidden, num_hidden, kernel_size=1),
        )

    def forward(self, x):
        return x + self.layers(x)
        
