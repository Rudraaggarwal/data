import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn, optim
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL

# Define constants
CSV_FILE = 'captions.csv'
IMG_DIR = 'icons/'
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

# Custom Dataset Class
class ImageCaptionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        caption = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, caption

# Data transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create dataset and dataloader
dataset = ImageCaptionDataset(csv_file=CSV_FILE, img_dir=IMG_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load models and tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# Freeze parameters of text_encoder and vae
for param in text_encoder.parameters():
    param.requires_grad = False

for param in vae.parameters():
    param.requires_grad = False

# Optimizer and loss function
optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet.to(device)
vae.to(device)
text_encoder.to(device)



for epoch in range(NUM_EPOCHS):
    for images, captions in dataloader:
        images = images.to(device)
        inputs = tokenizer(captions, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

        with torch.no_grad():
            text_embeddings = text_encoder(input_ids=inputs).last_hidden_state
            images_latents = vae.encode(images).latent_dist.sample()

        optimizer.zero_grad()
        # Assuming you want to use the last timestep
        timestep = torch.tensor([63]).to(device)  
        recon_images = unet(images_latents, timestep=timestep, encoder_hidden_states=text_embeddings).sample
        loss = criterion(recon_images, images)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item()}")
# Save the trained model
unet.save_pretrained("trained_unet")
vae.save_pretrained("trained_vae")

# Inference example
pipeline = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
)

# Generate an image
prompt = ""
images = pipeline(prompt).images
images[0].show()
