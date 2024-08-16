import openai
import pandas as pd
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the image captioning model

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def describe_image(image_path):
    """Generates a description for the given image."""
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)
        return description
    except FileNotFoundError:
        print(f"File {image_path} not found.")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

def process_images_in_folder(folder_path):
    """Processes all images in the folder and generates captions."""
    captions = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            if os.path.isfile(image_path):
                description = describe_image(image_path)
                if description:  # Check if description is not empty
                    caption = description  # Use the description as the caption
                    captions.append({'image': filename, 'caption': caption})
                else:
                    print(f"Failed to generate caption for {image_path}")
            else:
                print(f"File {image_path} is not a file.")
    
    return captions

def save_to_csv(captions, output_csv='captions_linkedin.csv'):
    """Saves the generated captions to a CSV file."""
    df = pd.DataFrame(captions)
    df.to_csv(output_csv, index=False)
    print(f"Captions saved to {output_csv}")

# Example usage:
folder_path = './linkedin' 
captions = process_images_in_folder(folder_path)
save_to_csv(captions)