import os
from PIL import Image
from IPython.display import Image as IPImage, display

# Define paths
colorised_path = "./dataset/colorised/"
greyscale_path = "./dataset/greyscale/"

# Ensure the output directory exists
os.makedirs(greyscale_path, exist_ok=True)

# Function to convert an image to greyscale
def convert_to_grayscale(input_image_path, output_image_path):
    try:
        # Open the input image
        image = Image.open(input_image_path)
        
        # Convert the image to grayscale using the Luminance method
        grayscale_image = image.convert('L')
        
        # Save the grayscale image
        grayscale_image.save(output_image_path)
        print(f"Successfully converted {input_image_path} to greyscale and saved to {output_image_path}")
    except Exception as e:
        print(f"An error occurred while processing {input_image_path}: {e}")

# Process images iteratively from 1 to 200
for i in range(1, 233):  # Adjust range as needed
    file_name = f"ROIs1868_summer_s2_59_p{i}.png"  # Adjust file name pattern if necessary
    image_path = os.path.join(colorised_path, file_name)
    greyscale_image_path = os.path.join(greyscale_path, file_name.replace('.png', '_greyscale.png'))

    # Check if the image file exists
    if os.path.isfile(image_path):
        try:
            # Convert to greyscale
            convert_to_grayscale(image_path, greyscale_image_path)

            # Display the output image in Colab (optional, if running in a Jupyter/Colab environment)
            display(IPImage(filename=greyscale_image_path))

        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print(f"File {image_path} does not exist, skipping.")
