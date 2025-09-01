from PIL import Image
import os

# Define the root directory containing subfolders with images
root_dir = "../resources/TypeOfClouds"


# Function to check and remove images smaller than 400x400
def remove_small_images(directory):
    # Iterate through all subfolders in the root directory
    for subfolder in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder)

        # Check if the path is a directory
        if os.path.isdir(subfolder_path):
            print(f"Processing subfolder: {subfolder}")
            # Iterate through all files in the subfolder
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)

                # Check if the file is an image (supports common formats)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    try:
                        # Open the image using PIL
                        with Image.open(file_path) as img:
                            # Get image dimensions
                            width, height = img.size
                            # Check if image is not 400x400
                            if width != 400 or height != 400:
                                # Delete the image file
                                os.remove(file_path)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                else:
                    print(f"Skipping non-image file: {file_path}")


if __name__ == "__main__":
    # Verify that the root directory exists
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} does not exist.")
    else:
        print(f"Starting to process images in {root_dir}")
        remove_small_images(root_dir)
        print("Image processing completed.")