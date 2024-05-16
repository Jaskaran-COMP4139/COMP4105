from PIL import Image

def resize_image(input_image_path, output_image_path, size):
    # Open the original image
    original_image = Image.open(input_image_path)
    # Resize the image
    resized_image = original_image.resize(size, Image.ANTIALIAS)
    # Save the resized image to the specified output path
    resized_image.save(output_image_path)

# Define your image path and size
input_path = 'car.gif'  # Modify with your actual image path
output_path = 'smallcar.gif'  # Desired path for the resized image
new_size = (28, 28)  # New dimensions for the image

# Call the function with your parameters
resize_image(input_path, output_path, new_size)

print("Image resized and saved as:", output_path)
