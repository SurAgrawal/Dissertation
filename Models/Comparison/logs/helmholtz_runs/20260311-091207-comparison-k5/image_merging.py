from PIL import Image, ImageDraw, ImageFont

# 1. Load your images in order
images = [
    Image.open('gd_solution_comparison.png'),
    Image.open('adam_solution_comparison.png'),
    Image.open('precond_solution_comparison.png'),
    Image.open('engd_solution_comparison.png')
]

# 2. Setup the title and dimensions
title_text = "Comparison - K = 5"  # <--- CHANGE THIS TO YOUR TITLE
title_space = 100               # Pixels of blank space at the top for the title

widths, heights = zip(*(i.size for i in images))
total_height = sum(heights) + title_space
max_width = max(widths)

# 3. Create a new blank image (white background)
new_im = Image.new('RGB', (max_width, total_height), color='white')

# 4. Prepare to draw the text
draw = ImageDraw.Draw(new_im)

# Try to load a nice, large font. (Arial for Windows/Mac, DejaVu for Linux)
try:
    font = ImageFont.truetype("arial.ttf", 50)
except IOError:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 50)
    except IOError:
        font = ImageFont.load_default()
        print("Note: Large font not found, using the small default font.")

# Center the title text
bbox = draw.textbbox((0, 0), title_text, font=font)
text_width = bbox[2] - bbox[0]
text_x = (max_width - text_width) // 2
text_y = 25 # Padding from the very top edge

# Draw the title
draw.text((text_x, text_y), title_text, fill="black", font=font)

# 5. Paste each image under the title
y_offset = title_space
for im in images:
    new_im.paste(im, (0, y_offset))
    y_offset += im.size[1]

# 6. Save the final result!
new_im.save('final_stacked_with_title.png')
print("Successfully combined with title!")