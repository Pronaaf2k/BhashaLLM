from PIL import Image, ImageDraw, ImageFont
import os

def create_image():
    # Setup
    width, height = 800, 200
    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Load Font
    font_path = "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf"
    try:
        font = ImageFont.truetype(font_path, 48)
    except IOError:
        print(f"Font not found at {font_path}")
        return

    # Text to draw (Bangla)
    text = "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি।"
    
    # Calculate position
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) / 2
    y = (height - text_height) / 2
    
    # Draw text
    draw.text((x, y), text, font=font, fill=(0, 0, 0))
    
    # Save
    out_path = "test_bangla.png"
    image.save(out_path)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    create_image()
