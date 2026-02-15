from PIL import Image, ImageDraw, ImageFont
import os

def create_image():
    # Settings
    text = "আমি বাংলায় গান গাই\nআমি বাংলার গান গাই\n১৯৫২ সালের ভাষা আন্দোলন"
    font_path = "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf"
    output_path = "/home/node/.openclaw/workspace/BhashaLLM/data/raw/sample_ocr.jpg"
    
    # Create white image
    img = Image.new('RGB', (800, 400), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(font_path, 48)
    except Exception as e:
        print(f"Error loading font: {e}")
        return

    # Draw text
    d.text((50, 50), text, fill=(0, 0, 0), font=font)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"Created sample image: {output_path}")

if __name__ == "__main__":
    create_image()
