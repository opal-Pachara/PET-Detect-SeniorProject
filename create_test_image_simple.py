#!/usr/bin/env python3
"""
สร้างรูปภาพทดสอบง่ายๆ สำหรับ Postman
"""

from PIL import Image, ImageDraw
import os

def create_simple_test_image():
    """สร้างรูปภาพทดสอบง่ายๆ"""
    print("🖼️ สร้างรูปภาพทดสอบ...")
    
    # สร้างรูปภาพสีขาว 640x480
    width, height = 640, 480
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # วาดขวด PET ใหญ่ๆ สีเขียวเข้ม
    bottle_x, bottle_y = 250, 100
    bottle_width, bottle_height = 100, 200
    draw.rectangle([bottle_x, bottle_y, bottle_x + bottle_width, bottle_y + bottle_height], 
                   fill='darkgreen', outline='black', width=3)
    
    # วาดฝาขวด (วงกลมสีเขียวเข้ม)
    cap_x, cap_y = bottle_x + bottle_width//2 - 20, bottle_y - 30
    draw.ellipse([cap_x, cap_y, cap_x + 40, cap_y + 40], 
                 fill='green', outline='black', width=3)
    
    # วาดฉลาก (สี่เหลี่ยมสีเหลือง)
    label_x, label_y = bottle_x + 15, bottle_y + 50
    label_width, label_height = 70, 80
    draw.rectangle([label_x, label_y, label_x + label_width, label_y + label_height], 
                   fill='yellow', outline='orange', width=2)
    
    # วาดขวดที่สอง (เล็กกว่า)
    bottle2_x, bottle2_y = 100, 150
    bottle2_width, bottle2_height = 80, 150
    draw.rectangle([bottle2_x, bottle2_y, bottle2_x + bottle2_width, bottle2_y + bottle2_height], 
                   fill='darkgreen', outline='black', width=3)
    
    # วาดฝาขวดที่สอง
    cap2_x, cap2_y = bottle2_x + bottle2_width//2 - 15, bottle2_y - 25
    draw.ellipse([cap2_x, cap2_y, cap2_x + 30, cap2_y + 30], 
                 fill='green', outline='black', width=3)
    
    # วาดกระป๋อง (สีเงิน)
    can_x, can_y = 450, 200
    can_width, can_height = 60, 120
    draw.rectangle([can_x, can_y, can_x + can_width, can_y + can_height], 
                   fill='silver', outline='gray', width=3)
    
    # บันทึกรูปภาพ
    filename = 'test_bottles.jpg'
    image.save(filename, 'JPEG', quality=95)
    
    print(f"✅ สร้างรูปภาพทดสอบสำเร็จ: {filename}")
    print(f"   ขนาด: {width}x{height} pixels")
    print(f"   เนื้อหา: ขวด PET 2 ขวด + กระป๋อง 1 กระป๋อง")
    print(f"   คาดหวัง: ขวด 2, ฝา 2, ฉลาก 1, กระป๋อง 1")
    
    return filename

if __name__ == "__main__":
    create_simple_test_image()
