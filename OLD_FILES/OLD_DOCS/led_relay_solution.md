# LED Control ด้วย Relay Module

## 🔧 วิธีแก้ไขโดยไม่ใช้ Transistor

### ตัวเลือก 1: Relay Module 5V (แนะนำ)
```
Raspberry Pi:
- GPIO 2 → Relay IN
- 5V (Pin 2) → Relay VCC  
- GND (Pin 6) → Relay GND

Relay Output:
- COM → 5V Power
- NO → LED (+)
- LED (-) → GND
```

### ตัวเลือก 2: เปลี่ยนเป็น LED 3.3V
```
GPIO 2 → Resistor 220Ω → LED (+) → LED (-) → GND
```

### ตัวเลือก 3: ใช้ GPIO ควบคุม 5V Rail
```
GPIO 2 → ควบคุม 5V Enable Pin (ถ้ามี)
```

### ตัวเลือก 4: Software Switch
```
- ถอด LED ออกเมื่อไม่ใช้
- เสียบเข้าเมื่อต้องการใช้
```
