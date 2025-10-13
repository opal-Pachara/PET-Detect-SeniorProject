# การต่อ LED 5V ที่ถูกต้อง

## ⚠️ ปัญหาปัจจุบัน:
LED ต่อตรงกับ 5V → ไฟติดตลอดเวลา (ไม่สามารถควบคุมได้)

## ✅ วิธีแก้ไข: ใช้ Transistor

### 🔧 อุปกรณ์:
- Transistor NPN (2N2222 หรือ BC547)
- Resistor 1kΩ
- Breadboard

### 📋 การต่อสาย:

```
Raspberry Pi:
┌─────────────────────────┐
│ Pin 2 (5V) ─────────────┼─── LED (+)
│                         │      │
│ Pin 3 (GPIO 2) ─[1kΩ]───┼─── Transistor Base (B)
│                         │      │
│ Pin 6 (GND) ────────────┼─── Transistor Emitter (E)
│                         │
└─────────────────────────┘
                                 │
                          LED (-) ── Transistor Collector (C)
```

### 🔌 ขั้นตอนการต่อ:

#### 1. ถอดสาย LED เดิม:
- ถอด LED ออกจาก 5V และ GND

#### 2. ต่อ Transistor (2N2222):
```
Pin ของ Transistor:
E (Emitter)   → Pi GND (Pin 6)
B (Base)      → Pi GPIO 2 (Pin 3) ผ่าน Resistor 1kΩ  
C (Collector) → LED (-)
```

#### 3. ต่อ LED:
```
LED (+) → Pi 5V (Pin 2)
LED (-) → Transistor Collector
```

### 📐 Transistor Pinout (2N2222):
```
   ┌─ Collector (C)
   │
 ┌─┴─┐
 │ ▲ │ ← Base (B)
 └─┬─┘
   │
   └─ Emitter (E)
```

### ⚡ หลักการทำงาน:
- **GPIO 2 = LOW** → Transistor OFF → LED ดับ
- **GPIO 2 = HIGH** → Transistor ON → LED ติด

### 🧪 ทดสอบ:
1. ต่อสายตามแผนภาพ
2. รัน: `python led_off.py` → LED ควรดับ
3. รัน: `python pi_client_subprocess.py` → LED ควรควบคุมได้

### ⚠️ ข้อสำคัญ:
- **ต้องมี Resistor 1kΩ** ป้องกัน GPIO เสียหาย
- **ตรวจสอบขา Transistor** ให้ถูกต้อง
- **ทดสอบด้วย Multimeter** ถ้าไม่แน่ใจ
