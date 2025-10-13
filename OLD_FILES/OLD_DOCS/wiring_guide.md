# 🔌 คู่มือการต่อสาย Stepper Motor Driver

## ⚡ External Power Supply (แนะนำ)

### 🔧 การเชื่อมต่อ:

```
External Power Supply → Driver:
-------------------------------
+12V    → VCC (บน Driver)
GND     → GND (บน Driver)

Raspberry Pi → Driver:
----------------------  
GPIO 18 → PUL+ (Pulse Signal) - Pin 12
GPIO 19 → DIR+ (Direction Signal) - Pin 35
GND     → PUL-, DIR- (Signal Ground) - Pin 14/39

Driver → Stepper Motor:
-----------------------
A+, A-  → Motor Coil A
B+, B-  → Motor Coil B
```

### ✅ ข้อดี External Power:
- มอเตอร์หมุนแรงกว่า
- Pi ไม่ร้อน ไม่ restart
- ปลอดภัยกว่า
- ใช้งานได้นาน

### ⚠️ ข้อควรระวัง:
- ตรวจสอบ Voltage (12V/24V)
- ตรวจสอบ Current rating (2-3A)
- แยก Ground ระหว่าง Power และ Signal
- อย่าต่อ VCC เข้า Pi

## 🛒 Power Supply ที่แนะนำ:

| **Type** | **Spec** | **ราคา** | **เหมาะสำหรับ** |
|----------|----------|----------|------------------|
| **12V 3A** | Switching PSU | ~200฿ | ✅ NEMA 17 |
| **24V 2A** | Switching PSU | ~300฿ | ✅ High Speed |
| **12V 2A** | Wall Adapter | ~150฿ | ⚠️ Light Load |

## 🧪 การทดสอบ:
```bash
# ทดสอบระบบ
python test_professional_stepper.py

# รันระบบจริง
python pi_client_with_stepper.py
```