# แก้ไขไฟ LED เบา

## ⚡ วิธีทำให้ไฟ LED สว่างขึ้น:

### 1. ลด Resistor:
- **330Ω** → **220Ω** หรือ **100Ω** (สว่างขึ้น)
- **ไม่ใช้ Resistor** (สว่างสุด แต่เสี่ยงไฟไหม้)

### 2. เปลี่ยนเป็น LED แรงดันต่ำ:
- **LED 3V** แทน LED 5V
- **LED Strip 3.3V**

### 3. ใช้ External Power:
```
GPIO 2 → Transistor Base
5V → LED (+) → LED (-) → Transistor Collector
Transistor Emitter → GND
```

## ⚠️ ข้อควรระวัง:
- **ไม่ใช้ Resistor**: LED อาจไหม้
- **Resistor น้อย**: LED ร้อน
- **Current มาก**: GPIO อาจเสียหาย
