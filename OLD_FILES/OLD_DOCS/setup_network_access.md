# 🌐 วิธีให้คนอื่นเข้าใช้เว็บ PET Detect

## 📋 ขั้นตอนการตั้งค่า

### 1. เปิด Firewall (ต้องรัน PowerShell เป็น Administrator)

```powershell
# เปิด API Port 5000
netsh advfirewall firewall add rule name="PET Detect API" dir=in action=allow protocol=TCP localport=5000

# เปิด Member System Port 9000  
netsh advfirewall firewall add rule name="PET Detect Member System" dir=in action=allow protocol=TCP localport=9000
```

### 2. ตรวจสอบ IP Address

```powershell
ipconfig
```

หาค่า `IPv4 Address` ของ `Wireless LAN adapter Wi-Fi` หรือ `Ethernet adapter`

### 3. ให้คนอื่นเข้าเว็บ

```
http://[IP_ADDRESS]:9000
```

**ตัวอย่าง:**
```
http://192.168.1.31:9000
```

## 🚀 การใช้งาน

1. **เข้าเว็บ**: `http://[IP_ADDRESS]:9000`
2. **สมัครสมาชิก**: ใส่ RFID ID แล้วตั้งรหัสผ่าน
3. **ดูคะแนน**: เข้าสู่ระบบแล้วดูประวัติการสแกน

## ⚠️ ข้อควรระวัง

- Windows Firewall ต้องเปิด
- ต้องอยู่ในเครือข่ายเดียวกัน (Wi-Fi เดียวกัน)
- IP Address อาจเปลี่ยนเมื่อรีสตาร์ท Router

## 🔧 แก้ไขปัญหา

**ถ้าเข้าไม่ได้:**
1. ตรวจสอบ Firewall
2. ตรวจสอบ IP Address ใหม่
3. ตรวจสอบว่า Server ยังทำงานอยู่
