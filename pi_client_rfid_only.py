#!/usr/bin/env python3
"""
ทดสอบ RFID ในระบบจริง - เฉพาะ RFID ก่อน
เพื่อแยกปัญหา GPIO conflict
"""

import time
import logging
import signal
import sys
from mfrc522 import SimpleMFRC522

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RFIDOnlyTest:
    def __init__(self):
        """Initialize RFID only"""
        try:
            self.rfid_reader = SimpleMFRC522()
            logger.info("RFID reader initialized successfully")
            print("✅ RFID reader พร้อมใช้งาน")
        except Exception as e:
            logger.error(f"RFID initialization failed: {e}")
            print(f"❌ RFID initialization failed: {e}")
            self.rfid_reader = None
            
        # Signal handler for graceful exit
        signal.signal(signal.SIGINT, self.signal_handler)
        self.running = True
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n🛑 กำลังหยุดระบบ...")
        self.running = False
        sys.exit(0)
    
    def read_rfid_with_timeout(self, timeout=30):
        """อ่าน RFID card พร้อม timeout"""
        if not self.rfid_reader:
            print("❌ RFID reader not initialized")
            return None, None
            
        print(f"🔍 รอการสแกน RFID card (timeout: {timeout} วินาที)...")
        print("📱 วางบัตร RFID ใกล้ตัวอ่าน...")
        
        start_time = time.time()
        last_status_time = start_time
        
        while time.time() - start_time < timeout and self.running:
            try:
                # ลองหลายวิธี RFID reading
                card_id = None
                text = ""
                
                # วิธีที่ 1: read_no_block
                try:
                    card_id, text = self.rfid_reader.read_no_block()
                    if card_id:
                        logger.debug("read_no_block success")
                except Exception as e:
                    logger.debug(f"read_no_block failed: {e}")
                
                # วิธีที่ 2: read_id_no_block (ถ้าวิธีแรกไม่ได้)
                if not card_id:
                    try:
                        card_id = self.rfid_reader.read_id_no_block()
                        text = ""
                        if card_id:
                            logger.debug("read_id_no_block success")
                    except Exception as e:
                        logger.debug(f"read_id_no_block failed: {e}")
                
                if card_id:
                    print(f"✅ RFID detected - ID: {card_id}")
                    print(f"📝 Text: '{text.strip()}'")
                    logger.info(f"RFID detected - ID: {card_id}")
                    return card_id, text
                
                # แสดงสถานะทุก 5 วินาที
                current_time = time.time()
                if current_time - last_status_time >= 5:
                    elapsed = int(current_time - start_time)
                    remaining = timeout - elapsed
                    print(f"⏳ รอ RFID... เหลือ {remaining} วินาที")
                    last_status_time = current_time
                
            except Exception as e:
                logger.debug(f"RFID read attempt failed: {e}")
            
            time.sleep(0.2)
        
        if self.running:
            print("❌ RFID timeout - ไม่พบบัตร")
        
        return None, None
    
    def run_continuous_test(self):
        """รันทดสอบ RFID อย่างต่อเนื่อง"""
        print("🚀 เริ่มทดสอบ RFID แบบต่อเนื่อง")
        print("กด Ctrl+C เพื่อหยุด")
        print("=" * 50)
        
        scan_count = 0
        
        while self.running:
            try:
                scan_count += 1
                print(f"\n🔄 รอบที่ {scan_count}:")
                
                card_id, text = self.read_rfid_with_timeout(timeout=30)
                
                if card_id:
                    print(f"🎉 สแกนสำเร็จ!")
                    print(f"   Card ID: {card_id}")
                    print(f"   Text: '{text.strip() if text else 'ไม่มี'}'")
                    
                    # รอให้ยกบัตรออก
                    print("⏳ รอให้ยกบัตรออกก่อนสแกนรอบต่อไป...")
                    time.sleep(3)
                else:
                    print("❌ ไม่พบบัตร RFID")
                
                # หน่วงเวลาก่อนรอบต่อไป
                if self.running:
                    print("💤 รอ 2 วินาทีก่อนรอบต่อไป...")
                    time.sleep(2)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in continuous test: {e}")
                time.sleep(1)
        
        print("\n✅ ทดสอบ RFID เสร็จสิ้น")

def main():
    """Main function"""
    print("🔧 ทดสอบ RFID เฉพาะ - แยกจาก Stepper Motor")
    print("=" * 60)
    
    # Create RFID test instance
    rfid_test = RFIDOnlyTest()
    
    if not rfid_test.rfid_reader:
        print("❌ ไม่สามารถเริ่มต้น RFID ได้")
        return
    
    # Run continuous test
    rfid_test.run_continuous_test()

if __name__ == "__main__":
    main()