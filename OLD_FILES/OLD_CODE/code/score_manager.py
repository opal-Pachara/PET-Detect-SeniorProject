from datetime import datetime
from typing import List, Dict, Optional
from pymongo import MongoClient
import logging

logger = logging.getLogger(__name__)

class ScoreManager:
    """จัดการระบบคะแนนของผู้ใช้"""
    
    def __init__(self, db_connection_string: str):
        """เริ่มต้น ScoreManager"""
        self.client = MongoClient(db_connection_string)
        self.db = self.client["pet"]
        self.scores_collection = self.db["user_scores"]
        self.history_collection = self.db["score_history"]
    
    def save_score(self, username: str, score: int, counts: Dict[str, int], 
                   image_info: Optional[Dict] = None) -> bool:
        """บันทึกคะแนนของผู้ใช้"""
        try:
            # สร้างข้อมูลคะแนน
            score_data = {
                "username": username,
                "score": score,
                "counts": counts,
                "timestamp": datetime.now(),
                "image_info": image_info or {}
            }
            
            # บันทึกลงประวัติ
            self.history_collection.insert_one(score_data)
            
            # อัปเดตคะแนนรวมของผู้ใช้
            user_score = self.scores_collection.find_one({"username": username})
            
            if user_score:
                # อัปเดตข้อมูลที่มีอยู่
                total_score = user_score.get("total_score", 0) + score
                total_detections = user_score.get("total_detections", 0) + 1
                best_score = max(user_score.get("best_score", 0), score)
                
                self.scores_collection.update_one(
                    {"username": username},
                    {
                        "$set": {
                            "total_score": total_score,
                            "total_detections": total_detections,
                            "best_score": best_score,
                            "last_updated": datetime.now()
                        },
                        "$inc": {
                            "total_bottles": counts.get("bottle", 0),
                            "total_caps": counts.get("cap", 0),
                            "total_labels": counts.get("label", 0)
                        }
                    }
                )
            else:
                # สร้างข้อมูลใหม่
                new_user_score = {
                    "username": username,
                    "total_score": score,
                    "total_detections": 1,
                    "best_score": score,
                    "total_bottles": counts.get("bottle", 0),
                    "total_caps": counts.get("cap", 0),
                    "total_labels": counts.get("label", 0),
                    "created_at": datetime.now(),
                    "last_updated": datetime.now()
                }
                self.scores_collection.insert_one(new_user_score)
            
            logger.info(f"บันทึกคะแนนสำเร็จสำหรับผู้ใช้ {username}: {score}")
            return True
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการบันทึกคะแนน: {e}")
            return False
    
    def get_user_score(self, username: str) -> Optional[Dict]:
        """ดึงข้อมูลคะแนนของผู้ใช้"""
        try:
            return self.scores_collection.find_one({"username": username})
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลคะแนน: {e}")
            return None
    
    def get_user_history(self, username: str, limit: int = 10) -> List[Dict]:
        """ดึงประวัติคะแนนของผู้ใช้"""
        try:
            history = list(self.history_collection.find(
                {"username": username}
            ).sort("timestamp", -1).limit(limit))
            
            # แปลง ObjectId เป็น string
            for record in history:
                record["_id"] = str(record["_id"])
            
            return history
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงประวัติ: {e}")
            return []
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """ดึงตารางคะแนนสูงสุด"""
        try:
            leaderboard = list(self.scores_collection.find().sort("total_score", -1).limit(limit))
            
            # แปลง ObjectId เป็น string
            for record in leaderboard:
                record["_id"] = str(record["_id"])
            
            return leaderboard
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงตารางคะแนน: {e}")
            return []
    
    def get_user_stats(self, username: str) -> Dict:
        """ดึงสถิติของผู้ใช้"""
        try:
            user_score = self.get_user_score(username)
            if not user_score:
                return {
                    "total_score": 0,
                    "total_detections": 0,
                    "best_score": 0,
                    "average_score": 0,
                    "total_bottles": 0,
                    "total_caps": 0,
                    "total_labels": 0,
                    "rank": None
                }
            
            # คำนวณคะแนนเฉลี่ย
            total_detections = user_score.get("total_detections", 0)
            total_score = user_score.get("total_score", 0)
            average_score = total_score / total_detections if total_detections > 0 else 0
            
            # หาอันดับ
            all_users = list(self.scores_collection.find().sort("total_score", -1))
            rank = None
            for i, user in enumerate(all_users):
                if user["username"] == username:
                    rank = i + 1
                    break
            
            return {
                "total_score": total_score,
                "total_detections": total_detections,
                "best_score": user_score.get("best_score", 0),
                "average_score": round(average_score, 2),
                "total_bottles": user_score.get("total_bottles", 0),
                "total_caps": user_score.get("total_caps", 0),
                "total_labels": user_score.get("total_labels", 0),
                "rank": rank
            }
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงสถิติ: {e}")
            return {}
    
    def delete_user_data(self, username: str) -> bool:
        """ลบข้อมูลคะแนนของผู้ใช้"""
        try:
            self.scores_collection.delete_one({"username": username})
            self.history_collection.delete_many({"username": username})
            logger.info(f"ลบข้อมูลคะแนนสำเร็จสำหรับผู้ใช้ {username}")
            return True
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการลบข้อมูลคะแนน: {e}")
            return False 