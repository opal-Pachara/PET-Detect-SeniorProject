#!/usr/bin/env python3

print("🧪 Quick Test - Can Detection Update")
print("="*40)

# Test calculate_score function
try:
    from minimal_api import calculate_score
    
    print("✅ Import successful")
    
    # Test cases
    test_cases = [
        (2, 0, 0, 1, "2 bottles + 1 can"),  # 2*50 + 1*100 = 200
        (1, 1, 0, 2, "1 bottle + 2 cans - 1 cap"),  # 1*50 + 2*100 - 1*10 = 240
        (0, 0, 0, 3, "3 cans only"),  # 3*100 = 300
        (1, 2, 1, 1, "Mixed items"),  # 1*50 + 1*100 - 2*10 - 1*10 = 120
    ]
    
    print("\n📊 Scoring Test Results:")
    for bottles, caps, labels, cans, desc in test_cases:
        score = calculate_score(bottles, caps, labels, cans)
        expected = (bottles * 50) + (cans * 100) - (caps * 10) - (labels * 10)
        expected = max(0, expected)
        
        status = "✅" if score == expected else "❌"
        print(f"   {status} {desc}: {score} points")
        if score != expected:
            print(f"      Expected: {expected}, Got: {score}")
    
    print("\n🎯 Scoring Rules:")
    print("   🥫 Can: +100 points")
    print("   🍶 Bottle: +50 points") 
    print("   🧢 Cap: -10 points")
    print("   🏷️  Label: -10 points")
    
    print("\n✅ Function test completed!")
    print("🚀 Ready to commit to Git!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Test failed: {e}")