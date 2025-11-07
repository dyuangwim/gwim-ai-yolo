import cv2, sys
from battery_detector import BatteryDetector

img_path = "/home/pi/batch_images/sample_10packs.jpg"
weights  = "/home/pi/models/battery_ncnn_model"   # 就用你 detect_batch 的同一个路径

img = cv2.imread(img_path)
if img is None:
    print("Cannot read:", img_path)
    sys.exit(1)

det = BatteryDetector(weights, imgsz=416, conf=0.50, threads=4)
bats = det.detect_full(img)

print("Total batteries:", len(bats))
for i, b in enumerate(bats):
    print(i, b)

for b in bats:
    x1, y1, x2, y2 = b["xyxy"]
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

out_path = "/home/pi/batch_out/test_full_battery.jpg"
cv2.imwrite(out_path, img)
print("Saved to", out_path)
