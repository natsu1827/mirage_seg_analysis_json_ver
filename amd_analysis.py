import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# ==========================================
# 設定區
# ==========================================
INPUT_RAW_DIR = "raw_image"
INPUT_MASK_DIR = "seg_mask"
OUTPUT_DIR = "output_result"

# 模擬前端輸入：若無數據請設為 None，若有數據請設為 float (例如 3.5)
# USER_PIXEL_SPACING = 3.5 
USER_PIXEL_SPACING = None 

# AROI JSON Label 定義
LABEL_SRF = 161    # Mask 6
LABEL_CYST = 115   # Mask 5
LABEL_PED = 138    # Mask 5
LABEL_DRUSEN = 69  # Mask 3

# 色盲友善配色 (Hex Color for JSON, RGB for Pillow)
# 考量：高對比度、避開紅綠混淆 (Protanopia/Deuteranopia)
COLORS = {
    "SRF":    {"hex": "#00FFFF", "rgb": (0, 255, 255)},    # Cyan (青色)
    "Cyst":   {"hex": "#FFA500", "rgb": (255, 165, 0)},    # Orange (橙色)
    "PED":    {"hex": "#FF00FF", "rgb": (255, 0, 255)},    # Magenta (洋紅)
    "Drusen": {"hex": "#FFFF00", "rgb": (255, 255, 0)}     # Yellow (黃色)
}

class AMDVisualizer:
    def __init__(self, raw_path, mask_path, filename, pixel_spacing=None):
        self.filename = filename
        self.pixel_spacing = pixel_spacing
        
        # 讀取影像
        self.raw_image = Image.open(raw_path).convert("RGBA")
        self.width, self.height = self.raw_image.size
        
        # 讀取 Mask 並處理尺寸
        mask_img = Image.open(mask_path)
        self.mask_array = np.array(mask_img)
        
        if self.mask_array.shape[:2] != (self.height, self.width):
            print(f"Warning: Resizing mask for {filename}")
            mask_img = mask_img.resize((self.width, self.height), resample=Image.NEAREST)
            self.mask_array = np.array(mask_img)

        self.draw = ImageDraw.Draw(self.raw_image)
        
        # 儲存結果數據 (準備寫入 JSON)
        self.results = {
            "filename": filename,
            "pixel_spacing_um": pixel_spacing,
            "measurements": {}
        }

    def _format_value(self, value_px, type="length"):
        """
        根據是否有 pixel_spacing 自動轉換單位
        type: 'length' or 'area'
        """
        if self.pixel_spacing is None:
            return {"value": round(float(value_px), 2), "unit": "px"}
        
        if type == "length":
            val_um = value_px * self.pixel_spacing
            return {"value": round(val_um, 1), "unit": "um"}
        elif type == "area":
            # 面積換算： pixel^2 * (um/pixel)^2 / 1,000,000 = mm^2
            val_mm2 = value_px * (self.pixel_spacing ** 2) / 1_000_000
            return {"value": round(val_mm2, 4), "unit": "mm2"}

    def draw_contours(self):
        """
        針對 SRF & Cyst：繪製空心輪廓 (Contour)
        """
        targets = [
            (LABEL_SRF, "SRF"),
            (LABEL_CYST, "Cyst")
        ]

        for label_val, name in targets:
            # 建立二值化 Mask
            binary_mask = (self.mask_array == label_val).astype(np.uint8) * 255
            pixel_count = np.sum(binary_mask > 0)
            
            if pixel_count > 0:
                # 1. 計算數值
                self.results["measurements"][name] = {
                    "type": "Area",
                    "data": self._format_value(pixel_count, type="area"),
                    "color": COLORS[name]["hex"]
                }

                # 2. 製作輪廓線
                # 使用 Pillow 的 FIND_EDGES 濾鏡
                mask_img = Image.fromarray(binary_mask, mode='L')
                # 膨脹一點點讓線條連續 (可選)
                # mask_img = mask_img.filter(ImageFilter.MaxFilter(3)) 
                edges = mask_img.filter(ImageFilter.FIND_EDGES)
                
                # 3. 上色並疊加
                # 建立純色層
                color_rgb = COLORS[name]["rgb"]
                color_layer = Image.new("RGBA", (self.width, self.height), color_rgb)
                
                # 使用邊緣圖作為 Alpha Mask 貼上顏色
                self.raw_image.paste(color_layer, (0, 0), mask=edges)

    def draw_vertical_caliper(self):
        """
        針對 PED：繪製垂直測量線
        """
        label_val = LABEL_PED
        name = "PED"
        y_idxs, x_idxs = np.where(self.mask_array == label_val)
        
        if len(x_idxs) == 0: return

        unique_xs = np.unique(x_idxs)
        max_h = 0
        best_x, best_y1, best_y2 = 0, 0, 0

        for x in unique_xs:
            ys = y_idxs[x_idxs == x]
            h = ys.max() - ys.min()
            if h > max_h:
                max_h = h
                best_x = x
                best_y1 = ys.min()
                best_y2 = ys.max()

        # 紀錄數據
        self.results["measurements"][name] = {
            "type": "Max Height",
            "data": self._format_value(max_h, type="length"),
            "color": COLORS[name]["hex"]
        }

        # 繪圖 (僅線條，無文字)
        color = COLORS[name]["rgb"]
        lw = 2
        cap = 8
        self.draw.line([(best_x, best_y1), (best_x, best_y2)], fill=color, width=lw)
        self.draw.line([(best_x-cap, best_y1), (best_x+cap, best_y1)], fill=color, width=lw)
        self.draw.line([(best_x-cap, best_y2), (best_x+cap, best_y2)], fill=color, width=lw)

    def draw_horizontal_caliper(self):
        """
        針對 Drusen：繪製水平測量線
        """
        label_val = LABEL_DRUSEN
        name = "Drusen"
        y_idxs, x_idxs = np.where(self.mask_array == label_val)
        
        if len(x_idxs) == 0: return

        base_y = np.max(y_idxs)
        bottom_pixels = x_idxs[y_idxs >= (base_y - 3)]
        
        if len(bottom_pixels) == 0: return

        min_x, max_x = np.min(bottom_pixels), np.max(bottom_pixels)
        width_px = max_x - min_x
        
        # 紀錄數據
        self.results["measurements"][name] = {
            "type": "Max Width",
            "data": self._format_value(width_px, type="length"),
            "color": COLORS[name]["hex"]
        }

        # 繪圖 (僅線條，無文字)
        color = COLORS[name]["rgb"]
        draw_y = base_y + 2
        lw = 2
        cap = 5
        self.draw.line([(min_x, draw_y), (max_x, draw_y)], fill=color, width=lw)
        self.draw.line([(min_x, draw_y-cap), (min_x, draw_y+cap)], fill=color, width=lw)
        self.draw.line([(max_x, draw_y-cap), (max_x, draw_y+cap)], fill=color, width=lw)

    def save_results(self):
        # 1. 儲存圖片
        final_img = self.raw_image.convert("RGB")
        img_filename = f"analyzed_{self.filename}"
        img_path = os.path.join(OUTPUT_DIR, img_filename)
        final_img.save(img_path)
        
        # 2. 儲存 JSON
        json_filename = f"result_{os.path.splitext(self.filename)[0]}.json"
        json_path = os.path.join(OUTPUT_DIR, json_filename)
        
        # 將圖片路徑也寫入 JSON，方便前端關聯
        self.results["output_image"] = img_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
            
        return img_path, json_path

# ==========================================
# 主程式
# ==========================================
def main():
    # 初始化資料夾
    for d in [INPUT_RAW_DIR, INPUT_MASK_DIR, OUTPUT_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    raw_files = sorted([f for f in os.listdir(INPUT_RAW_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    if not raw_files:
        print(f"請將圖片放入 {INPUT_RAW_DIR} 資料夾")
        return

    print(f"開始分析... (Pixel Spacing: {USER_PIXEL_SPACING if USER_PIXEL_SPACING else 'Not Provided'})")

    for filename in raw_files:
        raw_path = os.path.join(INPUT_RAW_DIR, filename)
        mask_path = os.path.join(INPUT_MASK_DIR, filename)
        
        if not os.path.exists(mask_path):
            print(f"[SKIP] 找不到 Mask: {filename}")
            continue
            
        try:
            # 傳入 USER_PIXEL_SPACING (None 或 float)
            viz = AMDVisualizer(raw_path, mask_path, filename, pixel_spacing=USER_PIXEL_SPACING)
            
            viz.draw_contours()          # SRF/Cyst (輪廓)
            viz.draw_vertical_caliper()  # PED (線條)
            viz.draw_horizontal_caliper()# Drusen (線條)
            
            img_out, json_out = viz.save_results()
            
            print(f"完成: {filename}")
            print(f"  -> 圖檔: {img_out}")
            print(f"  -> 數據: {json_out}")
            
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

if __name__ == "__main__":
    main()