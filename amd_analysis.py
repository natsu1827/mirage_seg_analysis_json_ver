import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ==========================================
# 設定區
# ==========================================
INPUT_RAW_DIR = "raw_image"
INPUT_MASK_DIR = "seg_mask"
OUTPUT_DIR = "output_result"

# 模擬前端輸入：若無數據請設為 None，若有數據請設為 float (例如 3.5)
USER_PIXEL_SPACING = None 

# AROI JSON Label 定義
LABEL_SRF = 161    # Mask 6
LABEL_CYST = 115   # Mask 5
LABEL_PED = 138    # Mask 5
LABEL_DRUSEN = 69  # Mask 3

# 色盲友善配色
COLORS = {
    "SRF":    {"hex": "#00FFFF", "rgb": (0, 255, 255)},    # Cyan
    "Cyst":   {"hex": "#FFA500", "rgb": (255, 165, 0)},    # Orange
    "PED":    {"hex": "#FF00FF", "rgb": (255, 0, 255)},    # Magenta
    "Drusen": {"hex": "#FFFF00", "rgb": (255, 255, 0)}     # Yellow
}

class AMDVisualizer:
    def __init__(self, raw_path, mask_path, filename, pixel_spacing=None):
        self.filename = filename
        self.pixel_spacing = pixel_spacing
        
        # 1. 讀取原始影像 (轉為 RGBA)
        self.raw_image = Image.open(raw_path).convert("RGBA")
        self.width, self.height = self.raw_image.size
        
        # 2. 建立一個全透明的圖層 (用於繪製標註)
        # (0, 0, 0, 0) 代表完全透明
        self.overlay_image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        
        # 3. 將畫筆設定在「透明圖層」上，而不是原圖上
        self.draw = ImageDraw.Draw(self.overlay_image)
        
        # 讀取 Mask 並處理尺寸
        mask_img = Image.open(mask_path)
        self.mask_array = np.array(mask_img)
        
        if self.mask_array.shape[:2] != (self.height, self.width):
            print(f"Warning: Resizing mask for {filename}")
            mask_img = mask_img.resize((self.width, self.height), resample=Image.NEAREST)
            self.mask_array = np.array(mask_img)

        # 儲存結果數據
        self.results = {
            "filename": filename,
            "pixel_spacing_um": pixel_spacing,
            "measurements": {}
        }

    def _format_value(self, value_px, type="length"):
        if self.pixel_spacing is None:
            return {"value": round(float(value_px), 2), "unit": "px"}
        
        if type == "length":
            val_um = value_px * self.pixel_spacing
            return {"value": round(val_um, 1), "unit": "um"}
        elif type == "area":
            val_mm2 = value_px * (self.pixel_spacing ** 2) / 1_000_000
            return {"value": round(val_mm2, 4), "unit": "mm2"}

    def draw_contours(self):
        """針對 SRF & Cyst：繪製空心輪廓 (畫在 overlay_image 上)"""
        targets = [
            (LABEL_SRF, "SRF"),
            (LABEL_CYST, "Cyst")
        ]

        for label_val, name in targets:
            binary_mask = (self.mask_array == label_val).astype(np.uint8) * 255
            pixel_count = np.sum(binary_mask > 0)
            
            if pixel_count > 0:
                self.results["measurements"][name] = {
                    "type": "Area",
                    "data": self._format_value(pixel_count, type="area"),
                    "color": COLORS[name]["hex"]
                }

                # 製作輪廓線
                mask_img = Image.fromarray(binary_mask, mode='L')
                edges = mask_img.filter(ImageFilter.FIND_EDGES)
                
                # 建立純色層
                color_rgb = COLORS[name]["rgb"]
                color_layer = Image.new("RGBA", (self.width, self.height), color_rgb)
                
                # 【關鍵修改】貼到 overlay_image 上，而不是 raw_image
                self.overlay_image.paste(color_layer, (0, 0), mask=edges)

    def draw_vertical_caliper(self):
        """針對 PED：繪製垂直測量線 (畫在 overlay_image 上)"""
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

        self.results["measurements"][name] = {
            "type": "Max Height",
            "data": self._format_value(max_h, type="length"),
            "color": COLORS[name]["hex"]
        }

        # 繪圖 (self.draw 已經指向 overlay_image)
        color = COLORS[name]["rgb"]
        lw = 2
        cap = 8
        self.draw.line([(best_x, best_y1), (best_x, best_y2)], fill=color, width=lw)
        self.draw.line([(best_x-cap, best_y1), (best_x+cap, best_y1)], fill=color, width=lw)
        self.draw.line([(best_x-cap, best_y2), (best_x+cap, best_y2)], fill=color, width=lw)

    def draw_horizontal_caliper(self):
        """針對 Drusen：繪製水平測量線 (畫在 overlay_image 上)"""
        label_val = LABEL_DRUSEN
        name = "Drusen"
        y_idxs, x_idxs = np.where(self.mask_array == label_val)
        
        if len(x_idxs) == 0: return

        base_y = np.max(y_idxs)
        bottom_pixels = x_idxs[y_idxs >= (base_y - 3)]
        
        if len(bottom_pixels) == 0: return

        min_x, max_x = np.min(bottom_pixels), np.max(bottom_pixels)
        width_px = max_x - min_x
        
        self.results["measurements"][name] = {
            "type": "Max Width",
            "data": self._format_value(width_px, type="length"),
            "color": COLORS[name]["hex"]
        }

        color = COLORS[name]["rgb"]
        draw_y = base_y + 2
        lw = 2
        cap = 5
        self.draw.line([(min_x, draw_y), (max_x, draw_y)], fill=color, width=lw)
        self.draw.line([(min_x, draw_y-cap), (min_x, draw_y+cap)], fill=color, width=lw)
        self.draw.line([(max_x, draw_y-cap), (max_x, draw_y+cap)], fill=color, width=lw)

    def save_results(self):
        base_name = os.path.splitext(self.filename)[0]
        
        # 1. 儲存純透明標註圖 (Overlay) - 必須存為 PNG 以保留透明度
        overlay_filename = f"{base_name}_overlay.png"
        overlay_path = os.path.join(OUTPUT_DIR, overlay_filename)
        self.overlay_image.save(overlay_path)
        
        # 2. 儲存合成圖 (Combined) - 將透明層疊在原圖上
        # 使用 alpha_composite 進行高品質疊圖
        combined_img = Image.alpha_composite(self.raw_image, self.overlay_image)
        analyzed_filename = f"{base_name}_analyzed.jpg"
        analyzed_path = os.path.join(OUTPUT_DIR, analyzed_filename)
        combined_img.convert("RGB").save(analyzed_path)
        
        # 3. 儲存 JSON
        json_filename = f"{base_name}_result.json"
        json_path = os.path.join(OUTPUT_DIR, json_filename)
        
        # 更新 JSON 資訊
        self.results["output_image"] = analyzed_filename  # 合成圖
        self.results["overlay_image"] = overlay_filename  # 透明圖 (新增)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
            
        return analyzed_path, overlay_path, json_path

# ==========================================
# 主程式
# ==========================================
def main():
    for d in [INPUT_RAW_DIR, INPUT_MASK_DIR, OUTPUT_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    raw_files = sorted([f for f in os.listdir(INPUT_RAW_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    if not raw_files:
        print(f"請將圖片放入 {INPUT_RAW_DIR} 資料夾")
        return

    print(f"開始分析... (Pixel Spacing: {USER_PIXEL_SPACING})")

    for filename in raw_files:
        raw_path = os.path.join(INPUT_RAW_DIR, filename)
        mask_path = os.path.join(INPUT_MASK_DIR, filename)
        
        if not os.path.exists(mask_path):
            print(f"[SKIP] 找不到 Mask: {filename}")
            continue
            
        try:
            viz = AMDVisualizer(raw_path, mask_path, filename, pixel_spacing=USER_PIXEL_SPACING)
            
            viz.draw_contours()
            viz.draw_vertical_caliper()
            viz.draw_horizontal_caliper()
            
            # 接收三個回傳路徑
            img_out, overlay_out, json_out = viz.save_results()
            
            print(f"完成: {filename}")
            print(f"  -> 合成圖: {img_out}")
            print(f"  -> 透明圖: {overlay_out}")
            print(f"  -> 數據檔: {json_out}")
            
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()