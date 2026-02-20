import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import label

# ==========================================
# 設定區
# ==========================================
INPUT_RAW_DIR = "raw_image"
INPUT_MASK_DIR = "seg_mask"
OUTPUT_DIR = "output_result"

# AROI JSON Label 定義
LABEL_SRF = 161    # Mask 6
LABEL_CYST = 115   # Mask 5
LABEL_PED = 138    # Mask 5

# PED 區域高度閾值：小於此值使用箭頭標示，否則使用工字線
MIN_HEIGHT_FOR_CALIPER = 15

# 色盲友善配色
COLORS = {
    "SRF":    {"hex": "#00FFFF", "rgb": (0, 255, 255)},    # Cyan
    "Cyst":   {"hex": "#FFA500", "rgb": (255, 165, 0)},    # Orange
    "PED":    {"hex": "#FF00FF", "rgb": (255, 0, 255)}     # Magenta
}

class AMDVisualizer:
    def __init__(self, raw_path, mask_path, filename):
        self.filename = filename
        
        # 1. 【關鍵】讀取原始影像，並鎖定尺寸為「標準」
        self.raw_image = Image.open(raw_path).convert("RGBA")
        self.width, self.height = self.raw_image.size
        # print(f"Debug: Raw Image Size: {self.width}x{self.height}")
        
        # 2. 建立一個與原圖「尺寸完全相同」的全透明圖層
        self.overlay_image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.overlay_image)
        
        # 3. 讀取 Mask 並強制調整至原圖尺寸
        mask_img = Image.open(mask_path)
        
        # 如果 Mask 尺寸跟原圖不一樣，強制 Resize Mask (使用 NEAREST 保持 Label 數值)
        if mask_img.size != (self.width, self.height):
            print(f"Warning: Mask size {mask_img.size} mismatch. Resizing to {self.width}x{self.height}")
            mask_img = mask_img.resize((self.width, self.height), resample=Image.NEAREST)
        
        # 保存 mask 圖像為 RGBA 格式，用於後續合成
        self.mask_image = mask_img.convert("RGBA")
        self.mask_array = np.array(mask_img)

        # 儲存結果數據
        self.results = {
            "filename": filename,
            "image_width": self.width,   # 紀錄尺寸供前端參考
            "image_height": self.height,
            "measurements": {}
        }

    def _format_value(self, value_px, type="length"):
        """一律輸出 px，換算由前端負責"""
        return {"value": round(float(value_px), 2), "unit": "px"}

    def _draw_arrow(self, tip_x, bottom_y, size, color):
        """繪製箭頭於色塊下方：尖端指向灰度值138封閉色塊的底部，三角形與尾巴繪在下方不遮住色塊
        
        Args:
            tip_x: 箭頭尖端 x 座標（色塊底部該排的中心）
            bottom_y: 灰度值138封閉色塊底部 y 座標（箭頭尖端指向此）
            size: 箭頭大小（約 5-8 像素）
            color: 箭頭顏色 (R, G, B)
        """
        arrow_size = size
        half_width = arrow_size * 0.4
        tail_len = 6  # 尾巴線條長度（像素）
        
        # 尖端對準色塊底部；三角形畫在色塊下方（不遮住色塊）
        # 尖端向下移動 3px，與色塊保持距離
        x_tip = tip_x
        y_tip = bottom_y + 3  # 向下 3px，與色塊保持距離
        y_base = bottom_y + arrow_size + 3  # 三角形底邊也相應下移
        x1 = tip_x - half_width
        y1 = y_base
        x2 = tip_x + half_width
        y2 = y_base
        
        # 實心三角形（箭頭，尖端朝上對準色塊底部）
        self.draw.polygon([(x1, y1), (x2, y2), (x_tip, y_tip)], fill=color)
        # 尾巴線條：從三角形底邊中心向下延伸
        self.draw.line([(tip_x, y_base), (tip_x, y_base + tail_len)], fill=color, width=1)

    def draw_contours(self):
        """針對 SRF & Cyst：繪製空心輪廓"""
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
                
                # 建立純色層 (尺寸確保與原圖一致)
                color_rgb = COLORS[name]["rgb"]
                color_layer = Image.new("RGBA", (self.width, self.height), color_rgb)
                
                # 貼到 overlay_image 上
                self.overlay_image.paste(color_layer, (0, 0), mask=edges)

    def draw_vertical_caliper(self):
        """針對 PED：對每個封閉區域繪製垂直測量線"""
        label_val = LABEL_PED
        name = "PED"
        
        # 建立二值化遮罩
        binary_mask = (self.mask_array == label_val).astype(int)
        
        # 使用連通組件標記找出所有封閉區域
        labeled_array, num_features = label(binary_mask)
        
        if num_features == 0:
            return
        
        # 儲存所有區域的測量值
        measurements_list = []
        color = COLORS[name]["rgb"]
        lw = 2
        cap = 8
        
        # 對每個連通區域處理
        for region_id in range(1, num_features + 1):
            # 提取該區域的所有像素座標
            y_idxs, x_idxs = np.where(labeled_array == region_id)
            
            if len(x_idxs) == 0:
                continue
            
            # 找出該區域內所有唯一的 x 座標
            unique_xs = np.unique(x_idxs)
            max_h = 0
            best_x, best_y1, best_y2 = 0, 0, 0
            
            # 對每個 x 座標計算垂直高度
            for x in unique_xs:
                ys = y_idxs[x_idxs == x]
                h = ys.max() - ys.min()
                if h > max_h:
                    max_h = h
                    best_x = x
                    best_y1 = ys.min()
                    best_y2 = ys.max()
            
            # 記錄該區域的測量值
            measurements_list.append(self._format_value(max_h, type="length"))
            
            # 根據區域高度選擇標示方式
            if max_h < MIN_HEIGHT_FOR_CALIPER:
                # 小區域：箭頭指向灰度值138封閉色塊的底部（最底一排像素的中心）
                bottom_y = float(y_idxs.max())
                bottom_row_x = x_idxs[y_idxs == bottom_y]
                tip_x = float((bottom_row_x.min() + bottom_row_x.max()) / 2)
                arrow_size = 7
                self._draw_arrow(tip_x, bottom_y, arrow_size, color)
            else:
                # 大區域：使用工字線（垂直線 + 上下橫線）
                self.draw.line([(best_x, best_y1), (best_x, best_y2)], fill=color, width=lw)
                self.draw.line([(best_x-cap, best_y1), (best_x+cap, best_y1)], fill=color, width=lw)
                self.draw.line([(best_x-cap, best_y2), (best_x+cap, best_y2)], fill=color, width=lw)
        
        # 將所有測量值儲存為數組格式
        if measurements_list:
            self.results["measurements"][name] = {
                "type": "Max Height",
                "data": measurements_list,
                "color": COLORS[name]["hex"]
            }

    def draw_legend(self):
        """在左上角繪製圖例，僅顯示有量測結果的 SRF/Cyst/PED"""
        if not self.results["measurements"]:
            return
        order = ["SRF", "Cyst", "PED"]
        x0, y0 = 12, 12
        block_size = 12
        gap = 4
        for name in order:
            if name not in self.results["measurements"]:
                continue
            color = COLORS[name]["rgb"]
            self.draw.rectangle([x0, y0, x0 + block_size, y0 + block_size], fill=color)
            self.draw.text((x0 + block_size + gap, y0), name, fill=(255, 255, 255))
            y0 += block_size + gap

    def save_results(self):
        base_name = os.path.splitext(self.filename)[0]
        
        # 1. 儲存純透明標註圖 (Overlay)
        # 尺寸保證：self.overlay_image 初始化時即鎖定為 self.width/height
        overlay_filename = f"{base_name}_overlay.png"
        overlay_path = os.path.join(OUTPUT_DIR, overlay_filename)
        self.overlay_image.save(overlay_path)
        
        # 2. 儲存 JSON
        json_filename = f"{base_name}_result.json"
        json_path = os.path.join(OUTPUT_DIR, json_filename)
        
        self.results["overlay_image"] = overlay_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
            
        return overlay_path, json_path

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

    print(f"開始分析...")

    for filename in raw_files:
        raw_path = os.path.join(INPUT_RAW_DIR, filename)
        mask_path = os.path.join(INPUT_MASK_DIR, filename)
        
        if not os.path.exists(mask_path):
            print(f"[SKIP] 找不到 Mask: {filename}")
            continue
            
        try:
            viz = AMDVisualizer(raw_path, mask_path, filename)
            
            viz.draw_contours()
            viz.draw_vertical_caliper()
            viz.draw_legend()
            
            overlay_out, json_out = viz.save_results()
            
            print(f"完成: {filename}")
            print(f"  -> 尺寸: {viz.width}x{viz.height}")
            print(f"  -> 透明圖: {overlay_out}")
            
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()