"""
測試用合成圖生成工具
用於生成原始圖像 + overlay 和 mask + overlay 的合成圖
"""

import os
from PIL import Image
from amd_analysis import AMDVisualizer, INPUT_RAW_DIR, INPUT_MASK_DIR, OUTPUT_DIR


def create_composite_images(viz):
    """為 AMDVisualizer 實例建立測試用合成圖
    
    Args:
        viz: AMDVisualizer 實例，需已完成 draw_contours() 和 draw_vertical_caliper()
    
    Returns:
        tuple: (analyzed_path, mask_analyzed_path) 兩個合成圖的路徑
    """
    base_name = os.path.splitext(viz.filename)[0]
    
    # 讀取已生成的 overlay
    overlay_path = os.path.join(OUTPUT_DIR, f"{base_name}_overlay.png")
    if not os.path.exists(overlay_path):
        raise FileNotFoundError(f"找不到 overlay 檔案: {overlay_path}")
    
    overlay_image = Image.open(overlay_path).convert("RGBA")
    
    # 1. 原始圖像 + overlay
    combined_img = Image.alpha_composite(viz.raw_image, overlay_image)
    analyzed_path = os.path.join(OUTPUT_DIR, f"{base_name}_analyzed.png")
    combined_img.convert("RGB").save(analyzed_path)
    print(f"  -> 合成圖: {analyzed_path}")
    
    # 2. Mask + overlay
    mask_combined_img = Image.alpha_composite(viz.mask_image, overlay_image)
    mask_analyzed_path = os.path.join(OUTPUT_DIR, f"{base_name}_mask_analyzed.png")
    mask_combined_img.convert("RGB").save(mask_analyzed_path)
    print(f"  -> Mask合成圖: {mask_analyzed_path}")
    
    return analyzed_path, mask_analyzed_path


# def create_composite_from_files(filename):
#     """從檔案路徑建立合成圖（不依賴 AMDVisualizer 實例）
    
#     Args:
#         filename: 原始圖檔名稱（例如 "patient1_0091.png"）
    
#     Returns:
#         tuple: (analyzed_path, mask_analyzed_path) 兩個合成圖的路徑
#     """
#     base_name = os.path.splitext(filename)[0]
    
#     # 讀取原始圖像
#     raw_path = os.path.join(INPUT_RAW_DIR, filename)
#     if not os.path.exists(raw_path):
#         raise FileNotFoundError(f"找不到原始圖像: {raw_path}")
#     raw_image = Image.open(raw_path).convert("RGBA")
    
#     # 讀取 mask 圖像
#     mask_path = os.path.join(INPUT_MASK_DIR, filename)
#     if not os.path.exists(mask_path):
#         raise FileNotFoundError(f"找不到 mask 圖像: {mask_path}")
#     mask_image = Image.open(mask_path).convert("RGBA")
    
#     # 調整 mask 尺寸以符合原始圖像
#     if mask_image.size != raw_image.size:
#         mask_image = mask_image.resize(raw_image.size, resample=Image.NEAREST)
    
#     # 讀取 overlay
#     overlay_path = os.path.join(OUTPUT_DIR, f"{base_name}_overlay.png")
#     if not os.path.exists(overlay_path):
#         raise FileNotFoundError(f"找不到 overlay 檔案: {overlay_path}")
#     overlay_image = Image.open(overlay_path).convert("RGBA")
    
#     # 1. 原始圖像 + overlay
#     combined_img = Image.alpha_composite(raw_image, overlay_image)
#     analyzed_path = os.path.join(OUTPUT_DIR, f"{base_name}_analyzed.png")
#     combined_img.convert("RGB").save(analyzed_path)
#     print(f"  -> 合成圖: {analyzed_path}")
    
#     # 2. Mask + overlay
#     mask_combined_img = Image.alpha_composite(mask_image, overlay_image)
#     mask_analyzed_path = os.path.join(OUTPUT_DIR, f"{base_name}_mask_analyzed.png")
#     mask_combined_img.convert("RGB").save(mask_analyzed_path)
#     print(f"  -> Mask合成圖: {mask_analyzed_path}")
    
#     return analyzed_path, mask_analyzed_path


if __name__ == "__main__":
    """獨立執行：處理 output_result 中所有已生成的 overlay"""
    import glob
    
    overlay_files = glob.glob(os.path.join(OUTPUT_DIR, "*_overlay.png"))
    
    if not overlay_files:
        print(f"在 {OUTPUT_DIR} 中找不到 overlay 檔案")
        exit(1)
    
    print(f"找到 {len(overlay_files)} 個 overlay 檔案，開始生成合成圖...")
    
    for overlay_path in overlay_files:
        base_name = os.path.basename(overlay_path).replace("_overlay.png", "")
        # 嘗試找出對應的原始檔案名稱
        raw_files = glob.glob(os.path.join(INPUT_RAW_DIR, f"{base_name}.*"))
        
        if not raw_files:
            print(f"[SKIP] 找不到對應的原始檔案: {base_name}")
            continue
        
        filename = os.path.basename(raw_files[0])
        
        try:
            print(f"處理: {filename}")
            create_composite_from_files(filename)
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            import traceback
            traceback.print_exc()
