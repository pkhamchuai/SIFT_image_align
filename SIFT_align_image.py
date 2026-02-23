'''
How to run SIFT_align_image.py (Standalone SIFT Matching)

Method 1 (Command Line arguments):
python SIFT_align_image.py --img_list ./images/eye1_src.jpg ./images/eye1_tgt.jpg --method RANSAC
python SIFT_align_image.py --img_list 'test_images/NIR_M121HW_0.jpg' 'test_images/RGB_M121LW_0.jpg'

Method 2 (Read from CSV):
python SIFT_align_image.py --img_file my_pairs.csv --nn_threshold 0.7
'''

import argparse
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
import numpy as np
import csv
from datetime import datetime
from tqdm import tqdm

# === Import tools from Functions (Pure Numpy / OpenCV) ===
from SIFT_align_image_functions import (
    load_and_preprocess,
    check_transform_validity,
    DL_affine_plot_color
)

image_size = 256

def run_sift_match(src_gray, tgt_gray, nn_thresh):
    """Extract and match keypoints using SIFT."""
    
    src_uint8 = (src_gray * 255).astype(np.uint8)
    tgt_uint8 = (tgt_gray * 255).astype(np.uint8)
    
    sift = cv2.SIFT_create()
    kp1_cv, desc1 = sift.detectAndCompute(src_uint8, None)
    kp2_cv, desc2 = sift.detectAndCompute(tgt_uint8, None)
    
    heatmap1, heatmap2 = None, None 
    
    if desc1 is None or desc2 is None or len(kp1_cv) < 3 or len(kp2_cv) < 3:
        return np.array([]), np.array([]), heatmap1, heatmap2
        
    bf = cv2.BFMatcher()
    
    def match_sift_ratio(d1, d2, ratio):
        matches_knn = bf.knnMatch(d1, d2, k=2)
        good = []
        for m_n in matches_knn:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio * n.distance:
                    good.append(m)
        return good
        
    threshold = nn_thresh
    good_matches = match_sift_ratio(desc1, desc2, threshold)
    
    while len(good_matches) < 4 and threshold < 0.95:
        threshold += 0.1
        good_matches = match_sift_ratio(desc1, desc2, threshold)
        
    if len(good_matches) < 3:
        return np.array([]), np.array([]), heatmap1, heatmap2
        
    m1 = np.float32([kp1_cv[m.queryIdx].pt for m in good_matches])
    m2 = np.float32([kp2_cv[m.trainIdx].pt for m in good_matches])
    
    return m1, m2, heatmap1, heatmap2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Standalone SIFT Matching on Image Lists')
    
    parser.add_argument('--img_list', type=str, nargs='+', help='List of image pairs: src1 tgt1 src2 tgt2 ...')
    parser.add_argument('--img_file', type=str, help='Path to CSV/TXT file containing paths: src,tgt per line')
    
    parser.add_argument('--nn_threshold', type=float, default=0.7, 
                        help='Initial Nearest Neighbor ratio threshold for SIFT (default: 0.7)')
    parser.add_argument('--method', type=str, default='RANSAC', choices=['RANSAC', 'LMEDS'],
                        help='Affine estimation method (default: RANSAC)')
    
    parser.add_argument('--save', type=int, default=1, choices=[0, 1], 
                        help='0: Save only plot. 1: Save plot AND original color images')
    
    args = parser.parse_args()

    image_pairs = []
    if args.img_list:
        if len(args.img_list) % 2 != 0:
            print("Error: --img_list must have an even number of arguments.")
            exit(1)
        for i in range(0, len(args.img_list), 2):
            image_pairs.append((args.img_list[i], args.img_list[i+1]))
    elif args.img_file:
        if not os.path.exists(args.img_file):
            print(f"Error: File {args.img_file} not found.")
            exit(1)
        with open(args.img_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    image_pairs.append((parts[0].strip(), parts[1].strip()))
    else:
        print("Error: Must provide either --img_list or --img_file")
        exit(1)

    print(f"Found {len(image_pairs)} pairs to process.")

    est_method_flag = cv2.LMEDS if args.method == 'LMEDS' else cv2.RANSAC

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"output/SIFT_Standalone_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_rows = []

    for i, (src_path, tgt_path) in enumerate(tqdm(image_pairs, desc="Processing Pairs")):
        try:
            src_gray, src_rgb = load_and_preprocess(src_path, image_size)
            tgt_gray, tgt_rgb = load_and_preprocess(tgt_path, image_size)
        except Exception as e:
            print(f"Skip pair {i} due to error: {e}")
            continue

        m1, m2, heatmap1, heatmap2 = run_sift_match(src_gray, tgt_gray, args.nn_threshold)

        prefix = f"pair_{i:03d}_{os.path.basename(src_path).split('.')[0]}"
        text_log = "SIFT Failed"
        
        M_est = None
        warped_src_rgb = None
        m1_trans_plot = None

        if len(m1) >= 3:
            M_est, inliers = cv2.estimateAffinePartial2D(m1, m2, method=est_method_flag)

        if M_est is not None:
            warped_src_rgb = cv2.warpAffine(src_rgb, M_est, (image_size, image_size))
            is_bad, reason = check_transform_validity(warped_src_rgb, M_est)
            if not is_bad:
                inl_cnt = int(np.sum(inliers))
                text_log = f"SIFT Win (Inliers:{inl_cnt}/{len(m1)})"
                m1_trans_plot = cv2.transform(m1.reshape(-1, 1, 2), M_est).reshape(-1, 2).T
            else:
                text_log = f"Bad Transform: {reason}"

        m1_plot = m1.T if len(m1) > 0 else None
        m2_plot = m2.T if len(m2) > 0 else None

        # --- No Tensor needed here! Just pass Numpy Array (M_est) ---
        DL_affine_plot_color(prefix, output_dir,
                       "Result", text_log, src_rgb, tgt_rgb,
                       warped_src_rgb if warped_src_rgb is not None else src_rgb,
                       m1_plot, m2_plot, m1_trans_plot,
                       None, None,
                       affine_params_true=None,
                       affine_params_predict=M_est, 
                       heatmap1=heatmap1, heatmap2=heatmap2, plot=1)

        csv_src_path = src_path
        csv_tgt_path = tgt_path
        csv_warp_path = ""

        if args.save == 1:
            save_src = os.path.join(output_dir, f"{prefix}_src.png")
            save_tgt = os.path.join(output_dir, f"{prefix}_tgt.png")
            cv2.imwrite(save_src, cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_tgt, cv2.cvtColor(tgt_rgb, cv2.COLOR_RGB2BGR))
            csv_src_path = save_src
            csv_tgt_path = save_tgt

            if warped_src_rgb is not None:
                save_warp = os.path.join(output_dir, f"{prefix}_warp.png")
                cv2.imwrite(save_warp, cv2.cvtColor(warped_src_rgb, cv2.COLOR_RGB2BGR))
                csv_warp_path = save_warp

        csv_rows.append([i, csv_src_path, csv_tgt_path, csv_warp_path, len(m1), text_log])

    csv_file_path = os.path.join(output_dir, "results_log.csv")
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'source', 'target', 'warped', 'num_matches', 'status'])
        writer.writerows(csv_rows)

    print(f"\nFinished processing. Results saved to {output_dir}")
    print(f"CSV Log generated at: {csv_file_path}")