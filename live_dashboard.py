import time
import os
import re
import sys

LOG_FILE = r"C:\Users\krish\.gemini\antigravity-cli\brain\07d3110b-8e6e-49d8-85be-30c1a11fdf41\.system_generated\tasks\task-572.log"

def draw_dashboard():
    total_imgs = 83058
    current_img = 0
    dupes = 0
    skipped = 0
    speed = 0.0
    is_training = False
    
    while True:
        try:
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in reversed(lines):
                    if "TRAINING FINAL DEEP NEURAL NETWORK" in line:
                        is_training = True
                        break
                    if "extracting features" in line:
                        # e.g.: [150/83058] extracting features … (39.0 img/s, 0 dupes, 0 skipped)
                        m = re.search(r'\[(\d+)/(\d+)\].*?\(([\d.]+) img/s, (\d+) dupes, (\d+) skipped', line)
                        if m:
                            current_img = int(m.group(1))
                            total_imgs = int(m.group(2))
                            speed = float(m.group(3))
                            dupes = int(m.group(4))
                            skipped = int(m.group(5))
                            break
        except Exception:
            pass
            
        # Clear screen for live update
        os.system('cls' if os.name == 'nt' else 'clear')
        
        pct = (current_img / total_imgs) * 100 if total_imgs > 0 else 0
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = '█' * filled + '-' * (bar_len - filled)
        
        print("="*65)
        print(" 🚀 PIPELINE BAY : LIVE TELEMETRY & ARCHITECTURE DIAGRAM 🚀")
        print("="*65)
        print()
        
        if not is_training:
            print(" [STAGE 1] : DATA HARMONIZATION & FEATURE EXTRACTION")
            print(f"   Progress: [{bar}] {pct:.1f}%")
            print(f"   Images Processed: {current_img} / {total_imgs}")
            print(f"   Extraction Speed: {speed:.1f} frames/sec")
            print(f"   Duplicates Nuked: {dupes}")
            print()
            
            t = int(time.time() * 3)
            p1 = "===>" if t % 3 == 0 else ("-==>" if t % 3 == 1 else "--=>")
            p2 = "===>" if t % 3 == 1 else ("-==>" if t % 3 == 2 else "--=>")
            p3 = "===>" if t % 3 == 2 else ("-==>" if t % 3 == 0 else "--=>")
            p_down = " ||\n \\/" if t % 2 == 0 else " ||\n  |"
            
            print("   --- LIVE PIPELINE FLOW (ANIMATED) ---")
            print(f"   [Raw Image Dataset] {p1} [MediaPipe Detector]")
            print(f"                                   {p_down}")
            print(f"   [478-Point Face Mesh] <{p2[::-1].replace('>','<')}- [Bounding Box Crop]")
            print(f"            {p_down}")
            print(f"   [971-D Vector] {p3} [Data Pool: {current_img} samples]")
            
        else:
            print(" [STAGE 2] : DEEP NEURAL NETWORK TRAINING")
            print("   Model: 3-Layer MLP (2048 -> 1024 -> 512)")
            print("   Status: Optimizing Weights with ADAM (Live Loss updating...)")
            print()
            t = int(time.time() * 5)
            w1 = "*" if t % 4 == 0 else "-"
            w2 = "*" if t % 4 == 1 else "-"
            w3 = "*" if t % 4 == 2 else "-"
            w4 = "*" if t % 4 == 3 else "-"
            
            print("   --- NEURAL NETWORK OPTIMIZATION FLOW ---")
            print(f"   [971-D Inputs]")
            print(f"         \\")
            print(f"          \\-{w1}---( 2048 Neurons )---{w2}-\\")
            print(f"           \\                          \\")
            print(f"            \\-{w3}---( 1024 Neurons )---{w4}--- [ classifier.pkl ]")
            print(f"           /                          /")
            print(f"          /-{w1}---( 512 Neurons  )---{w2}-/")
            print(f"         /")
            print(f"   [Adam Optimizer: Backpropagating Error]")
        
        print()
        print("=================================================================")
        print(" Dashboard refreshing every 1 second... (Press Ctrl+C to exit)")
        time.sleep(1)

if __name__ == "__main__":
    draw_dashboard()
