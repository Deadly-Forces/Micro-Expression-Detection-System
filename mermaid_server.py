import os
import re
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

LOG_FILE = r"C:\Users\krish\.gemini\antigravity-cli\brain\07d3110b-8e6e-49d8-85be-30c1a11fdf41\.system_generated\tasks\task-1035.log"

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Enterprise Pipeline Bay</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
        }
        body { font-family: 'Inter', sans-serif; background: var(--bg-color); color: var(--text-primary); margin: 0; padding: 40px; display: flex; flex-direction: column; align-items: center; overflow: hidden; }
        h1 { font-weight: 500; color: #ffffff; letter-spacing: -0.5px; margin-bottom: 40px; font-size: 28px; }
        
        /* Stats Grid */
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; width: 100%; max-width: 1000px; margin-bottom: 50px; }
        .stat-card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 24px; text-align: left; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
        .stat-val { font-size: 28px; font-weight: 600; color: #ffffff; margin-bottom: 8px; }
        .stat-lbl { font-size: 13px; font-weight: 500; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
        
        /* Pipeline Canvas */
        .pipeline-container { position: relative; width: 100%; max-width: 1000px; height: 350px; background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
        
        /* Flow Nodes */
        .node { position: absolute; background: var(--bg-color); border: 1px solid var(--border-color); padding: 16px 20px; border-radius: 8px; display: flex; flex-direction: column; align-items: center; justify-content: center; z-index: 10; font-size: 14px; font-weight: 500; color: #fff; width: 160px; transition: all 0.5s ease; }
        .node-subtitle { font-size: 12px; color: var(--text-secondary); margin-top: 6px; font-weight: 400; }
        
        .node.active { border-color: var(--accent-blue); box-shadow: 0 0 0 2px rgba(88,166,255,0.2); }
        .node.success { border-color: var(--accent-green); box-shadow: 0 0 0 2px rgba(63,185,80,0.2); }
        .node.error { border-color: var(--accent-red); box-shadow: 0 0 0 2px rgba(248,81,73,0.2); }
        
        /* Specific Node Positions */
        #node-source { top: 140px; left: 40px; }
        #node-filter { top: 30px; left: 240px; }
        #node-detector { top: 140px; left: 420px; }
        #node-vector { top: 140px; left: 760px; }
        
        /* SVG Flow Lines */
        svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; pointer-events: none; }
        .path-line { fill: none; stroke: var(--border-color); stroke-width: 2; }
        .path-active { fill: none; stroke: var(--accent-blue); stroke-width: 2; stroke-dasharray: 8, 8; animation: flow 20s linear infinite; }
        .path-dupe { fill: none; stroke: var(--accent-red); stroke-width: 2; stroke-dasharray: 4, 4; animation: flow 2s linear infinite; opacity: 0.5; }
        .path-success { fill: none; stroke: var(--accent-green); stroke-width: 2; stroke-dasharray: 8, 8; animation: flow 15s linear infinite; }
        
        @keyframes flow { to { stroke-dashoffset: -1000; } }
        
        /* Data Packets */
        .packet { fill: var(--accent-blue); filter: drop-shadow(0 0 4px var(--accent-blue)); }
        .packet-success { fill: var(--accent-green); filter: drop-shadow(0 0 4px var(--accent-green)); }
        .packet-error { fill: var(--accent-red); opacity: 0.8; }
        
        /* Terminal Log */
        .terminal-container { width: 100%; max-width: 1000px; margin-top: 30px; background: #000; border: 1px solid var(--border-color); border-radius: 8px; padding: 15px; box-sizing: border-box; text-align: left; }
        .terminal-header { color: var(--text-secondary); font-size: 12px; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
        .terminal-log { color: #00ff00; font-family: 'Courier New', Courier, monospace; font-size: 13px; line-height: 1.4; white-space: pre-wrap; overflow-y: auto; max-height: 150px; }
    </style>
</head>
<body>
    <h1>Enterprise Pipeline Telemetry</h1>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-val" id="val-prog">0%</div>
            <div class="stat-lbl">Pipeline Progress</div>
        </div>
        <div class="stat-card">
            <div class="stat-val" id="val-imgs">0 / 0</div>
            <div class="stat-lbl">Processed Volumes</div>
        </div>
        <div class="stat-card">
            <div class="stat-val" id="val-dupe">0</div>
            <div class="stat-lbl">Redundancies Filtered</div>
        </div>
        <div class="stat-card">
            <div class="stat-val" id="val-spd">0</div>
            <div class="stat-lbl">Throughput (FPS)</div>
        </div>
    </div>
    
    <div class="pipeline-container">
        <svg>
            <path class="path-line" d="M 240 175 L 420 175" />
            <path class="path-line" d="M 620 175 L 760 175" />
            
            <path class="path-active" id="p1" d="M 240 175 L 420 175" />
            <path class="path-success" id="p2" d="M 620 175 L 760 175" />
            <path class="path-dupe" id="p3" d="M 140 140 L 140 65 L 240 65" />
            
            <circle class="packet" r="4"><animateMotion dur="2s" repeatCount="indefinite"><mpath href="#p1"/></animateMotion></circle>
            <circle class="packet-success" r="4"><animateMotion dur="2s" repeatCount="indefinite"><mpath href="#p2"/></animateMotion></circle>
            <circle class="packet-error" r="3"><animateMotion dur="0.8s" repeatCount="indefinite"><mpath href="#p3"/></animateMotion></circle>
        </svg>
        
        <div class="node active" id="node-source">
            Data Lake
            <div class="node-subtitle" id="txt-pool">Initializing...</div>
        </div>
        
        <div class="node error" id="node-filter">
            Integrity Filter
            <div class="node-subtitle" id="txt-dupe">0 conflicts</div>
        </div>
        
        <div class="node active" id="node-detector">
            Compute Engine
            <div class="node-subtitle" id="txt-spd">0 ops/sec</div>
        </div>
        
        <div class="node success" id="node-vector">
            Feature Store
            <div class="node-subtitle" id="txt-pct">0% synced</div>
        </div>
    </div>
    
    <div class="terminal-container">
        <div class="terminal-header">Raw Compiler Output (tail -15)</div>
        <div class="terminal-log" id="term-log">Waiting for log data...</div>
    </div>

    <!-- Injecting Mermaid library and the architectures -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({startOnLoad:true, theme: 'dark'});</script>
    
    <div style="width: 100%; max-width: 1000px; margin-top: 30px; display: flex; gap: 20px;">
        <div class="terminal-container" style="flex: 1; margin-top: 0; text-align: center;">
            <div class="terminal-header">Neural Flow Architecture</div>
            <div class="mermaid">
flowchart TD
    CAM[Webcam Feed] --> FD[Face Detector]
    FD --> LM[Landmark Extractor]
    LM --> MF[Motion Features]
    MF --> MLP[classifier.pkl]
    MLP --> SM[Softmax]
    SM --> UI[Live OpenCV Overlay]
    
    style CAM fill:#1e1e1e,stroke:#58a6ff,stroke-width:2px,color:#fff
    style FD fill:#161b22,stroke:#3fb950,stroke-width:2px,color:#fff
    style LM fill:#161b22,stroke:#3fb950,stroke-width:2px,color:#fff
    style MF fill:#161b22,stroke:#3fb950,stroke-width:2px,color:#fff
    style MLP fill:#161b22,stroke:#f85149,stroke-width:2px,color:#fff
    style SM fill:#161b22,stroke:#f85149,stroke-width:2px,color:#fff
    style UI fill:#1e1e1e,stroke:#58a6ff,stroke-width:2px,color:#fff
            </div>
        </div>
        <div class="terminal-container" style="flex: 1; margin-top: 0; text-align: center;">
            <div class="terminal-header">Project Working Tree</div>
            <div class="mermaid">
graph LR
    ROOT[OpenCV] --> DATA[data/]
    ROOT --> SCR[scripts/]
    ROOT --> SRC[src/microex/]
    
    SCR --> TR[train.py]
    SCR --> LV[live.py]
    
    style ROOT fill:#161b22,stroke:#58a6ff,stroke-width:2px,color:#fff
    style DATA fill:#161b22,stroke:#58a6ff,stroke-width:2px,color:#fff
    style SCR fill:#161b22,stroke:#58a6ff,stroke-width:2px,color:#fff
    style SRC fill:#161b22,stroke:#58a6ff,stroke-width:2px,color:#fff
    style TR fill:#0d1117,stroke:#8b949e,stroke-width:1px,color:#fff
    style LV fill:#0d1117,stroke:#8b949e,stroke-width:1px,color:#fff
            </div>
        </div>
    </div>

    <script>
        async function update() {
            try {
                let res = await fetch('/status');
                let data = await res.json();
                
                if (data.raw_log) {
                    document.getElementById('term-log').innerText = data.raw_log;
                }
                
                if (!data.is_training) {
                    document.getElementById('val-prog').innerText = data.pct + '%';
                    document.getElementById('val-imgs').innerText = data.current_img.toLocaleString() + ' / ' + data.total_imgs.toLocaleString();
                    document.getElementById('val-dupe').innerText = data.dupes.toLocaleString();
                    document.getElementById('val-spd').innerText = data.speed;
                    
                    document.getElementById('txt-pool').innerText = 'Volume: ' + data.total_imgs.toLocaleString();
                    document.getElementById('txt-dupe').innerText = 'Conflicts: ' + data.dupes.toLocaleString();
                    document.getElementById('txt-spd').innerText = 'Active: ' + data.speed + ' fps';
                    document.getElementById('txt-pct').innerText = 'Synced: ' + data.pct + '%';
                }
                
                let speed_val = parseFloat(data.speed);
                let dur = speed_val > 0 ? (30 / speed_val).toFixed(2) + 's' : '3s';
                document.querySelectorAll('.packet animateMotion').forEach(el => el.setAttribute('dur', dur));
                
                if (data.is_training) {
                    let m = Math.floor(data.training_elapsed / 60);
                    let s = data.training_elapsed % 60;
                    document.getElementById('val-prog').innerText = "100%";
                    document.getElementById('val-imgs').innerText = "Compiling...";
                    document.getElementById('val-dupe').innerText = m + "m " + s + "s";
                    document.getElementById('val-spd').innerText = data.cpu_speed;
                    document.querySelector('.stats-grid .stat-card:nth-child(3) .stat-lbl').innerText = "Time Elapsed";
                    
                    document.getElementById('node-source').innerHTML = "Feature Store<div class='node-subtitle'>Optimizing Target</div>";
                    document.getElementById('node-detector').innerHTML = "Adam Optimizer<div class='node-subtitle'>Backpropagating</div>";
                    document.getElementById('node-vector').innerHTML = "Master Model<div class='node-subtitle'>Building classifier.pkl</div>";
                }
            } catch(e) {}
        }
        setInterval(update, 1000);
        update();
    </script>
</body>
</html>
"""

import time
import subprocess

last_cpu_time = 0
last_cpu_str = "Max CPU"

def get_live_cpu():
    global last_cpu_time, last_cpu_str
    if time.time() - last_cpu_time > 3:
        try:
            p = subprocess.run(["powershell", "-Command", "Get-CimInstance Win32_Processor | Select-Object LoadPercentage, CurrentClockSpeed | ConvertTo-Json"], capture_output=True, text=True, timeout=1)
            cpu_data = json.loads(p.stdout)
            if isinstance(cpu_data, list):
                cpu_data = cpu_data[0]
            load = cpu_data.get("LoadPercentage", 100)
            clock = cpu_data.get("CurrentClockSpeed", 3000)
            last_cpu_str = f"{load}% @ {round(clock/1000, 2)}GHz"
        except:
            pass
        last_cpu_time = time.time()
    return last_cpu_str

class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
            
        elif self.path == '/status':
            total_imgs = 83058
            current_img = 0
            dupes = 0
            speed = 0.0
            is_training = False
            training_elapsed = 0
            live_cpu = get_live_cpu()
            raw_log = ""
            
            try:
                if os.path.exists(LOG_FILE):
                    with open(LOG_FILE, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                    raw_log = "".join(lines[-15:])
                    
                    import datetime
                    for line in reversed(lines):
                        if "TRAINING FINAL DEEP NEURAL NETWORK" in line:
                            is_training = True
                            m_time = re.search(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                            if m_time:
                                try:
                                    dt = datetime.datetime.strptime(m_time.group(1), "%Y-%m-%d %H:%M:%S")
                                    training_elapsed = int((datetime.datetime.now() - dt).total_seconds())
                                except:
                                    pass
                            break
                        if "extracting features" in line:
                            m = re.search(r'\[(\d+)/(\d+)\].*?\(([\d.]+) img/s, (\d+) dupes', line)
                            if m:
                                current_img = int(m.group(1))
                                total_imgs = int(m.group(2))
                                speed = float(m.group(3))
                                dupes = int(m.group(4))
                                break
                                
                    if is_training:
                        raw_log += f"\n> [SYSTEM HEARTBEAT] Active Optimization Loop... {live_cpu} [Elapsed: {training_elapsed}s]"
            except Exception:
                pass
                
            pct = round((current_img / total_imgs) * 100, 1) if total_imgs > 0 else 0
            
            payload = {
                "current_img": current_img,
                "total_imgs": total_imgs,
                "pct": pct,
                "speed": speed,
                "dupes": dupes,
                "is_training": is_training,
                "training_elapsed": training_elapsed,
                "cpu_speed": live_cpu,
                "raw_log": raw_log
            }
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server = HTTPServer(('127.0.0.1', 8008), DashboardHandler)
    print("Serving Live Dashboard at http://127.0.0.1:8008")
    server.serve_forever()
