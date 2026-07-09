import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.microex.pipeline import MicroExpressionPipeline, PipelineConfig

logging.basicConfig(level=logging.INFO)

def main():
    print("=" * 60)
    print("STARTING LIVE WEBCAM MICRO-EXPRESSION DETECTION")
    print("=" * 60)
    print("Please ensure your webcam is connected.")
    print("Press 'q' in the video window to quit.")
    print("=" * 60)

    config = PipelineConfig(
        input_mode="webcam",
        camera_index=0,
        model_path="models/classifier.pkl",
        face_detection_backend="haar",
        confidence_threshold=0.3,
        show_overlay=True
    )

    pipeline = MicroExpressionPipeline(config)
    
    try:
        pipeline.run_realtime()
    except Exception as e:
        print(f"Error during webcam streaming: {e}")
    finally:
        pipeline.release()

if __name__ == "__main__":
    main()
