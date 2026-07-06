import argparse
import sys
import logging
from src.microex.pipeline import MicroExpressionPipeline, PipelineConfig

def main():
    parser = argparse.ArgumentParser(description="Micro-Expression Detection System")
    parser.add_argument("--mode", choices=["webcam", "video", "dataset"], default="webcam",
                        help="Input mode: webcam, video file, or dataset.")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file (if mode=video)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset directory (if mode=dataset)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam index (default 0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

    # Set up the pipeline config
    config = PipelineConfig(
        input_mode=args.mode,
        camera_index=args.camera,
        video_path=args.video,
        dataset_path=args.dataset,
        face_detection_backend="mediapipe",  # Using mediapipe as the default detector
        model_path="models/classifier.pkl",  # Load our newly trained model!
        show_overlay=True
    )

    pipeline = MicroExpressionPipeline(config)

    try:
        if args.mode == "webcam":
            print("Starting webcam... (Press 'q' in the video window to quit)")
            pipeline.run_realtime()
        elif args.mode == "video":
            if not args.video:
                print("Error: --video path required when mode=video")
                sys.exit(1)
            pipeline.run_video(args.video)
        elif args.mode == "dataset":
            if not args.dataset:
                print("Error: --dataset path required when mode=dataset")
                sys.exit(1)
            pipeline.run_batch(args.dataset)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pipeline.release()

if __name__ == "__main__":
    main()
