from ultralytics import YOLO

def main():
    # Load a pretrained YOLO26n model
    model = YOLO("runs/detect/train/weights/last.pt")

    # Train the model
    train_results = model.train(
        data="dataset-config.yaml",  # Path to dataset configuration file
        epochs=300,  # Number of training epochs
        imgsz=1280,  # Image size for training
        device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
        batch=-1,
    )

    # Resume training
    # model = YOLO("runs/detect/train/weights/last.pt")
    # train_results = model.train(
    #     resume=True,
    # )

    # Use best model
    # model = YOLO("runs/detect/train/weights/best.pt")

    # Evaluate the model's performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    results = model("imgs/IMG_20260317_091245.jpg", imgsz=1280)  # Predict on an image
    results[0].show()  # Display results

if __name__ == "__main__":
    main()