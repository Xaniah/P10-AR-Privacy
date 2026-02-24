from ultralytics import YOLO

def main():
    # Load a pretrained YOLO26n model
    model = YOLO("yolo26s.pt")

    # Train the model on the COCO8 dataset for 100 epochs
    # train_results = model.train(
    #     data="WIDER-FACE.yaml",  # Path to dataset configuration file
    #     epochs=100,  # Number of training epochs
    #     imgsz=640,  # Image size for training
    #     device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    # )

    # Resume training
    model = YOLO("runs/detect/train/weights/last.pt")
    train_results = model.train(
        resume=True,
    )

    # Evaluate the model's performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    results = model("627c86ee27d5960019ee3343.webp")  # Predict on an image
    results[0].show()  # Display results

if __name__ == "__main__":
    main()