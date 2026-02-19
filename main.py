from ultralytics import YOLO

def main():
    # Load a pretrained YOLO26n model
    model = YOLO("yolo26n.pt")

    # Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(
        data="coco8.yaml",  # Path to dataset configuration file
        epochs=100,  # Number of training epochs
        imgsz=640,  # Image size for training
        device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    )

    # Evaluate the model's performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    results = model("path/to/image.extension")  # Predict on an image
    results[0].show()  # Display results

if __name__ == "__main__":
    main()