
from pathlib import Path
from PIL import Image
from src.predict import predict_image

project_dir = Path(__file__).resolve().parent
image_path = project_dir / "dataset" / "test" / "real" / "real_000000.jpg"

if not image_path.exists():
    raise FileNotFoundError(
        f"Test image not found: {image_path}\n"
        "Check the filename inside dataset/test/real."
    )

image = Image.open(image_path).convert("RGB")
result = predict_image(image)

if "error" in result:
    print(result["error"])
else:
    print("Prediction:", result["label"])
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print("Cropped face size:", result["face_image"].size)
