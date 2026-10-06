
from pathlib import Path
from PIL import Image
from src.gradcam import generate_gradcam

project_dir = Path(__file__).resolve().parent
image_path = (
    project_dir / "dataset" / "test" / "real" / "real_000000.jpg"
)

if not image_path.exists():
    raise FileNotFoundError(
        f"Image not found: {image_path}. Check the filename."
    )

image = Image.open(image_path).convert("RGB")
face, overlay = generate_gradcam(image)

if face is None:
    print("No face detected. Try another image.")
else:
    output_dir = project_dir / "outputs" / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    face.save(output_dir / "cropped_face.png")
    overlay.save(output_dir / "gradcam_overlay.png")

    print("Grad-CAM test completed!")
    print("Cropped face:", output_dir / "cropped_face.png")
    print("Heatmap overlay:", output_dir / "gradcam_overlay.png")
