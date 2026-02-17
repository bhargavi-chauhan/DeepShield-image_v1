import torch
from PIL import Image
from torchvision import transforms

from models.image_model import DeepShieldImageModel
from utils.inference import predict_authenticity
from utils.explainability import GradCAM, explanation_strength, overlay_heatmap
from utils.multicrop import multi_crop_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = DeepShieldImageModel().to(device)
model.load_state_dict(torch.load("models/image_model.pth", map_location=device))

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load image
img_path = "test_image0.jpg"
image = Image.open(img_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

# 🔥 Multi-Crop Consensus
avg_score = multi_crop_inference(model, image, device, transform)

# 🔥 Final Decision
result = predict_authenticity(model, image_tensor, device)

print("\n🔍 DeepShield Prediction Report")
print("--------------------------------")
print(f"Authenticity Score : {avg_score:.4f}")
print(f"Prediction         : {result['label']}")
print(f"Confidence Level   : {result['confidence']}")
print(f"System Action      : {result['action']}")

# 🔥 Explainability Check
gradcam = GradCAM(model, model.backbone.features[-1])
heatmap = gradcam.generate(image_tensor.to(device))
strength = explanation_strength(heatmap)

if strength < 0.15:
    print("⚠ WARNING: Weak visual evidence detected")
    print("⚠ Prediction reliability is LOW")

overlay_heatmap(heatmap, img_path, "outputs/gradcam_result.jpg")
