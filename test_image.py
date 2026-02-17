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
img_path = "test_image11.jpg"
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

overlay_heatmap(heatmap, img_path, "gradcam_output.jpg")


# import torch
# from torchvision import transforms
# from PIL import Image

# from models.image_model import DeepShieldImageModel
# from utils.explainability import (
#     GradCAM,
#     overlay_heatmap,
#     explainability_consistency
# )

# # ------------------------------------
# # Device
# # ------------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ------------------------------------
# # Load Model
# # ------------------------------------
# model = DeepShieldImageModel().to(device)
# model.load_state_dict(torch.load("models/image_model.pth", map_location=device))
# model.eval()

# # ------------------------------------
# # Preprocessing
# # ------------------------------------
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# # ------------------------------------
# # NEW: Confidence Zone
# # ------------------------------------
# def confidence_zone(score):
#     if score >= 0.8:
#         return "HIGH CONFIDENCE REAL"
#     elif score >= 0.6:
#         return "LIKELY REAL"
#     elif score >= 0.4:
#         return "UNCERTAIN"
#     elif score >= 0.2:
#         return "LIKELY FAKE"
#     else:
#         return "HIGH CONFIDENCE FAKE"

# # ------------------------------------
# # NEW: Authenticity Prediction
# # ------------------------------------
# def predict_authenticity(model, image_tensor):
#     with torch.no_grad():
#         logits = model(image_tensor)
#         prob_real = torch.sigmoid(logits).item()

#     label = "REAL" if prob_real >= 0.5 else "FAKE"
#     zone = confidence_zone(prob_real)

#     return prob_real, label, zone

# # ------------------------------------
# # NEW: Explainability Report
# # ------------------------------------
# def generate_explainability_report(image_path):
#     image = Image.open(image_path).convert("RGB")
#     input_tensor = transform(image).unsqueeze(0).to(device)

#     # Prediction
#     score, label, zone = predict_authenticity(model, input_tensor)

#     # Grad-CAM
#     target_layer = model.backbone.features[-3]
#     gradcam = GradCAM(model, target_layer)
#     heatmap = gradcam.generate(input_tensor)

#     overlay_heatmap(
#         heatmap,
#         image_path,
#         "outputs/gradcam_result.jpg"
#     )

#     consistency = explainability_consistency(heatmap)

#     # Self-doubt flag
#     self_doubt = ""
#     if 0.45 <= score <= 0.55 and heatmap.max() < 0.4:
#         self_doubt = "⚠ Model uncertainty detected"

#     # --------------------------------
#     # FINAL REPORT
#     # --------------------------------
#     print("\n🧠 DEEPSHIELD EXPLAINABILITY REPORT")
#     print("----------------------------------")
#     print(f"Authenticity Score (Real): {score:.4f}")
#     print(f"Prediction: {label}")
#     print(f"Confidence Zone: {zone}")
#     print(f"Explainability Check: {consistency}")
#     if self_doubt:
#         print(self_doubt)
#     print("Grad-CAM saved at outputs/gradcam_result.jpg")

# # ------------------------------------
# # Run
# # ------------------------------------
# if __name__ == "__main__":
#     generate_explainability_report("test0.jpg")
# --------------------------------------------------------------------------------------------------------------

# import torch
# from torchvision import transforms
# from PIL import Image

# from models.image_model import DeepShieldImageModel
# from utils.explainability import GradCAM, overlay_heatmap

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = DeepShieldImageModel().to(device)
# model.load_state_dict(torch.load("models/image_model.pth", map_location=device))
# model.eval()

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# def predict_authenticity(image_tensor):
#     with torch.no_grad():
#         logits = model(image_tensor)
#         prob_real = torch.sigmoid(logits).item()

#     label = "REAL" if prob_real >= 0.5 else "FAKE"
#     return prob_real, label

# def generate_explainability_report(image_path):
#     image = Image.open(image_path).convert("RGB")
#     tensor = transform(image).unsqueeze(0).to(device)

#     score, label = predict_authenticity(tensor)

#     print("\n🧠 DeepShield Prediction")
#     print("------------------------")
#     print(f"Authenticity Score (Real): {score:.4f}")
#     print(f"Predicted Label: {label}")

#     target_layer = model.backbone.features[-3]
#     gradcam = GradCAM(model, target_layer)
#     heatmap = gradcam.generate(tensor)

#     overlay_heatmap(heatmap, image_path)
#     print("Explainability report generated.")

# if __name__ == "__main__":
#     generate_explainability_report("test.jpg")
