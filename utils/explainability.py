import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.zero_grad()
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        target = output if output.item() >= 0.5 else 1 - output
        target.backward()

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(activations.shape[1]):
            activations[:, i] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= (torch.max(heatmap) + 1e-8)

        return heatmap.cpu().numpy()

# -------------------------------
# 🔥 Explanation Strength Metric
# -------------------------------
def explanation_strength(heatmap):
    return np.mean(heatmap)

# -------------------------------
# Heatmap Overlay
# -------------------------------
def overlay_heatmap(heatmap, original_image_path, output_path):
    image = cv2.imread(original_image_path)
    image = cv2.resize(image, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(image, 0.4, heatmap, 0.6, 0)
    cv2.imwrite(output_path, superimposed)

    print(f"Grad-CAM saved at {output_path}")



# import torch
# import cv2
# import numpy as np

# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
#         self._register_hooks()

#     def _register_hooks(self):

#         def forward_hook(module, input, output):
#             self.activations = output

#         def backward_hook(module, grad_in, grad_out):
#             self.gradients = grad_out[0]

#         self.target_layer.register_forward_hook(forward_hook)
#         self.target_layer.register_full_backward_hook(backward_hook)

#     def generate(self, input_tensor):
#         self.model.zero_grad()
#         input_tensor.requires_grad = True

#         output = self.model(input_tensor)
#         score = torch.sigmoid(output)

#         score.backward()

#         gradients = self.gradients.detach()
#         activations = self.activations.detach()

#         pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

#         for i in range(activations.shape[1]):
#             activations[:, i, :, :] *= pooled_gradients[i]

#         heatmap = torch.mean(activations, dim=1).squeeze()
#         heatmap = torch.relu(heatmap)
#         heatmap /= (heatmap.max() + 1e-8)

#         return heatmap.cpu().numpy()

# # ------------------------------------
# # Heatmap overlay
# # ------------------------------------
# def overlay_heatmap(heatmap, image_path, output_path):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (224, 224))

#     heatmap = cv2.resize(heatmap, (224, 224))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     result = cv2.addWeighted(image, 0.4, heatmap, 0.6, 0)
#     cv2.imwrite(output_path, result)

# # ------------------------------------
# # NEW: Explainability Consistency Check
# # ------------------------------------
# def explainability_consistency(heatmap):
#     h, w = heatmap.shape
#     center = heatmap[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]

#     face_energy = center.mean()
#     total_energy = heatmap.mean()

#     if face_energy >= total_energy:
#         return "Face-focused explanation (reliable)"
#     else:
#         return "Background-focused explanation (⚠ unreliable)"



# # import torch
# # import cv2
# # import numpy as np

# # class GradCAM:
# #     def __init__(self, model, target_layer):
# #         self.model = model
# #         self.target_layer = target_layer
# #         self.gradients = None
# #         self.activations = None
# #         self._register_hooks()

# #     def _register_hooks(self):
# #         def forward_hook(module, input, output):
# #             self.activations = output

# #         def backward_hook(module, grad_in, grad_out):
# #             self.gradients = grad_out[0]

# #         self.target_layer.register_forward_hook(forward_hook)
# #         self.target_layer.register_full_backward_hook(backward_hook)

# #     def generate(self, input_tensor):
# #         self.model.zero_grad()
# #         input_tensor.requires_grad_(True)

# #         logits = self.model(input_tensor)
# #         prob = torch.sigmoid(logits)

# #         # 🎯 Class-aware Grad-CAM
# #         target = logits if prob.item() >= 0.5 else -logits
# #         target.backward()

# #         gradients = self.gradients
# #         activations = self.activations

# #         pooled_gradients = torch.mean(gradients, dim=(0, 2, 3))
# #         weighted_activations = activations * pooled_gradients[None, :, None, None]

# #         heatmap = torch.mean(weighted_activations, dim=1).squeeze()
# #         heatmap = torch.relu(heatmap)
# #         heatmap /= heatmap.max() + 1e-8

# #         return heatmap.detach().cpu().numpy()


# # def overlay_heatmap(heatmap, image_path, output_path="outputs/gradcam_result.jpg"):
# #     image = cv2.imread(image_path)
# #     image = cv2.resize(image, (224, 224))

# #     heatmap = cv2.resize(heatmap, (224, 224))
# #     heatmap = np.uint8(255 * heatmap)
# #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# #     combined = cv2.addWeighted(image, 0.4, heatmap, 0.6, 0)
# #     cv2.imwrite(output_path, combined)

# #     print(f"Grad-CAM saved at {output_path}")
