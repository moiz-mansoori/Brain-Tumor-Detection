"""
Grad-CAM implementation for CNN-based MRI brain tumor classification.
Compatible with fine-tuned VGG-16 and nested Keras models.
"""

from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import cv2


def _find_last_conv_and_parent(model: Model) -> tuple[Optional[layers.Layer], Optional[Model]]:
    """
    Programmatically identification of the final Conv2D layer.
    Correctly traverses nested sub-models if necessary.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer, model
        if hasattr(layer, "layers"):
            res, p = _find_last_conv_and_parent(layer)
            if res is not None:
                return res, p
    return None, None


def generate_gradcam(
    model: Model,
    img_array: np.ndarray,
    target_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None
) -> np.ndarray:
    """
    Final robust Grad-CAM implementation for Keras 3.
    """
    # 0. Input validation
    if img_array.ndim != 4:
        raise ValueError(f"Input image must have 4 dims, got {img_array.ndim}")

    img_tensor = tf.cast(img_array, tf.float32)

    # 1. Identify target conv layer and its parent model
    target_layer = None
    parent_model = None

    if target_layer_name:
        def _find_by_name(m, name):
            for l in m.layers:
                if l.name == name: return l, m
                if hasattr(l, "layers"):
                    res, p = _find_by_name(l, name)
                    if res: return res, p
            return None, None
        target_layer, parent_model = _find_by_name(model, target_layer_name)
    
    if target_layer is None:
        target_layer, parent_model = _find_last_conv_and_parent(model)

    if target_layer is None:
        raise ValueError("Could not find a suitable Conv2D layer.")

    # 2. Get the "Real" prediction index from the original model
    if pred_index is None:
        preds_real = model(img_tensor, training=False)
        pred_index = int(tf.argmax(preds_real[0]))

    # 3. Build sub-model for bridging
    sub_grad_model = Model(
        inputs=parent_model.input, 
        outputs=[target_layer.output, parent_model.output]
    )

    # Find where parent_model connects to the top-level
    parent_layer_in_top = None
    for layer in model.layers:
        if layer.name == parent_model.name:
            parent_layer_in_top = layer
            break
        if hasattr(layer, "layers") and any(l is target_layer for l in layer.layers):
            parent_layer_in_top = layer
            break

    # Re-assemble post-layers
    post_layers = []
    if parent_layer_in_top:
        found = False
        for layer in model.layers:
            if found: post_layers.append(layer)
            if layer == parent_layer_in_top: found = True

    # 4. Gradient computation in Tape
    with tf.GradientTape() as tape:
        conv_outputs, base_output = sub_grad_model(img_tensor, training=False)
        tape.watch(conv_outputs)  # Explicitly watch conv activations (MUST be after definition)
        
        x = base_output
        for layer in post_layers:
            # Handle training-sensitive layers (Dropout, BN)
            if hasattr(layer, "call") and "training" in layer.call.__code__.co_varnames:
                x = layer(x, training=False)
            elif "dropout" in layer.name.lower():
                x = layer(x, training=False)
            else:
                x = layer(x)
        
        # predictions shape (1, num_classes)
        # Use one-hot masking instead of indexing to avoid "shape=(4,)" errors
        num_classes = x.shape[-1]
        one_hot_mask = tf.one_hot([pred_index], num_classes)
        class_score = tf.reduce_sum(x * one_hot_mask)

    # 5. Compute gradients and generate heatmap
    grads = tape.gradient(class_score, conv_outputs)
    if grads is None:
        # Final fallback: if sub-model bridging still fails, try standard tracing
        # which might work now that we're using a safer score selection
        grad_model = Model(inputs=model.inputs, outputs=[target_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            class_score = tf.reduce_sum(predictions * tf.one_hot([pred_index], num_classes))
        grads = tape.gradient(class_score, conv_outputs)
        
    if grads is None:
        raise RuntimeError("Grad-CAM failure: Path is not differentiable.")
        
    pooled_grads = tf.reduce_mean(grads, axis=[0, 1, 2])
    conv_outputs = conv_outputs[0]
    
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Normalize
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap) + 1e-10
    heatmap /= max_val
    
    return heatmap.numpy()


def overlay_heatmap(
    img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image using OpenCV.
    """
    heatmap_img = np.uint8(255 * heatmap)
    heatmap_img = cv2.applyColorMap(heatmap_img, colormap)
    heatmap_img = cv2.resize(heatmap_img, (img.shape[1], img.shape[0]))

    overlay = cv2.addWeighted(heatmap_img, alpha, img, 1 - alpha, 0)
    return overlay
