import numpy as np
from sklearn.metrics import f1_score, jaccard_score, roc_auc_score


class MetricsComputer:
    """
    A class for computing various evaluation metrics for binary classification problems, 
    particularly for image and pixel-level anomaly detection tasks.
    """
    def __init__(self, threshold: float = 0.5) -> None:
        """
        Initialize the MetricsComputer with a threshold for binary classification.

        Args:
            threshold (float): The threshold for binarizing anomaly maps. Default is 0.5.
        """
        self.threshold = threshold

    def compute_image_auc(self, i_pred: np.ndarray, i_label: np.ndarray) -> float:
        """
        Compute the Image-level Area Under the Curve (AUC) score.

        Args:
            i_pred (np.ndarray): Predicted scores or probabilities for images. Shape (N,).
            i_label (np.ndarray): Ground truth labels for images. Shape (N,).

        Returns:
            float: Image-level AUC score.
        """
        return roc_auc_score(i_label, i_pred)

    def compute_pixel_auc(self, masks_flat: np.ndarray, anomaly_map_flat: np.ndarray) -> float:
        """
        Compute the Pixel-level Area Under the Curve (AUC) score.

        Args:
            masks_flat (np.ndarray): Flattened ground truth binary masks. Shape (N,).
            anomaly_map_flat (np.ndarray): Flattened predicted anomaly maps. Shape (N,).

        Returns:
            float: Pixel-level AUC score.
        """
        return roc_auc_score(masks_flat.ravel(), anomaly_map_flat.ravel())

    def compute_f1_score(self, masks: np.ndarray, anomaly_map: np.ndarray) -> float:
        """
        Compute the F1 score based on binarized predictions.

        Args:
            masks (np.ndarray): Ground truth binary masks. Shape (H, W) or (N, H, W).
            anomaly_map (np.ndarray): Predicted anomaly maps. Shape (H, W) or (N, H, W).

        Returns:
            float: F1 score.
        """
        anomaly_map_binary = (anomaly_map > self.threshold).astype(np.float32).ravel()
        masks_binary = masks.ravel()
        return f1_score(masks_binary, anomaly_map_binary)

    def compute_iou(self, masks: np.ndarray, anomaly_map: np.ndarray) -> float:
        """
        Compute the Intersection over Union (IoU) score.

        Args:
            masks (np.ndarray): Ground truth binary masks. Shape (H, W) or (N, H, W).
            anomaly_map (np.ndarray): Predicted anomaly maps. Shape (H, W) or (N, H, W).

        Returns:
            float: IoU score.
        """
        anomaly_map_binary = (anomaly_map > self.threshold).astype(np.float32).ravel()
        masks_binary = masks.ravel()
        return jaccard_score(masks_binary, anomaly_map_binary)

    def compute_all_metrics(
        self, 
        masks: np.ndarray, 
        anomaly_map: np.ndarray, 
        i_pred: np.ndarray, 
        i_label: np.ndarray
    ) -> dict:
        """
        Compute all evaluation metrics and return them in a dictionary.

        Args:
            masks (np.ndarray): Ground truth binary masks. Shape (H, W) or (N, H, W).
            anomaly_map (np.ndarray): Predicted anomaly maps. Shape (H, W) or (N, H, W).
            i_pred (np.ndarray): Predicted scores or probabilities for images. Shape (N,).
            i_label (np.ndarray): Ground truth labels for images. Shape (N,).

        Returns:
            dict: A dictionary containing the computed metrics:
                - "image_auc": Image-level AUC score.
                - "pixel_auc": Pixel-level AUC score.
                - "f1": F1 score.
                - "iou": IoU score.
        """
        image_auc = self.compute_image_auc(i_pred, i_label)
        pixel_auc = self.compute_pixel_auc(masks, anomaly_map)
        f1 = self.compute_f1_score(masks, anomaly_map)
        iou = self.compute_iou(masks, anomaly_map)
        return {
            "image_auc": image_auc,
            "pixel_auc": pixel_auc,
            "f1": f1,
            "iou": iou
        }