"""
Phoneme-Enhanced Self-Supervised Learning (PESSL) module for SerenaNet.

This module implements the PESSL component that uses k-means clustering
for phoneme-enhanced pre-training with masked prediction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Optional FAISS for fast k-means (GPU when CUDA available)
_faiss_available = False
try:
    import faiss  # type: ignore

    _faiss_available = True
except ModuleNotFoundError:
    from sklearn.cluster import KMeans  # type: ignore

from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PESSL(nn.Module):
    """
    Phoneme-Enhanced Self-Supervised Learning module.
    
    This module implements masked prediction pre-training using k-means clustering
    to create phoneme-like pseudo-targets for self-supervised learning.
    
    Args:
        input_dim (int): Dimension of input features
        num_clusters (int): Number of k-means clusters (pseudo-phonemes)
        proj_dim (int): Dimension for projection head
        mask_prob (float): Probability of masking features
        mask_length (int): Length of mask spans
        temperature (float): Temperature for contrastive loss
    """
    
    def __init__(
        self,
        input_dim: int,
        num_clusters: int,
        proj_dim: int = 256,
        mask_prob: float = 0.15,
        mask_length: int = 10,
        temperature: float = 0.1
    ):
        super(PESSL, self).__init__()
        
        self.feature_dim = input_dim
        self.n_clusters = num_clusters
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.temperature = temperature
        
        # K-means clustering (will be fitted during training)
        self.kmeans = None
        self.cluster_centers = None
        self.is_fitted = False
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(input_dim))
        
        # Projection heads for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # Classifier for cluster prediction
        self.cluster_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_clusters)
        )
        
        # Layer norm for feature normalization
        self.feature_norm = nn.LayerNorm(input_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def fit_kmeans(self, features: torch.Tensor, max_samples: int = 100000):
        """
        Fit k-means clustering on features.
        
        Args:
            features (torch.Tensor): Features to cluster (N, D)
            max_samples (int): Maximum number of samples to use for clustering
        """
        # Convert to numpy and subsample if necessary
        if features.dim() == 3:
            # Reshape from (B, T, D) to (B*T, D)
            features = features.view(-1, features.size(-1))

        features_np = features.detach().cpu().numpy()
        
        if len(features_np) > max_samples:
            # Randomly subsample
            indices = np.random.choice(len(features_np), max_samples, replace=False)
            features_np = features_np[indices]
        
        logger.info(f"Fitting k-means with {len(features_np)} samples...")
        
        # ----------------------------------------------
        # Choose backend: FAISS (GPU/CPU) → sklearn fallback
        # ----------------------------------------------
        if _faiss_available:
            d = features_np.shape[1]
            n_clusters = self.n_clusters

            # faiss expects float32
            features32 = features_np.astype(np.float32)

            # Use GPU if CUDA is visible
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()  # type: ignore
                flat_config = faiss.GpuIndexFlatConfig()  # type: ignore
                flat_config.useFloat16 = False
                flat_config.device = 0
                index = faiss.GpuIndexFlatL2(res, d, flat_config)  # type: ignore
            else:
                index = faiss.IndexFlatL2(d)  # type: ignore

            self.kmeans = faiss.Clustering(d, n_clusters)  # type: ignore
            self.kmeans.niter = 20  # fewer iters than default 25
            self.kmeans.max_points_per_centroid = 8192
            self.kmeans.train(features32, index)  # type: ignore

            centroids = faiss.vector_to_array(self.kmeans.centroids).reshape(n_clusters, d)  # type: ignore
        else:
            from sklearn.cluster import KMeans  # local import to avoid unused error

            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')  # type: ignore[arg-type]
            kmeans.fit(features_np)
            centroids = kmeans.cluster_centers_
            self.kmeans = kmeans

        # Store cluster centers as a learnable parameter (not updated by optimizer)
        self.cluster_centers = nn.Parameter(torch.from_numpy(centroids).float(), requires_grad=False)
        self.is_fitted = True
        
        logger.info("K-means clustering completed!")
    
    def get_cluster_assignments(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get cluster assignments for features.
        
        Args:
            features (torch.Tensor): Input features (B, T, D)
            
        Returns:
            torch.Tensor: Cluster assignments (B, T)
        """
        if not self.is_fitted:
            raise ValueError("K-means must be fitted before getting cluster assignments")
        
        # Reshape for processing
        original_shape = features.shape
        features_flat = features.view(-1, features.size(-1))  # (B*T, D)
        
        # Get cluster assignments using numpy
        features_np = features_flat.detach().cpu().numpy().astype(np.float32)

        if _faiss_available and not isinstance(self.kmeans, (list, tuple)) and hasattr(self.kmeans, 'centroids'):
            # Use nearest-centroid search via FAISS IndexFlatL2
            d = features_np.shape[1]
            index = faiss.IndexFlatL2(d)  # type: ignore
            centroids = faiss.vector_to_array(self.kmeans.centroids).reshape(self.n_clusters, d)  # type: ignore
            index.add(centroids)
            distances, cluster_ids = index.search(features_np, 1)
            cluster_ids = cluster_ids.flatten().astype(np.int64)
        else:
            cluster_ids = self.kmeans.predict(features_np)  # type: ignore[attr-defined]
        
        # Convert back to tensor and reshape
        cluster_ids = torch.from_numpy(cluster_ids).to(features.device)
        cluster_ids = cluster_ids.view(original_shape[:-1])  # (B, T)
        
        return cluster_ids
    
    def create_masks(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create random masks for features.
        
        Args:
            batch_size (int): Batch size
            seq_len (int): Sequence length
            device (torch.device): Device for tensors
            
        Returns:
            torch.Tensor: Boolean mask (B, T)
        """
        masks = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        for b in range(batch_size):
            # Determine number of masks
            num_masks = int(seq_len * self.mask_prob / self.mask_length)
            
            for _ in range(num_masks):
                # Random start position
                start = torch.randint(0, max(1, seq_len - self.mask_length), (1,)).item()
                end = min(start + self.mask_length, seq_len)
                masks[b, start:end] = True
        
        return masks
    
    def apply_masks(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Apply masks to features by replacing with mask token.
        
        Args:
            features (torch.Tensor): Input features (B, T, D)
            masks (torch.Tensor): Boolean masks (B, T)
            
        Returns:
            torch.Tensor: Masked features (B, T, D)
        """
        masked_features = features.clone()
        
        # Replace masked positions with mask token
        mask_token_expanded = self.mask_token.expand_as(features)
        masked_features[masks] = mask_token_expanded[masks]
        
        return masked_features
    
    def compute_contrastive_loss(
        self, 
        features: torch.Tensor, 
        masked_features: torch.Tensor, 
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss on masked features.
        
        Args:
            features (torch.Tensor): Input features (B, T, D)
            masked_features (torch.Tensor): Masked features (B, T, D)
            masks (torch.Tensor): Boolean masks (B, T)
            
        Returns:
            torch.Tensor: Contrastive loss
        """
        features_flat = features.view(-1, features.size(-1))
        masked_features_flat = masked_features.view(-1, masked_features.size(-1))

        # Project features
        proj_features = self.projection_head(features_flat)
        proj_masked_features = self.projection_head(masked_features_flat)

        # Compute contrastive loss only on masked positions
        masked_indices = masks.view(-1)
        
        # Positive pairs (original vs. masked at same position)
        positives = F.cosine_similarity(
            proj_features[masked_indices],
            proj_masked_features[masked_indices]
        ).exp()

        # Negative pairs (masked vs. all others)
        negatives = torch.exp(
            F.cosine_similarity(
                proj_masked_features[masked_indices].unsqueeze(1),
                proj_features.unsqueeze(0),
                dim=-1
            ) / self.temperature
        ).sum(dim=1)
        
        loss = -torch.log(positives / (negatives + 1e-9)).mean()
        return loss
    
    def compute_cluster_prediction_loss(
        self, 
        features: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cluster prediction loss on masked positions.
        
        Args:
            features (torch.Tensor): Input features (B, T, D)
            targets (torch.Tensor): Target cluster assignments (B, T)
            masks (torch.Tensor): Boolean masks (B, T)
            
        Returns:
            Tuple containing:
                - Cluster prediction loss
                - Perplexity
        """
        if not self.is_fitted:
            return torch.tensor(0.0, device=features.device), torch.tensor(0.0, device=features.device)
        
        # Get cluster predictions
        cluster_logits = self.cluster_classifier(features)  # (B, T, n_clusters)
        
        # Only compute loss on masked positions
        if masks.sum() == 0:
            return torch.tensor(0.0, device=features.device), torch.tensor(0.0, device=features.device)
        
        masked_logits = cluster_logits[masks]
        masked_targets = targets[masks].long()
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(masked_logits, masked_targets)
        
        # Compute perplexity
        probs = F.softmax(masked_logits, dim=-1)
        perplexity = torch.exp(torch.mean(-torch.log(probs.clamp(min=1e-9))))
        
        return loss, perplexity
    
    def forward(
        self,
        features: torch.Tensor,
        masked_features: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for PESSL. Computes total loss and perplexity.
        """
        if masks is None:
            masks = self.create_masks(features.size(0), features.size(1), features.device)

        contrastive_loss = self.compute_contrastive_loss(features, masked_features, masks)
        
        with torch.no_grad():
            targets = self.get_cluster_assignments(features)
        
        cluster_loss, perplexity = self.compute_cluster_prediction_loss(masked_features, targets, masks)

        total_loss = contrastive_loss + cluster_loss
        return total_loss, perplexity
    
    def compute_loss(
        self, 
        features: torch.Tensor, 
        masked_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if masked_features is None:
            masks = self.create_masks(features.size(0), features.size(1), features.device)
            masked_features = self.apply_masks(features, masks)
        else:
            masks = torch.ones_like(features, dtype=torch.bool, device=features.device)

        loss, _ = self.forward(features, masked_features, masks)
        return loss
