"""GPU cached feature for GraphBolt."""
import torch

from dgl.cuda import GPUCache

from ..feature_store import Feature

__all__ = ["GPUCachedFeature"]


class GPUCachedFeature(Feature):
    r"""GPU cached feature wrapping a fallback feature."""

    def __init__(self, fallback_feature: Feature, cache_size: int):
        """Initialize GPU cached feature with a given fallback.
        Places the GPU cache to torch.cuda.current_device().

        Parameters
        ----------
        fallback_feature : Feature
            The fallback feature.
        cache_size : int
            The capacity of the GPU cache, the number of features to store.

        Examples
        --------
        >>> import torch
        >>> torch_feat = torch.arange(0, 8)
        >>> cache_size = 5
        >>> fallback_feature = TorchBasedFeature(torch_feat)
        >>> feature = GPUCachedFeature(fallback_feature, cache_size)
        >>> feature.read()
        tensor([0, 1, 2, 3, 4, 5, 6, 7])
        >>> feature.read(torch.tensor([0, 1, 2]))
        tensor([0, 1, 2])
        >>> feature.update(torch.ones(3, dtype=torch.long),
        ... torch.tensor([0, 1, 2]))
        >>> feature.read(torch.tensor([0, 1, 2, 3]))
        tensor([1, 1, 1, 3])
        """
        super(GPUCachedFeature, self).__init__()
        assert isinstance(fallback_feature, Feature), (
            f"The fallback_feature must be an instance of Feature, but got "
            f"{type(fallback_feature)}."
        )
        self._fallback_feature = fallback_feature
        self.cache_size = cache_size
        # Fetching the feature dimension from the underlying feature.
        feat0 = fallback_feature.read(torch.tensor([0]))
        self.item_shape = (-1,) + feat0.shape[1:]
        feat0 = torch.reshape(feat0, (1, -1))
        self.flat_shape = (-1, feat0.shape[1])
        self._feature = GPUCache(cache_size, feat0.shape[1])

    def read(self, ids: torch.Tensor = None):
        """Read the feature by index.

        The returned tensor is always in GPU memory, no matter whether the
        fallback feature is in memory or on disk.

        Parameters
        ----------
        ids : torch.Tensor, optional
            The index of the feature. If specified, only the specified indices
            of the feature are read. If None, the entire feature is returned.

        Returns
        -------
        torch.Tensor
            The read feature.
        """
        if ids is None:
            return self._fallback_feature.read().to("cuda")
        keys = ids.to("cuda")
        values, missing_index, missing_keys = self._feature.query(keys)
        missing_values = self._fallback_feature.read(missing_keys).to("cuda")
        missing_values = missing_values.reshape(self.flat_shape)
        values = values.to(missing_values.dtype)
        values[missing_index] = missing_values
        self._feature.replace(missing_keys, missing_values)
        return torch.reshape(values, self.item_shape)

    def update(self, value: torch.Tensor, ids: torch.Tensor = None):
        """Update the feature.

        Parameters
        ----------
        value : torch.Tensor
            The updated value of the feature.
        ids : torch.Tensor, optional
            The indices of the feature to update. If specified, only the
            specified indices of the feature will be updated. For the feature,
            the `ids[i]` row is updated to `value[i]`. So the indices and value
            must have the same length. If None, the entire feature will be
            updated.
        """
        if ids is None:
            self._fallback_feature.update(value)
            size = min(self.cache_size, value.shape[0])
            self._feature.replace(
                torch.arange(0, size, device="cuda"),
                value[:size].to("cuda").reshape(self.flat_shape),
            )
        else:
            assert ids.shape[0] == value.shape[0], (
                f"ids and value must have the same length, "
                f"but got {ids.shape[0]} and {value.shape[0]}."
            )
            self._fallback_feature.update(value, ids)
            self._feature.replace(
                ids.to("cuda"), value.to("cuda").reshape(self.flat_shape)
            )
