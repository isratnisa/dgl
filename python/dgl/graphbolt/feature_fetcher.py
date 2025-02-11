"""Feature fetchers"""

from torchdata.datapipes.iter import Mapper


class FeatureFetcher(Mapper):
    """A feature fetcher used to fetch features for node/edge in graphbolt."""

    def __init__(self, datapipe, feature_store, feature_keys):
        """
        Initlization for a feature fetcher.

        Parameters
        ----------
        datapipe : DataPipe
            The datapipe.
        feature_store : FeatureStore
            A storage for features, support read and update.
        feature_keys : (str, str, str)
            Features need to be read, with each feature being uniquely identified
            by a triplet '(domain, type_name, feature_name)'.
        """
        super().__init__(datapipe, self._read)
        self.feature_store = feature_store
        self.feature_keys = feature_keys

    def _read(self, data):
        """
        Fill in the node/edge features field in data.

        Parameters
        ----------
        data : DataBlock
            An instance of the 'DataBlock' class. Even if 'node_feature' or
            'edge_feature' is already filled, it will be overwritten for
            overlapping features.

        Returns
        -------
        DataBlock
            An instance of 'DataBlock' filled with required features.
        """
        data.node_feature = {}
        num_layer = len(data.sampled_subgraphs) if data.sampled_subgraphs else 0
        data.edge_feature = [{} for _ in range(num_layer)]
        for key in self.feature_keys:
            domain, type_name, feature_name = key
            if domain == "node" and data.input_nodes is not None:
                nodes = (
                    data.input_nodes
                    if not type_name
                    else data.input_nodes[type_name]
                )
                if nodes is not None:
                    data.node_feature[
                        (type_name, feature_name)
                    ] = self.feature_store.read(
                        domain,
                        type_name,
                        feature_name,
                        nodes,
                    )
            elif domain == "edge" and data.sampled_subgraphs is not None:
                for i, subgraph in enumerate(data.sampled_subgraphs):
                    if subgraph.reverse_edge_ids is not None:
                        edges = (
                            subgraph.reverse_edge_ids
                            if not type_name
                            # TODO(#6211): Clean up the edge type converter.
                            else subgraph.reverse_edge_ids.get(
                                tuple(type_name.split(":")), None
                            )
                        )
                        if edges is not None:
                            data.edge_feature[i][
                                (type_name, feature_name)
                            ] = self.feature_store.read(
                                domain, type_name, feature_name, edges
                            )
        return data
