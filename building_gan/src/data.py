class DataCreatorHelper:
    @staticmethod
    def process_graph_data(local_graph_data: dict, global_graph_data: dict):
        unique_indices = {}
        for ni, node in enumerate(local_graph_data["node"]):
            unique_indices[node["floor"], node["type"], node["type_id"]] = ni

        return

    def process_voxel_data(voxel_data: dict):
        pass


class DataCreator(DataCreatorHelper):
    def __init___(self):
        pass
