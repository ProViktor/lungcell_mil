class DatasetMeta():
    def __init__(self, anndata_path, processed_x_path, x_embed_path, y_path, index_patient_dict, index_bag_dict, index_split_dict):
        self.path = anndata_path
        self.processed_x_path = processed_x_path
        self.x_embed_path = x_embed_path
        self.y_path = y_path
        self.index_patient_dict = index_patient_dict
        self.index_bag_dict = index_bag_dict
        self.index_split_dict = index_split_dict
        self.desc = "This is a meta object with the information about how the cell atlass dataset of 'hlca_subset.h5ad' is split to training, validation and testing"