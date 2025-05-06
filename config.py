
raw_dir = r"./rawData/"
raw_drapra3_cadets_dir = raw_dir+"drapra3/cadets/"
# The directory to save all artifacts
processed_dir = "ProcessedData/drapra3/cadets/"
# processed_train_dir = processed_dir+"train/data/"
processed_train_dir = processed_dir+"train/data/data_mutil/"
processed_test_dir = processed_dir+"test/data/"
processed_evaluation_dir = processed_dir+"evaluation/data/"
groundtruth_dir = processed_dir+"evaluation/"

propagationRules_file = r"ProcessedData/drapra3/cadets/trans.txt"
processed_train_vec = "ProcessedData/drapra3/cadets/train/vec/"
processed_test_vec = "ProcessedData/drapra3/cadets/test/vec/"
processed_evaluation_vec = r"ProcessedData/drapra3/cadets/evaluation/vec/"

graph_catalogue = r"ProcessedData/drapra3/cadets/"
train_graph = graph_catalogue+"train/graph/"
test_graph = graph_catalogue+"test/graph/"
evaluation_graph = graph_catalogue+"evaluation/graph/"
# wl_train_all_file = r"ProcessedData/drapra3/cadets/train/data/WL_mutil_vec_d_3_128.txt"
wl_train_all_file = r"ProcessedData/drapra3/cadets/train/data/data_mutil/WL_vec_d_3_128.txt"
modelFilePathDir = r'models_encoder'

struct_size = 128
property_size = 16
anomalous_size = 15

train_subject_num = 25958
test_subject_num = 28271
evaluation_subject_num = 19810

alpha = 0.4
beta = 0.4
yam = 0.2