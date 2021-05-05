class Config:
    def __init__(self, mode = None):
        if mode == None:
            self.test_map = {
                "normal": "test/voxceleb1_veri_test.txt",
                "hard": "test/voxceleb1_veri_test.txt",
                "extend": "test/voxceleb1_veri_test.txt"

            }

            self.train_map = {
                "voxclb2_train": "test/voxlb2_train.txt",
                "voxclb2_val": "test/voxlb2_val.txt",

            }

            self.result_path = "test/results"
