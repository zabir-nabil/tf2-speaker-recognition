class Config:
    def __init__(self, mode = None):
        if mode == None:
            self.test_map = {
                "normal": "test/voxceleb1_veri_test.txt",
                "hard": "test/voxceleb1_veri_test.txt",
                "extend": "test/voxceleb1_veri_test.txt"

            }

            self.result_path = "test/results"
