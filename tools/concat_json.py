import json
import os
from pathlib import Path


class JSONConcat(object):
    def __init__(self, parent_path):
        self.parent_path = Path(parent_path)
        self.all_json_files = []
        self.file_dict = []
        for filename in os.listdir(parent_path):
            if filename.endswith(".json"):
                filepath = str(os.path.join(parent_path, filename))
                self.all_json_files.append(filepath)
        for json_path in self.all_json_files:
            file_infos = []
            with open(json_path, "r") as f:
                for line in f:
                    info = json.loads(line)
                    file_infos.append(info.copy())
            min_epoch = file_infos[0]["epoch"]
            max_epoch = file_infos[-1]["epoch"]
            self.file_dict.append(dict(
                infos=file_infos,
                cover=(min_epoch, max_epoch),
                path=json_path
            ))
            self.file_dict = sorted(self.file_dict, key=lambda i: i["path"])

    def concat(self):
        concat_infos = []
        for i in range(len(self.file_dict)-1):
            this_dict = self.file_dict[i].copy()
            min_epoch, max_epoch = this_dict["cover"]
            if this_dict["cover"][1] >= self.file_dict[i+1]["cover"][0]:
                max_epoch = self.file_dict[i+1]["cover"][0] - 1
                for info in this_dict["infos"]:
                    if min_epoch <= info["epoch"] <= max_epoch:
                        concat_infos.append(info)
        concat_infos += self.file_dict[-1]["infos"]
        filename = os.path.join(self.parent_path, self.parent_path.parts[-1] + ".log.json")
        with open(filename, "w") as f:
            for info in concat_infos:
                json.dump(info, f)
                f.write("\n")


if __name__ == "__main__":
    data_name = "exp_mghead_v3_0"
    data_dir = "/mnt/proj50/zhengwu/saved_model/KITTI/proj52/megvii/second/"
    data_path = data_dir + data_name
    TEST = JSONConcat(parent_path=data_path)
    TEST.concat()
