import json
import uuid
from pathlib import Path

import kaggle

from config import config

path = Path(config["weights_save"])
kaggle.api.dataset_initialize(path)

f = open(path / "dataset-metadata.json", mode="r")
kaggle_config = json.loads(f.read())
kaggle_config["title"] = config["title"]
kaggle_config["id"] = "kevinmathewt/" + str(uuid.uuid1())
f.close()

f = open(path / "dataset-metadata.json", mode="w")
json.dump(kaggle_config, f)
f.close()

print(kaggle_config)

kaggle.api.dataset_create_new(config["weights_save"])