from pathlib import Path
import pandas as pd
import numpy as np
from skimage import io as skio
from dataclasses import dataclass, field
from skimage import measure as skms
from sklearn.model_selection import train_test_split
from typing import List


@dataclass
class Plant:
    index: int = field(repr=True)
    plant_label: int = field(repr=True)
    area: int = field(repr=True)


@dataclass
class Patch:
    id: int = field(init=True, repr=True)
    fname: str = field(init=True, repr=True)
    plants: List[Plant] = field(default_factory=list, repr=False)
    labels: np.ndarray = field(init=False, repr=False)
    image: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._load_image()
        self.generate_plants()

    def _load_image(self):  # path to the image folder
        image = skio.imread(f"data/gt/{self.fname}.png")
        # images are separated by a 4px wide grid
        msk = image[136:264, 4:132, :3]
        img = image[4:132, 4:132, :3]
        self.image = img
        self._encode_masks(msk)
        return

    def _encode_masks(self, rgb_mask):
        """
        encodes 4D numpy array
        """
        labels = np.array([[199, 199, 199], [31, 119, 180], [255, 127, 14]], dtype=np.uint8)
        self.labels = labels
        rgb_mask = rgb_mask.reshape((-1, 128, 3))
        label_map = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
        for idx, label in enumerate(labels):
            label_map[(rgb_mask == np.array(label)).all(axis=2)] = idx
        self.mask = label_map

        return

    @classmethod
    def get_binary_labelmap(cls, label_map, label):
        binary_map = np.zeros(label_map.shape[:2], dtype=np.uint8)
        binary_map[label_map != label] = 0
        binary_map[label_map == label] = 1
        return binary_map

    def generate_plants(self):
        plants = []
        idx = 0
        for label in range(1, len(self.labels)):
            binary_map = self.get_binary_labelmap(self.mask, label)
            cc_map = skms.label(binary_map, background=0)
            props = skms.regionprops(cc_map)
            for prop in props:
                plant = Plant(idx, label,prop["area"])
                plants.append(plant)
                idx += 1

        self.plants = plants
        return plants

    def get_area_by_plant_label(self, plant_label):
        areas = []
        for plant in self.plants:
            if plant.plant_label == plant_label:
                areas.append(plant.area)

        return np.sum(areas)


@dataclass
class PatchStratifier:
    fnames: List = field(init=True, repr=False)
    patches: List[Patch] = field(init=False, repr=False)
    df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        patches = []
        for idx, fname in enumerate(self.fnames):
            p = Patch(idx, fname)
            patches.append(p)
        self.patches = patches

    def get_number_of_plants(self):
        num_plants = [[pa.fname, len(pa.plants)] for pa in self.patches]
        num_plants_df = pd.DataFrame(num_plants, columns=["filename", "num_plants"])
        return num_plants_df

    def get_plant_types(self):
        areas = [[pa.fname, pa.get_area_by_plant_label(1), pa.get_area_by_plant_label(2)] for pa in self.patches]
        area_df = pd.DataFrame(areas, columns=["filename", "area_sorghum", "area_weed"])

        # append plant_type: 1: only sorghum, 2: only weed, 3: both
        plant_types = []
        for idx, val in area_df.iterrows():
            if val[1] == 0 and val[2] != 0:
                plant_type = 2
            elif val[1] != 0 and val[2] == 0:
                plant_type = 1
            else:
                plant_type = 3
            plant_types.append(plant_type)
        area_df["plant_type"] = plant_types
        return area_df[["filename", "plant_type"]]

    def create_stratify_table(self):
        num_plants = self.get_number_of_plants()
        plant_types = self.get_plant_types()
        df = pd.concat([num_plants, plant_types], axis=1)
        df = df.loc[:,~df.columns.duplicated()].copy() # remove duplicate columns
        self.df = df
        self.df.to_csv("data/stratify_table.csv")
        return

    def stratify(self):
        trainval, test = train_test_split(self.df, test_size=0.0769, random_state=42, stratify=self.df[["plant_type", "num_plants"]])
        train, val = train_test_split(trainval, test_size=0.25, random_state=42, stratify=trainval[["plant_type", "num_plants"]])
        train = train.sort_values(by="filename")
        val = val.sort_values(by="filename")
        test = test.sort_values(by="filename")
        print(f"training samples: {len(train)}")
        print(f"validation samples: {len(val)}")
        print(f"test samples: {len(test)}")
        train["filename"].to_csv(f"data/splits/train.csv", index=False)
        val["filename"].to_csv(f"data/splits/val.csv", index=False)
        test["filename"].to_csv(f"data/splits/test.csv", index=False)


if __name__ == "__main__":
    img_path = Path("./data/gt")
    data_ls = sorted(list(img_path.glob("*.png")))
    data_ls = [img_path.stem for img_path in data_ls]

    p = PatchStratifier(fnames=data_ls)
    p.create_stratify_table()
    p.stratify()
