import logging

import cv2
import numpy as np
import onnxruntime as ort
from onnx import ModelProto
from onnxconverter_common.float16 import convert_float_to_float16
from onnxmltools.utils import load_model


CARD_NAMES = [
    "empty",
    "archer_queen",
    "archers",
    "baby_dragon",
    "arrows",
    "balloon",
    "bandit",
    "barbarians",
    "barbarian_barrel",
    "bats",
    "inferno_dragon",
    "clone",
    "battle_ram",
    "bomber",
    "log",
    "royal_delivery",
    "bowler",
    "graveyard",
    "freeze",
    "cannon_cart",
    "dark_prince",
    "dart_goblin",
    "e_dragon",
    "e_giant",
    "e_spirit",
    "lightning",
    "tornado",
    "earthquake",
    "fireball",
    "rocket",
    "e_wizard",
    "elite_barbarians",
    "elixir_golem",
    "executioner",
    "firecracker",
    "fire_spirit",
    "fisherman",
    "flying_machine",
    "giant_regular",
    "giant_skeleton",
    "goblin_gang",
    "goblin_giant",
    "goblins",
    "golden_knight",
    "golem",
    "guards",
    "healer",
    "heal_spirit",
    "hog_rider",
    "hunter",
    "ice_golem",
    "ice_spirit",
    "ice_wizard",
    "goblin_demolisher",
    "knight",
    "lava_hound",
    "little_prince",
    "lumberjack",
    "magic_archer",
    "mega_knight",
    "mega_minion",
    "mighty_miner",
    "miner",
    "mini_pekka",
    "minion_horde",
    "minions",
    "monk",
    "mother_witch",
    "musketeer",
    "night_witch",
    "pekka",
    "phoenix",
    "_prince",
    "princess",
    "ram_rider",
    "rascals",
    "royal_ghost",
    "royal_giant",
    "royal_hogs",
    "poison",
    "royal_recruits",
    "skeleton_army",
    "skeleton_barrel",
    "skeleton_dragons",
    "skeleton_king",
    "skeletons",
    "sparky",
    "spear_goblins",
    "three_musketeers",
    "valkyrie",
    "wall_breakers",
    "wizard",
    "bomb_tower",
    "cannon_tower",
    "inferno_tower",
    "mortar",
    "tesla",
    "xbow",
    "barbarian_hut",
    "elixir_collector",
    "furnace",
    "goblin_cage",
    "goblin_drill",
    "snowball",
    "goblin_hut",
    "rage",
    "tombstone",
    "witch",
    "zap_spell",
    "zappies",
]


#sort card_names alphabetically
CARD_NAMES = sorted(CARD_NAMES)

def make_card_name_list():
    index2name = {
        0: "_prince",
        1: "archer_queen",
        2: "archers",
        3: "arrows",
        4: "",
        5: "",
        6: "",
        7: "",
        8: "",
        9: "",
        10: "",
        11: "",
        12: "",
        13: "",
        14: "",
        15: "",
        16: "",
        17: "",
        18: "",
        19: "",
        20: "",
        21: "",
        22: "",
        23: "",
        24: "",
        25: "",
        26: "",
        27: "",
        28: "empty",
        29: "",
        30: "",
        31: "",
        32: "",
        33: "",
        34: "",
        35: "",
        36: "",
        37: "giant",
        38: "",
        39: "",
        40: "",
        41: "",
        42: "",
        43: "",
        44: "goblin_hut",
        45: "",
        46: "",
        47: "",
        48: "",
        49: "",
        50: "",
        51: "",
        52: "",
        53: "",
        54: "",
        55: "ice_spirit",
        56: "",
        57: "inferno_dragon",
        58: "",
        59: "",
        60: "",
        61: "",
        62: "",
        63: "",
        64: "",
        65: "",
        66: "",
        67: "",
        68: "",
        69: "",
        70: "",
        71: "",
        72: "",
        73: "",
        74: "",
        75: "",
        76: "",
        77: "",
        78: "",
        79: "",
        80: "",
        81: "",
        82: "",
        83: "",
        84: "",
        85: "",
        86: "",
        87: "",
        88: "",
        89: "",
        90: "",
        91: "",
        92: "",
        93: "",
        94: "",
        95: "",
        96: "snowball",
        97: "",
        98: "",
        99: "",
        100: "",
        101: "",
        102: "",
        103: "",
        104: "",
        105: "",
        106: "",
        107: "",
        108: "",
        109: "",
    }

    pass


class OnnxDetector:
    def __init__(self, model_path, use_gpu=False):
        self.model_path = model_path

        providers = list(
            set(ort.get_available_providers())
            & {"CUDAExecutionProvider" if use_gpu else None, "CPUExecutionProvider"}
        )
        logging.info(f"Using providers: {providers}")

        mdl_in = load_model(model_path)
        mdl: ModelProto = convert_float_to_float16(mdl_in)
        self.sess = ort.InferenceSession(
            mdl.SerializeToString(),
            providers=providers,
        )

        self.output_name = self.sess.get_outputs()[0].name

        input_ = self.sess.get_inputs()[0]
        self.input_name = input_.name
        self.model_height, self.model_width = input_.shape[2:]

    def preprocess(self, x: np.ndarray):
        x = cv2.resize(x, (self.model_width, self.model_height))
        return x

    def fix_bboxes(self, x, width, height, padding):
        x[:, [0, 2]] -= padding[0]
        x[:, [1, 3]] -= padding[2]
        x[..., [0, 2]] *= width / (self.model_width - padding[0] - padding[1])
        x[..., [1, 3]] *= height / (self.model_height - padding[2] - padding[3])
        return x

    def _infer(self, x: np.ndarray):
        """
        x,y,3 -> 1,3,x,y
        """

        if x.dtype == np.uint8:
            x = x.astype(np.float16) / 255.0
        else:
            x = x.astype(np.float16)
        x = np.expand_dims(x.transpose(2, 0, 1), axis=0)
        return self.sess.run([self.output_name], {self.input_name: x})[0]

    def run(self, image):
        raise NotImplementedError


import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# init the model
model_path = r"runs\classify\train16\weights\best.onnx"
use_gpu = True
detector = OnnxDetector(model_path, use_gpu)

# get inputs
val_images_path = r"dataset\val"


def get_random_val_image_path():
    label_folder = random.choice(os.listdir(val_images_path))
    random_image_file = random.choice(
        os.listdir(os.path.join(val_images_path, label_folder))
    )
    random_image_path = os.path.join(val_images_path, label_folder, random_image_file)
    return random_image_path


def convert_image_path_to_numpy_input(image_path):
    def resize_pil_image(image, width, height):
        return image.resize((width, height))

    image = Image.open(image_path)
    image = resize_pil_image(image, detector.model_width, detector.model_height)
    iar = np.array(image)
    return iar


def detect_on_1_image(image_path):
    def graph_all_outputs(data):
        names = list(data.keys())
        values = list(data.values())

        # Create the bar graph
        plt.bar(names, values)

        # Add labels and title
        plt.xlabel("Names")
        plt.ylabel("Values")
        plt.title("Bar Graph of Names and Values")

        # Show the plot
        plt.show()

    def show_image(np_iar):
        plt.imshow(np_iar)
        plt.axis("off")  # Turn off axis labels
        plt.show()

    def parse_output(output):
        cardName2prob = {}
        for i in range(len(output)):
            cardName2prob[CARD_NAMES[i]] = output[i]

        # sort cardName2prob by prob
        cardName2prob = dict(
            sorted(cardName2prob.items(), key=lambda item: item[1], reverse=True)
        )

        # graph_all_outputs(cardName2prob)

        # for card, prob in cardName2prob.items():
        #     print(f"{card}: {prob}")

        # get the highest card name, highest card index, and highest card prob
        highest_card_name = list(cardName2prob.keys())[0]
        highest_card_index = CARD_NAMES.index(highest_card_name)
        highest_card_prob = cardName2prob[highest_card_name]

        return highest_card_name, highest_card_index, highest_card_prob

    np_iar = convert_image_path_to_numpy_input(image_path)
    show_image(np_iar)
    model_output = detector._infer(np_iar)[0]
    highest_card_name, highest_card_index, highest_card_prob = parse_output(
        model_output
    )
    print(
        "{:^20} : {:^3} : {:%}".format(
            highest_card_name, highest_card_index, highest_card_prob
        )
    )


def get_all_images_for_label(label):
    label_folder = os.path.join(val_images_path, label)
    return [os.path.join(label_folder, image) for image in os.listdir(label_folder)]


def check_label_folder(label_folder_name,_1_image_per=False):
    image_paths = get_all_images_for_label(label_folder_name)
    for image_path in image_paths:
        detect_on_1_image(image_path)
        if _1_image_per:
            break


def check_all_label_folders():
    label_folders = os.listdir(val_images_path)
    random.shuffle(label_folders)
    for label_folder in label_folders:
        check_label_folder(label_folder,_1_image_per=True)


check_all_label_folders()
