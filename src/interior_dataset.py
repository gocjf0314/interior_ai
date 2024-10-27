import os
import random
import json
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from typing import Literal, Tuple
import torch.utils.data as data
import torchvision.transforms.functional as TF

from utils.label_reader import InteriorLabelLeader


def room_caption(room_annotation: str):
    return ""


def object_caption(object_annotation: str):
    return ""


def create_room_prompt(room_style, room_type, room_theme_color):
    pass


def create_object_prompt(obj_style, object_type, obj_theme_color, room_caption):
    pass


def parse_tag(tag: str):
    return tag.replace('_', ' ')


class InteriorDataset(data.Dataset):
    def __init__(
            self,
            dataroot_path: str,
            phase: Literal["train", "test"],
            order: Literal["paired", "unpaired"] = "paired",
            size: Tuple[int, int] = (512, 384),
    ):
        super(InteriorDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size

        # Normalization
        # 데이터 간의 크기, 범위 차이로 인해 생길 수 있는
        # 노이즈 발생과 모델 성능 저하를 방지하기 위함
        self.norm = transforms.Normalize([0.5], [0.5])
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.transform2D = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        # 이미지, 텍스트 데이터 텐서화 모듈
        self.toTensor = transforms.ToTensor()

        # json 으로 만든 라벨 파일 읽기
        # annotation 리스트 설정 후 해당 데이터 파싱
        # annotation_path = os.path.join(dataroot_path, phase, phase + "_annotations.json")
        # with open(annotation_path, "r") as anno_file:
        #     train_anno = json.load(anno_file)

        data_anno_info = InteriorLabelLeader()

        # 학습 데이터 라벨 데이터
        train_anno = data_anno_info.train_annotation

        # 객체 tag 정보 리스트
        object_tag_list = data_anno_info.get_object_all_tags()

        # 방 tag 정보 리스트
        room_tag_list = data_anno_info.get_object_all_tags()

        # 파일 이름과 해당 annotation 내용 matching
        self.annotation_pair = {}
        im_names = []
        obj_names = []
        dataroot_names = []
        for anno_data in train_anno:
            dataroot_names.append(dataroot_path)

            im_names.append(anno_data["room_image"])

            object_anno = dict(anno_data["object"])
            object_filename = object_anno["obj_image"]
            obj_names.append(object_filename)

            # 방 태그 정보
            annotation_str = ""
            room_tags = list(anno_data["room_tag_info"])
            for idx, tag_info in enumerate(room_tags):
                # tag_name = tag_info["tag_name"]
                tag_category = tag_info["tag_category"]
                if idx != 0:
                    annotation_str += ", " + tag_category
                else:
                    annotation_str += tag_category

            self.annotation_pair[object_filename] = {
                "room_anno": parse_tag(annotation_str),
                "object_anno": ""
            }

            # 객체 태그 정보
            annotation_str = ""
            object_tags = list(object_anno["obj_tag_info"])
            for idx, tag_info in enumerate(object_tags):
                # tag_name = tag_info["tag_name"]
                tag_category = tag_info["tag_category"]
                if idx != 0:
                    annotation_str += ", " + tag_category
                else:
                    annotation_str += tag_category
            self.annotation_pair[object_filename]["object_anno"] = parse_tag(annotation_str)

        self.order = order

        if phase == "train":
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")

        with open(filename, "r") as f:
            for line in f.readlines():
                if phase == "train":
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == "paired":
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                obj_names.append(c_name)
                dataroot_names.append(dataroot_path)

        self.im_names = im_names
        self.obj_name = obj_names
        self.dataroot_names = dataroot_names
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)  # Augmentation
        self.clip_processor = CLIPImageProcessor()

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, index):
        # 마스킹 영역에 적용 될 대상 이미지
        obj_name = self.obj_name[index]

        # 원본/정답 이미지
        im_name = self.im_names[index]

        # subject_txt = self.txt_preprocess['train']("shirt")
        # 생성자에서 parsing 된 annotation_pair로 부터 옷 이미지에 대한
        # if obj_name in self.annotation_pair:
        #     object_annotation = self.annotation_pair[obj_name]
        # else:
        #     object_annotation = "shirts"
        object_annotation = self.annotation_pair[obj_name]["object_anno"]
        room_annotation = self.annotation_pair[obj_name]["room_annotation"]

        # 마스킹 영역에 적용할 대상 이미지
        object = Image.open(os.path.join(self.dataroot, self.phase, "objects", obj_name))

        # 모델이 생성한 noise 와 비교할 정답 이미지
        # 리사이징 + 정규화 + 텐서화
        im_pil_big = Image.open(
            os.path.join(self.dataroot, self.phase, "rooms", im_name)
        ).resize((self.width, self.height))
        image = self.transform(im_pil_big)

        # 정답 이미지에서 마스크 할 영역이 마스킹 된 마스크 이미지
        # 모델에 학습 시킬 사이즈로 리사이징 후 tensor 형태로 변환
        # mask 이미지의 첫 번째 채널만 반환
        # (3, w, h) -> (1, w, h)
        mask_im_name = obj_name.split('.')[0] + "_mask" + obj_name.split('.')[-1]
        mask_path = os.path.join(self.dataroot, self.phase, "object_mask", mask_im_name)
        mask = Image.open(mask_path).resize((self.width, self.height))
        mask = self.toTensor(mask)
        mask = mask[:1]

        # # Dense pose 이미지 텐서화
        # densepose_name = im_name
        # densepose_map = Image.open(
        #     os.path.join(self.dataroot, self.phase, "image-densepose", densepose_name)
        # )
        # pose_img = self.toTensor(densepose_map)  # [-1,1]

        # 깊이 이미지 텐서화
        ddepthmap_name = im_name.split('.')[0] + "_depth" + obj_name.split('.')[-1]
        depthmap_path = os.path.join(self.dataroot, self.phase, "image-depthmap", ddepthmap_name)
        depth_map = Image.open(depthmap_path)
        depth_img = self.toTensor(depth_map)  # [-1,1]

        # 학습 데이터일 경우
        # 랜덤 함수에 의해 50% 확률로 증강 효과 적용
        if self.phase == "train":
            # 이미지 수평 반전으로 데이터 방향성을 다양화
            # 마스킹 적용 대상, 원본 이미지, depth map 이미지, 마스크 이미지
            if random.random() > 0.5:
                object = self.flip_transform(object)
                mask = self.flip_transform(mask)
                image = self.flip_transform(image)
                depth_img = self.flip_transform(depth_img)

            # ColorJitter > 밝기, 대비, 채도, 색조 조정 / 모델의 색상 편향을 줄이고 다양성 유도
            # 설정된 color_jitter의 값들로 마스킹 영역에 적용 될 대상과 정답 이미지에 적용
            if random.random() > 0.5:
                color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5, hue=0.5)
                fn_idx, b, c, s, h = transforms.ColorJitter.get_params(color_jitter.brightness, color_jitter.contrast,
                                                                       color_jitter.saturation, color_jitter.hue)

                image = TF.adjust_contrast(image, c)
                image = TF.adjust_brightness(image, b)
                image = TF.adjust_hue(image, h)
                image = TF.adjust_saturation(image, s)

                object = TF.adjust_contrast(object, c)
                object = TF.adjust_brightness(object, b)
                object = TF.adjust_hue(object, h)
                object = TF.adjust_saturation(object, s)

            # scale > 크기 변형, 다양한 크기에 대한 학습 유도
            # 0.8 ~ 1.2 사이의 임의의 값으로 이미지 배율 조정
            # 적용 대상: 정답 이미지, 마스크 이미지, map 이미지
            if random.random() > 0.5:
                def apply_uniform(img, scale):
                    return transforms.functional.affine(
                        img,
                        angle=0,
                        translate=[0, 0],
                        scale=scale,
                        shear=0,
                    )
                scale_val = random.uniform(0.8, 1.2)
                image = apply_uniform(image, scale_val)
                mask = apply_uniform(mask, scale_val)
                depth_img = apply_uniform(depth_img, scale_val)

            # translate > x, y 각각의 축 기준으로 -0.2 ~ 0.2 사이의 랜던 값으로 이미지 위치 이동
            # 대상이 여러 위치에 있는 상황을 만들어 모델 일반성 향상 유도
            if random.random() > 0.5:
                def apply_shifting(img, x, y):
                    return transforms.functional.affine(
                        img,
                        angle=0,
                        translate=[x * img.shape[-1], y * img.shape[-2]],
                        scale=1,
                        shear=0,
                    )
                shift_valx = random.uniform(-0.2, 0.2)
                shift_valy = random.uniform(-0.2, 0.2)
                image = apply_shifting(image, shift_valx, shift_valy)
                mask = apply_shifting(mask, shift_valx, shift_valy)
                depth_img = apply_shifting(depth_img, shift_valx, shift_valy)

        # 텐서화 된 마스크(값 범위 0 ~ 1) 이미지의 마스킹 영역 반전
        mask = 1 - mask

        # CLIP Image-Preprocessor 로 미리 전처리 된 이미지 tensor 값 저장
        object_trim = self.clip_processor(images=object, return_tensors="pt").pixel_values

        # 0.5 기준으로 구분지어 mask 값을 바이너리화
        # 0, 1 로 나누어 mask 영역 확실하게 지정
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # 정답 이미지에 마스크 이미지를 곱하여
        # 원본 이미지에서 마스킹 영역만 유지되고 나머지는 0(검정색)으로 처리 된 이미지(im_mask) 반환
        im_mask = image * mask

        # map 이미지 정규화
        depth_img = self.norm(depth_img)

        # c_name: 마스킹에 적용 될 대상 이미지
        # image: 정답 이미지
        # cloth_trim: 가공된 마스킹에 적용 될 대상 이미지
        # cloth_pure: 텐서화 된 마스킹에 적용 될 대상 이미지
        # inpaint_mask: 원래 마스킹 될 영역으로 다시 적용
        # im_mask: 원본 이미지에서 마스킹 영역만 남기고 나머지는 검정색인 이미지
        # caption: 모델 생성 자체에 적용 될 prompt
        # caption_cloth: 마스킹 대상 이미지와 매칭 되는 caption
        # annotation: 마스킹에 적용 될 대상의 annotation
        # pose_img: depth map 이미지
        result = {
            "obj_name": obj_name,
            "image": image,
            "object": object_trim,
            "object_pure": self.transform(object),
            "inpaint_mask": 1 - mask,
            "im_mask": im_mask,
            "caption": room_caption(room_annotation),
            "caption_object": object_caption(object_annotation),
            "annotation": object_annotation,
            "depth_img": depth_img,
        }

        return result
