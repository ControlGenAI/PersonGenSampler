import os
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose, Resize, Normalize, InterpolationMode, ToTensor, RandomCrop, RandomHorizontalFlip, CenterCrop
)

BICUBIC = InterpolationMode.BICUBIC


class ImageDataset(Dataset):
    def __init__(
        self,
        train_data_dir,
        resolution=512,
        rand=False,
        repeats=100,
    ):
        self.train_data_dir = train_data_dir
        self.data_fnames = [
            os.path.join(r, f) for r, d, fs in os.walk(self.train_data_dir)
            for f in fs
        ]
        self.num_images = len(self.data_fnames)
        self._length = self.num_images * repeats

        self.resolution = resolution
        self.rand = rand

    def process_img(self, img_file):
        if isinstance(img_file, str):
            img_file = [img_file]
        input_img = []
        for img in img_file:
            image = Image.open(img).convert('RGB')
            w, h = image.size
            crop = min(w, h)
            if self.rand:
                image = Resize(560, interpolation=InterpolationMode.BILINEAR, antialias=True)(image)
                image = RandomCrop(self.resolution)(image)
                image = RandomHorizontalFlip()(image)
            else:
                image = image.crop(((w - crop) // 2, (h - crop) // 2, (w + crop) // 2, (h + crop) // 2))
            image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
            input_img.append(ToTensor()(image))
        input_img = torch.cat(input_img)

        return input_img

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        image_file = self.data_fnames[index % self.num_images]
        input_img = self.process_img(image_file)

        example['image_path'] = image_file
        example['image'] = input_img

        return example


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = Compose(
            [
                Resize(size, interpolation=InterpolationMode.BILINEAR, antialias=True),
                CenterCrop(size) if center_crop else RandomCrop(size),
                ToTensor(),
                Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch
