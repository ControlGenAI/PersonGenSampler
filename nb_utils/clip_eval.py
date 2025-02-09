"""Based on https://github.com/rinongal/textual_inversion/blob/main/evaluation/clip_eval.py"""
import os
from typing import List, Optional, Dict, Tuple

import clip

import numpy as np

import torch
import torch.backends.cuda

from torchvision import transforms

import PIL
from PIL.Image import Image

from .eval_sets import evaluation_sets
from .utils import prompt_to_empty_prompts
from .images_viewer import MultifolderViewer

from .face_align.cosface.net import Sphere

from .face_align.PIPNet.alignment.alignment import norm_crop
from .face_align.PIPNet.alignment.landmarks import get_5_from_98
from .face_align.PIPNet.lib.tools import get_lmk_model, demo_image


def get_prompt_info(config):
    real_data_dir = config.get('test_data_dir', config.get('train_data_dir', config.get('instance_data_dir', None)))
    return real_data_dir, config['placeholder_token'], config['class_name']


class CLIPExtractor:
    def __init__(self, device, model='ViT-B/32') -> None:
        self.device = device
        self.model, preprocess = clip.load(model, device=self.device)

        self.preprocess = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[-1.0, -1.0, -1.0],
                    std=[2.0, 2.0, 2.0])
            ] +                               # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1]
            preprocess.transforms[:2] +  # to match CLIP input scale assumptions
            preprocess.transforms[4:]    # + skip convert PIL to tensor
        )
        for transform in self.preprocess.transforms:
            if isinstance(transform, transforms.Resize):
                transform.antialias = False

    @staticmethod
    def _images_to_tensor(images: List[np.ndarray]) -> torch.Tensor:
        """
        Convert list of numpy.ndarray images with numpy.uint8 encoding ([0, 255] range)
            to torch.Tensor with torch.float32 encoding ([-1.0, 1.0] range)
        """
        images = np.stack(images)
        images = torch.from_numpy(np.transpose(images, axes=(0, 3, 1, 2)))
        return torch.clamp(images / 127.5 - 1.0, min=-1.0, max=1.0)

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def _encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def _encode_images(self, images: List[np.ndarray]) -> torch.Tensor:
        images = self._images_to_tensor(images)
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:
        tokens = clip.tokenize(text).to(self.device)
        text_features = self._encode_text(tokens)

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: List[np.ndarray], norm: bool = True) -> torch.Tensor:
        image_features = self._encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features


class DINOExtractor:
    def __init__(self, device, model='dinov2_vits14') -> None:
        self.device = device
        self.dino_model = torch.hub.load('facebookresearch/dinov2', model).to(self.device)

        self.preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    @torch.no_grad()
    def _encode_images(self, images: List[Image]) -> torch.Tensor:
        images = torch.stack([self.preprocess(image) for image in images])
        images = images.to(self.device)
        return self.dino_model(images)

    def get_image_features(self, img: List[Image], norm: bool = True) -> torch.Tensor:
        image_features = self._encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features


class IdentityEvaluator:
    def __init__(
            self, device: torch.device, pip_weights_path, id_weights_path,
            align_mode: str = 'ffhq', img_size: int = 512
    ):
        self.align_mode = align_mode
        self.img_size = img_size

        ''' face alignment '''
        self.net, self.detector = get_lmk_model(pip_weights_path)
        self.net.to(device).eval()
        self.trans_arr_to_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])
        print('[IdentityEvaluator] alignment model loaded')

        ''' face recognition '''
        self.id_model = self._load_fr_net(id_weights_path).to(device)
        self.trans_matrix = torch.tensor([[
                [1.07695457, -0.03625215, -1.56352194 / 512],
                [0.03625215, 1.07695457, -5.32134629 / 512],
            ]], device=device, dtype=torch.float32
        )
        print('[IdentityEvaluator] face recognition model loaded')

    @torch.no_grad()
    def __call__(self, ori1: torch.Tensor, ori2: torch.Tensor):
        n1, _, _, _ = ori1.shape
        crops, has_faces = self._check_lmk_box_for_tensor(torch.cat([ori1, ori2], dim=0))
        crop1 = crops[:n1].to(self.trans_matrix.device)
        crop2 = crops[n1:].to(self.trans_matrix.device)
        cos_sim_mx = self._img_to_img_id_sim(crop1, crop2)

        cos_sim_mx[~has_faces[:n1]] = -1.0
        cos_sim_mx[:, ~has_faces[n1:]] = -1.0

        proper_cos_similarity = cos_sim_mx[has_faces[:n1]][:, has_faces[n1:]]
        if proper_cos_similarity.numel() == 0:
            cos_similarity = -1.0
        else:
            cos_similarity = proper_cos_similarity.mean().item()

        return {
            'cos_similarity': cos_similarity,
            'cos_similarity_mx': cos_sim_mx.cpu().numpy().tolist(),

            "num_has_face_1": torch.sum(has_faces[:n1]).item(),
            "num_has_face_2": torch.sum(has_faces[n1:]).item(),

            "num_no_face_1": torch.sum(~has_faces[:n1]).item(),
            "num_no_face_2": torch.sum(~has_faces[n1:]).item(),

            "has_faces_mx_1": has_faces[:n1].cpu().numpy().tolist(),
            "has_faces_mx_2": has_faces[n1:].cpu().numpy().tolist(),
        }

    def _check_lmk_box_for_tensor(self, img_tensor: torch.Tensor):
        img_arr = ((img_tensor + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        img_cropped, has_faces = [], []

        for img in img_arr:
            cropped, success = self._check_lmk_box_for_one_image(img)
            img_cropped.append(self.trans_arr_to_tensor(cropped).to(img_tensor.device))
            has_faces.append(success)

        img_cropped = torch.stack(img_cropped, dim=0)
        has_faces = torch.tensor(has_faces)

        return img_cropped, has_faces

    def _check_lmk_box_for_one_image(self, img_arr: np.ndarray):
        full_img = img_arr.astype(np.uint8)
        lmks = demo_image(full_img, self.net, self.detector)
        if len(lmks) > 0:
            lmk = get_5_from_98(lmks[0])
            cropped_img = norm_crop(full_img, lmk, self.img_size, mode=self.align_mode, borderValue=0.0)
            return cropped_img, True
        else:
            return full_img, False

    def _img_to_img_id_sim(self, face1: torch.Tensor, face2: torch.Tensor):
        import torch.nn.functional as F
        n1, c, h, w = face1.shape
        n2, c, h, w = face2.shape
        if n1 < 1 or n2 < 1:
            return 0, 0, 0
        faces = torch.cat([face1, face2], dim=0)

        ''' align to insightface '''
        M = self.trans_matrix.repeat(faces.size()[0], 1, 1)  # to (B,2,3)
        # noinspection PyTypeChecker
        grid = F.affine_grid(M, size=faces.size(), align_corners=True)  # 得到grid 用于grid sample
        faces = F.grid_sample(faces, grid, align_corners=True, mode="bilinear", padding_mode="zeros")  # warp affine
        faces = F.interpolate(faces, size=112, mode="bilinear", align_corners=True)

        feats = self.id_model(faces)
        feats = F.normalize(feats, dim=-1, p=2)
        feat1 = feats[:n1]
        feat2 = feats[n1:]

        cos_sim = F.cosine_similarity(feat1[:, None, :], feat2[None, :, :], dim=-1)  # (n1,n2)

        return cos_sim

    @staticmethod
    def _load_fr_net(weights_path):
        id_model = Sphere()

        weights_path = weights_path or os.path.join(
            os.path.dirname(__file__), 'face_align', 'cosface', 'net_sphere20_data_vggface2_acc_9955.pth'
        )
        if not os.path.exists(weights_path):
            print('Downloading Sphere Net weights...')
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            torch.hub.download_url_to_file('https://nxt.2a2i.org/index.php/s/qdBrjgzw44pxfxN/download', weights_path)
        weights = torch.load(weights_path, weights_only=True)
        id_model.load_state_dict(weights)
        id_model.requires_grad_(False)
        id_model.eval()

        return id_model


class ExpEvaluator:
    def __init__(self, device):
        self.device = device
        self.clip_extractor = CLIPExtractor(device=device)
        self.dino_extractor = DINOExtractor(device=device)

    @staticmethod
    def _calc_similarity(left_features, right_features):
        similarity_matrix = left_features @ right_features.T
        return similarity_matrix.mean().item(), similarity_matrix.cpu().numpy().tolist()

    @torch.no_grad()
    def _get_image_features(self, images: List[np.ndarray], resolution=None):
        # noinspection PyPep8Naming
        PIL_images = [PIL.Image.fromarray(image) for image in images]

        if resolution is not None:
            images = [
                np.array(image.resize((resolution, resolution), resample=PIL.Image.Resampling.BICUBIC))
                for image in PIL_images
            ]

        images_features = self.clip_extractor.get_image_features(images)
        dino_images_features = self.dino_extractor.get_image_features(PIL_images)

        return images_features, dino_images_features

    @torch.no_grad()
    def __call__(self, viewer: MultifolderViewer, config):
        results = {
            'image_similarities': {},
            'image_similarities_mx': {},

            'dino_image_similarities': {},
            'dino_image_similarities_mx': {},

            'text_similarities': {},
            'text_similarities_mx': {},

            'text_similarities_with_class': {},
            'text_similarities_mx_with_class': {},

            'text_similarities_prompt': {},
            'text_similarities_with_class_prompt': {},
        }
        # Metrics for the real images
        real_data_dir, placeholder_token, class_name = get_prompt_info(config)

        real_image_paths, real_images = viewer.load_images(real_data_dir)

        real_images_features, dino_real_images_features = self._get_image_features(real_images, config['resolution'])

        results['real_image_similarity'], results['real_image_similarity_mx'] = (
            self._calc_similarity(real_images_features, real_images_features)
        )
        results['dino_real_image_similarity'], results['dino_real_image_similarity_mx'] = (
            self._calc_similarity(dino_real_images_features, dino_real_images_features)
        )

        for label, images in viewer.images.items():
            if len(images) == 0:
                print(f'No images found for the label {label}')
                continue

            # Visual metrics for the images for the target prompt (label)
            images_features, dino_images_features = self._get_image_features(images)

            results['image_similarities'][label], results['image_similarities_mx'][label] = (
                self._calc_similarity(real_images_features, images_features)
            )
            results['dino_image_similarities'][label], results['dino_image_similarities_mx'][label] = (
                self._calc_similarity(dino_real_images_features, dino_images_features)
            )

            # Textual metrics for the images for the target prompt (label)
            empty_label, empty_label_with_class = prompt_to_empty_prompts(label, placeholder_token, class_name)

            empty_label_features = self.clip_extractor.get_text_features(empty_label)
            empty_label_with_class_features = self.clip_extractor.get_text_features(empty_label_with_class)

            results['text_similarities'][label], results['text_similarities_mx'][label] = (
                self._calc_similarity(empty_label_features, images_features)
            )
            results['text_similarities_prompt'][label] = empty_label

            results['text_similarities_with_class'][label], results['text_similarities_mx_with_class'][label] = (
                self._calc_similarity(empty_label_with_class_features, images_features)
            )
            results['text_similarities_with_class_prompt'][label] = empty_label_with_class

        return results


class ExpEvaluatorWithID(ExpEvaluator):
    def __init__(self, device, pip_weights_path=None, id_weights_path=None) -> None:
        super().__init__(device)
        self.id_evaluator = IdentityEvaluator(
            device, pip_weights_path, id_weights_path, align_mode='ffhq', img_size=512
        )

        self.id_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __call__(self, viewer: MultifolderViewer, config):
        results = super().__call__(viewer, config)
        results |= {
            'no_faces': {},
            'has_faces': {},
            'has_faces_mx': {},
            'id_similarities': {},
            'id_similarities_mx': {},
        }

        real_data_dir, placeholder_token, class_name = get_prompt_info(config)
        real_image_paths, _ = viewer.load_images(real_data_dir)
        real_images = torch.cat([
            self.id_transform(PIL.Image.open(path).convert('RGB')).unsqueeze(0) for path in real_image_paths
        ], dim=0)

        real_id_results = self.id_evaluator(real_images, real_images)
        results['real_id_similarity'], results['real_id_similarity_mx'] = (
            real_id_results['cos_similarity'], real_id_results['cos_similarity_mx']
        )
        results['real_has_faces_mx'], results['real_no_faces'], results['real_has_faces'] = (
            real_id_results['has_faces_mx_2'],
            real_id_results['num_no_face_2'] / real_images.shape[0],
            real_id_results['num_has_face_2'] / real_images.shape[0]
        )

        for label, images_paths in viewer.images_paths.items():
            if len(images_paths) == 0:
                print(f'No images found for the label {label}')
                continue

            images = torch.cat([
                self.id_transform(PIL.Image.open(path).convert('RGB')).unsqueeze(0) for path in images_paths
            ], dim=0)

            id_results = self.id_evaluator(real_images, images)
            results['id_similarities'][label], results['id_similarities_mx'][label] = (
                id_results['cos_similarity'], id_results['cos_similarity_mx']
            )
            results['has_faces_mx'][label], results['no_faces'][label], results['has_faces'][label] = (
                id_results['has_faces_mx_2'],
                id_results['num_no_face_2'] / images.shape[0],
                id_results['num_has_face_2'] / images.shape[0]
            )

        return results


def narrow_similarities(
        similarities: Dict[str, float], prompts: List[str], holder: Optional[str], verbose: bool = False
) -> Tuple[Optional[float], Optional[float]]:
    """ Takes dictionary of similarities (prompt -> value) and aggregates only values from the list of prompts
    :param similarities: dictionary of text similarities
    :param prompts: subset of templated keys (i.e. "a photo of {0}") from dictionary of text similarities
    :param holder: target holder (i.e. placeholder or placeholder with class name)
    :param verbose: if True then print existing errors
    :return: mean and std text similarity over list of selected prompts. If some prompts are missing then returns None
    """
    try:
        result = []
        for prompt in prompts:
            result.append(similarities[prompt.format(holder)])

        return float(np.mean(result)), float(np.std(result))
    except Exception as ex:
        if verbose:
            print(f'Exception in narrow_similarities: {ex}')
        return None, None


def aggregate_similarities(data: dict) -> dict:
    """ Calculate necessary statistics over raw evaluation results
    :param dict data: output of ExpViewer._evaluate
    :return: dict with added aggregated stats
    """
    real_data_dir, placeholder_token, class_name = get_prompt_info(data['config'])

    result = {
        'dataset': os.path.basename(real_data_dir),
    }

    base_holder, base_holder_with_class = placeholder_token, f'{placeholder_token} {class_name}'
    specs = []
    for set_name, set_prompts in evaluation_sets.items():
        base_metrics_names = [
            '_'.join([set_name, 'text_similarity']).lstrip('_'),
            '_'.join([set_name, 'image_similarity']).lstrip('_'),
            '_'.join([set_name, 'with_class', 'text_similarity']).lstrip('_'),
            '_'.join([set_name, 'with_class', 'image_similarity']).lstrip('_'),

            '_'.join([set_name, 'no_face']).lstrip('_'),
            '_'.join([set_name, 'has_face']).lstrip('_'),
            '_'.join([set_name, 'id_similarity']).lstrip('_'),
            '_'.join([set_name, 'with_class', 'no_face']).lstrip('_'),
            '_'.join([set_name, 'with_class', 'has_face']).lstrip('_'),
            '_'.join([set_name, 'with_class', 'id_similarity']).lstrip('_'),
        ]

        for metric_prefix in ['', 'dino_', 'masked_', 'masked_dino_']:
            if metric_prefix + 'image_similarities' in data:
                specs += [
                    (metric_prefix + base_metrics_names[1], metric_prefix + 'image_similarities', set_prompts, base_holder),
                    (metric_prefix + base_metrics_names[3], metric_prefix + 'image_similarities', set_prompts, base_holder_with_class)
                ]

        if 'text_similarities' in data:
            specs += [
                (base_metrics_names[0], 'text_similarities', set_prompts, base_holder),
                (base_metrics_names[2], 'text_similarities', set_prompts, base_holder_with_class),

                (base_metrics_names[0] + '_with_class', 'text_similarities_with_class', set_prompts, base_holder),
                (base_metrics_names[2] + '_with_class', 'text_similarities_with_class', set_prompts, base_holder_with_class),
            ]

        if 'id_similarities' in data:
            specs += [
                (base_metrics_names[4], 'no_faces', set_prompts, base_holder),
                (base_metrics_names[5], 'has_faces', set_prompts, base_holder),
                (base_metrics_names[6], 'id_similarities', set_prompts, base_holder),
                (base_metrics_names[7], 'no_faces', set_prompts, base_holder_with_class),
                (base_metrics_names[8], 'has_faces', set_prompts, base_holder_with_class),
                (base_metrics_names[9], 'id_similarities', set_prompts, base_holder_with_class),
            ]

    for key, similarities, prompts, holder in specs:
        result[key], result[key + '_std'] = narrow_similarities(
            data[similarities], prompts, holder, verbose=False
        )

    return result
