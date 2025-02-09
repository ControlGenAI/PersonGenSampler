import os
import traceback
import contextlib
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Optional, Union, List, Tuple

import glob

from natsort import natsorted

import tqdm
import tqdm.autonotebook

import numpy as np

import ipywidgets
from ipyfilechooser import FileChooser
from IPython.display import display, clear_output, Latex

import matplotlib.pyplot as plt

from .configs import _LOAD_IMAGE_BACKEND
from .utils import prepare_axes, _safe_get, RegexFilter


def create_buttons_grid(grid_size: Tuple[int, int]):
    buttons_grid = ipywidgets.VBox([
        ipywidgets.HBox([
            ipywidgets.Button(
                description=f'({idx}, {jdx}) -> {idx * grid_size[1] + jdx}',
                layout=ipywidgets.Layout(width='{0}%'.format(100 / grid_size[1]))
            )
            for jdx in range(grid_size[1])
        ]) for idx in range(grid_size[0])
    ])

    buttons = [
        buttons_grid.children[idx].children[jdx]
        for idx in range(grid_size[0])
        for jdx in range(grid_size[1])
    ]
    return buttons_grid, buttons


def _save_figure_callback(fig: plt.Figure, chooser: FileChooser):
    save_path = os.path.join(chooser.selected)
    if not os.path.exists(save_path):
        fig.savefig(save_path, bbox_inches='tight', dpi=600, pad_inches=0)
    else:
        print(f'{save_path} already exists')


class MultifolderViewer:
    # noinspection PyUnresolvedReferences
    def __init__(
            self, directories: Union[List[str], str], labels: Optional[List[str]] = None,
            lazy_load: bool = True, info: dict = None, set_name: str = 'medium',
            filter_fn: RegexFilter = None, verbose: bool = True
    ):
        """
        :param Union[List[str], str] directories: list of directories or glob template to display
        :param Optional[List[str]] labels: list of labels for each directory. If None then use directory's basename
        :param bool lazy_load: Whether to load all images in class constructor
        :param dict info: Data for images labels and titles
        :param str set_name: Prefix for all keys from info of image labels and titles
        :param RegexFilter filter_fn: Filter directories w.r.t. their labels.
            Use a .normalize method to determine order of directories
        :param bool verbose: whether to log progress bar for the image loading
        """
        self.info = info
        self.set_name = set_name
        if isinstance(directories, str):
            assert labels is None
            directories = sorted([
                path
                for path in glob.glob(os.path.join(directories, '*'))
                if os.path.isdir(path)
            ])

        if labels is None:
            labels = [os.path.basename(_) for _ in directories]

        assert len(labels) == len(directories)

        if filter_fn is not None:
            filtered = [(directory, label) for directory, label in zip(directories, labels) if filter_fn(label)]
            directories, labels = zip(*filtered) if len(filtered) else ([], [])
            sort_fn: Callable = lambda args: filter_fn.normalize(args[1])
        else:
            sort_fn: Callable = lambda args: args[1]

        if len(labels):
            directories, labels = zip(*natsorted(zip(directories, labels), key=sort_fn))

        self.labels = labels
        self.directories = directories
        self._label_to_directory = dict(zip(self.labels, self.directories))

        self.images = {}
        self.images_paths = {}

        if not lazy_load:
            with ThreadPool(processes=4) as pool:
                def _load_wrapper(args):
                    _label, _directory = args
                    return _label, self.load_images(_directory)

                for label, (paths, images) in tqdm.tqdm(
                        pool.imap(_load_wrapper, zip(self.labels, self.directories)),
                        total=len(self.directories), disable=not verbose
                ):
                    self.images_paths[label], self.images[label] = paths, images

    @staticmethod
    def load_images(directory: str) -> Tuple[List[str], List[np.ndarray]]:
        """
        :param str directory: target path to the folder with images
        :return: list of paths and list of corresponding images
        """
        paths = sorted(glob.glob(os.path.join(directory, '*')))
        paths = [_ for _ in paths if not os.path.isdir(_)]
        try:
            images = [_LOAD_IMAGE_BACKEND(path) for path in paths]
        except Exception as ex:
            print(ex, traceback.format_exc())
            paths, images = [], []

        return paths, images

    def _show_random(self, output, ncolumns=3, save_path=None):
        output = output or contextlib.nullcontext()

        def _show(_):
            with output:
                clear_output(wait=True)

                grid_size = np.array([(len(self.directories) - 1) // ncolumns + 1, ncolumns])
                fig, axes = plt.subplots(*grid_size, figsize=2 * grid_size[::-1])
                prepare_axes(axes)

                for ax, label in tqdm.autonotebook.tqdm(
                        zip(axes.reshape(-1), self.labels), total=len(self.labels), leave=False
                ):
                    if label not in self.images:
                        directory = self._label_to_directory[label]
                        self.images_paths[label], self.images[label] = self.load_images(directory)

                    images = self.images[label]

                    if len(images) == 0:
                        continue

                    [idx] = np.random.choice(len(images), 1)
                    ax.imshow(images[idx])

                    if self.info is not None:
                        top_label, bottom_label = _get_random_image_annotations(self.info, label, idx)

                        ax.annotate(
                            top_label, xy=(0.01, 0.95),
                            xycoords='axes fraction', fontsize=6,
                            bbox=dict(boxstyle="round,pad=0.,rounding_size=0.01", alpha=0.5, color='w')
                        )
                        ax.annotate(
                            label, xy=(0.01, 0.885),
                            xycoords='axes fraction', fontsize=6,
                            bbox=dict(boxstyle="round,pad=0.,rounding_size=0.01", alpha=0.5, color='w')
                        )
                        ax.annotate(
                            os.path.basename(self.images_paths[label][idx]), xy=(0.01, 0.079),
                            xycoords='axes fraction', fontsize=6,
                            bbox=dict(boxstyle="round,pad=0.,rounding_size=0.01", alpha=0.5, color='w')
                        )
                        ax.annotate(
                            bottom_label, xy=(0.01, 0.017),
                            xycoords='axes fraction', fontsize=6,
                            bbox=dict(boxstyle="round,pad=0.,rounding_size=0.01", alpha=0.5, color='w')
                        )
                    else:
                        ax.annotate(
                            label, xy=(0.04, 0.9),
                            xycoords='axes fraction', fontsize=8,
                            bbox=dict(boxstyle="round", alpha=0.7, color='w')
                        )

                if self.info is not None:
                    subtitle = _get_random_figure_title(self.info, set_name=self.set_name)
                    fig.suptitle(subtitle, fontsize=14, y=0.95)
                    display(Latex(subtitle))
                fig.subplots_adjust(wspace=0, hspace=0)
                fig.patch.set_visible(False)

                fc = FileChooser('./', select_desc='Select path to save image')
                fc.default_filename = '*.jpg'

                if save_path is not None:
                    fig.savefig(save_path, bbox_inches='tight', dpi=600, pad_inches=0)
                    fig.clf()
                    plt.cla()
                    plt.clf()
                else:
                    fc.register_callback(partial(_save_figure_callback, fig=fig))
                    display(fc)
                    plt.show()

        return _show

    def _show_class(self, output, label, ncolumns, save_path=None):
        output = output or contextlib.nullcontext()

        def _show(_):
            with output:
                clear_output(wait=True)

                if label not in self.images:
                    directory = self._label_to_directory[label]
                    self.images_paths[label], self.images[label] = self.load_images(directory)

                images = self.images[label]
                paths = self.images_paths[label]

                if len(images) == 0:
                    print(f'{label} is empty')
                    return

                grid_size = np.array([(len(images) - 1) // ncolumns + 1, ncolumns])
                fig, axes = plt.subplots(*grid_size, figsize=2 * grid_size[::-1])
                prepare_axes(axes)

                print(os.path.split(paths[0])[0])
                folder_name = os.path.normpath(paths[0]).split(os.path.sep)[-2]

                if self.info is not None:
                    title = _get_class_figure_title(self.info, folder_name, set_name=self.set_name)
                else:
                    title = label

                for idx, (ax, image, path) in enumerate(zip(axes.reshape(-1), images, paths)):
                    image_name = os.path.basename(path)

                    if self.info is not None:
                        image_title = _get_image_title(self.info, folder_name, image_name, idx)
                    else:
                        image_title = image_name

                    ax.imshow(image)
                    ax.annotate(
                        image_title, xy=(0.025, 0.95),
                        xycoords='axes fraction', fontsize=5,
                        bbox=dict(boxstyle="round", alpha=0.5, color='w')
                    )

                fig.suptitle(title, y=1.2)
                fig.subplots_adjust(wspace=0, hspace=0)

                fc = FileChooser('./', select_desc='Select path to save image')
                fc.default_filename = '*.jpg'

                if save_path is not None:
                    fig.savefig(save_path, bbox_inches='tight', dpi=600, pad_inches=0)
                    fig.clf()
                    plt.cla()
                    plt.clf()
                else:
                    fc.register_callback(partial(_save_figure_callback, fig=fig))
                    display(fc)
                    plt.show()

        return _show

    def view(self, ncolumns: int = 3) -> ipywidgets.Widget:
        """Draw widget (in form of buttons grid) where each button shows images from the corresponding folder
        :param int ncolumns: number of columns in this buttons grid
        :return:
        """
        grid_size = np.array([(len(self.directories) - 1) // ncolumns + 1, ncolumns])

        output = ipywidgets.Output()

        rnd_button = ipywidgets.Button(description='Random Images', layout=ipywidgets.Layout(width='auto'))
        rnd_button.on_click(self._show_random(output, ncolumns=5))

        buttons_grid, buttons = create_buttons_grid(grid_size)
        for button, label in zip(buttons, self.labels):
            button.description = label
            button.on_click(self._show_class(output, label, ncolumns=ncolumns))

        selector = ipywidgets.VBox([
            rnd_button,
            buttons_grid,
            output
        ])

        return selector


def _get_random_figure_title(info, set_name: str):
    title_values = []
    for key in [
        'real_image_similarity', 'image_similarity', 'with_class_image_similarity',
        f'{set_name}_image_similarity', f'{set_name}_with_class_image_similarity',

        f'{set_name}_text_similarity', f'{set_name}_with_class_text_similarity',
        f'{set_name}_text_similarity_with_class', f'{set_name}_with_class_text_similarity_with_class',

        f'real_id_similarity', f'{set_name}_id_similarity', f'{set_name}_with_class_id_similarity',
        f'{set_name}_has_face', f'{set_name}_with_class_has_face',
    ]:
        value = _safe_get(key, info)
        title_values.append('-' if value is None else '${0:.3f}$'.format(value))
    title = (
        '$IS^{{R}}$/$IS^{{1}}$/$IS^{{1}}_{{wc}}$/$IS$/$IS_{{wc}}$: {}/{}/{}/{}/{}\n'
        '$TS$/$TS_{{wc}}$/$TS^{{wc}}$/$TS_{{wc}}^{{wc}}$: {}/{}/{}/{}\n'
        '$ID^{{R}}$/$ID$/$ID_{{wc}}$/$DT$/$DT_{{wc}}$: {}/{}/{}/{}/{}'
    ).format(
        *title_values
    )

    return title


def _get_random_image_annotations(info, prompt: str, image_idx: int):
    top_values = []
    for key in [
        ('image_similarities', prompt), ('id_similarities', prompt),
        ('text_similarities', prompt), ('text_similarities_with_class', prompt)
    ]:
        value = _safe_get(key, info)
        top_values.append('-' if value is None else '{0:.2f}'.format(value))
    top_label = '$IS$/$ID$: {}/{}, $TS$/$TS^{{wc}}$: {}/{}'.format(*top_values)

    bottom_values = []
    value = _safe_get(('image_similarities_mx', prompt), info)
    bottom_values.append(
        '-' if value is None else '{0:.2f}'.format(np.mean(np.array(value)[:, image_idx]))
    )

    value = _safe_get(('id_similarities_mx', prompt), info)
    bottom_values.append(
        '-' if value is None else '{0:.2f}'.format(np.mean(np.array(value)[:, image_idx]))
    )

    value = _safe_get(('text_similarities_mx', prompt), info)
    bottom_values.append('-' if value is None else '{0:.2f}'.format(value[0][image_idx]))

    value = _safe_get(('text_similarities_mx_with_class', prompt), info)
    bottom_values.append('-' if value is None else '{0:.2f}'.format(value[0][image_idx]))

    bottom_label = '$IS$/$ID$: {}/{}, $TS$/$TS^{{wc}}$: {}/{}'.format(*bottom_values)

    return top_label, bottom_label


def _get_class_figure_title(info, prompt: str, set_name: str):
    title_values = []
    for key in [
        'real_image_similarity', 'image_similarity', 'with_class_image_similarity_',
        f'{set_name}_image_similarity', f'{set_name}_with_class_image_similarity',

        f'{set_name}_text_similarity', f'{set_name}_with_class_text_similarity',
        f'{set_name}_text_similarity_with_class', f'{set_name}_with_class_text_similarity_with_class',

        f'real_id_similarity', f'{set_name}_id_similarity', f'{set_name}_with_class_id_similarity',
        f'{set_name}_has_face', f'{set_name}_with_class_has_face',

        ('image_similarities', prompt),

        ('text_similarities', prompt), ('text_similarities_with_class', prompt),

        ('id_similarities', prompt), ('has_faces', prompt)
    ]:
        value = _safe_get(key, info)
        title_values.append('-' if value is None else '${0:.3f}$'.format(value))

    title = (
        '$IS^{{R}}$/$IS^{{1}}$/$IS^{{1}}_{{wc}}$/$IS$/$IS_{{wc}}$: {}/{}/{}/{}/{}\n'
        '$TS$/$TS_{{wc}}$/$TS^{{wc}}$/$TS_{{wc}}^{{wc}}$: {}/{}/{}/{}\n'
        '$ID^{{R}}$/$ID$/$ID_{{wc}}$/$DT$/$DT_{{wc}}$: {}/{}/{}/{}/{}\n'
        '$IS$: {}, $TS$/$TS^{{wc}}$: {}/{}, $ID$/$DT$: {}/{} {}'
    ).format(
        *title_values, prompt
    )

    return title


def _get_image_title(info, prompt: str, image_name: str, image_idx: int):
    image_title_values = [image_name]

    value = _safe_get(('image_similarities_mx', prompt), info)
    image_title_values.append(
        '-' if value is None else '{0:.2f}'.format(np.mean(np.array(value)[:, image_idx]))
    )

    value = _safe_get(('id_similarities_mx', prompt), info)
    image_title_values.append(
        '-' if value is None else '{0:.2f}'.format(np.mean(np.array(value)[:, image_idx]))
    )

    value = _safe_get(('text_similarities_mx', prompt), info)
    image_title_values.append('-' if value is None else '{0:.2f}'.format(value[0][image_idx]))

    value = _safe_get(('text_similarities_mx_with_class', prompt), info)
    image_title_values.append('-' if value is None else '{0:.2f}'.format(value[0][image_idx]))

    image_title = '{0}, $IS$/$ID$: {1}/{2}, $TS$/$TS^{{wc}}$: {3}/{4}'.format(*image_title_values)

    return image_title
