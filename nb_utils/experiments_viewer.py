import os
import traceback
from functools import partial
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from typing import Any, Tuple, Callable, Optional, Dict, List

import glob
import tqdm
import tqdm.autonotebook

import regex
from natsort import natsorted

import numpy as np

import ipywidgets
from IPython.display import display


from .utils import _read_config
from .clip_eval import ExpEvaluator
from .images_viewer import create_buttons_grid, MultifolderViewer


def _display_widget(_, output: ipywidgets.Output, widget: MultifolderViewer):
    output.clear_output()
    with output:
        display(widget)


class ExpsViewer:
    EXP_NAME_PATTERN = '[0-9]+-.*-.*'
    SAMPLE_CHECKPOINT_IDX_PATTERN = 'checkpoint-([0-9]+)'
    SAMPLE_SPECS_PATTERN = 'ns([0-9]+)_gs([^_]+)(?:_(.*))?'
    SAMPLES_DIRS_PATTERN = os.path.join('checkpoint-*', 'samples', '*', 'version_0')

    def __class_getitem__(cls, version):
        name = f'{cls.__name__}<{version}>'
        cls_versioned = type(name, cls.__bases__, dict(cls.__dict__))
        cls_versioned.SAMPLES_DIRS_PATTERN = cls_versioned.SAMPLES_DIRS_PATTERN.replace(
            'version_0', f'version_{version}'
        )

        return cls_versioned

    def __init__(
            self, base_path: str, ncolumns: int = 5,
            all_info: Dict[Tuple[str, Tuple[str, str, str]], Any] = None, set_name: str = 'medium',
            exp_filter_fn: Callable[[str], bool] = None, lazy_load: bool = True,
            evaluator: Optional[ExpEvaluator] = None, filter_fn: Callable[[str], bool] = None,
            path_mapping: Optional[dict] = None, checkpoint_filter_fn=None
    ):
        """
        :param base_path: path to folder with experiments
        :param int ncolumns: number of columns in selector over all experiment
        :param all_info: experiments statistics to draw correct captions for images and figures
        :param set_name: target prompts set for displayed metrics
        :param exp_filter_fn: Filter experiments w.r.t. their names
        :param bool lazy_load: Whether to load all images in class constructor
        :param Optional[ExpEvaluator] evaluator: class to compute IS/TS for a given experiment
        :param Callable[[str], bool] filter_fn: Filter prompts for each experiment
            Also must have a .normalize method (Callable[[str], str]) that will be used to determine order of directories
        """
        self.base_path = base_path
        self.all_info = all_info
        self.set_name = set_name

        self.lazy_load = lazy_load
        self.evaluator = evaluator
        self.filter_fn = filter_fn
        self.checkpoint_filter_fn = checkpoint_filter_fn

        self.exps_names = sorted([
            name for name in os.listdir(base_path)
            if regex.match(ExpsViewer.EXP_NAME_PATTERN, name)
        ])
        if exp_filter_fn is not None:
            self.exps_names = list(filter(exp_filter_fn, self.exps_names))

        self._configs = {name: self._read_config(name, path_mapping) for name in self.exps_names}

        exps_grid_size = np.array([(len(self.exps_names) - 1) // ncolumns + 1, ncolumns])
        buttons_grid, buttons = create_buttons_grid(exps_grid_size)

        self.exps_views = {}
        output = ipywidgets.Output()
        for name, button in zip(self.exps_names, buttons):
            button.description = name
            button.on_click(partial(self._load_exp_view, name=name, output=output))

            config = self._configs[name]

            samples_dirs = glob.glob(os.path.join(config['output_dir'], ExpsViewer.SAMPLES_DIRS_PATTERN))
            if len(samples_dirs) == 0:
                button.style.button_color = 'black'

            is_svddiff, is_textual_inversion, is_custom_diffusion = (
                'SVDDiff' in config['exp_name'], 'TI' in config['exp_name'], 'CD' in config['exp_name']
            )
            if is_textual_inversion:
                button.style.button_color = 'green'
            elif is_custom_diffusion:
                button.style.button_color = 'purple'
            elif is_svddiff:
                button.style.button_color = None
            else:
                button.style.button_color = 'red'

        self.widget = ipywidgets.VBox([
            buttons_grid,
            output
        ])

    @staticmethod
    def _view_exp(exp_name, config, all_info, set_name, lazy_load=True, filter_fn=None, checkpoint_filter_fn=None):
        output = ipywidgets.Output()
        samples_dirs = natsorted(glob.glob(os.path.join(config['output_dir'], ExpsViewer.SAMPLES_DIRS_PATTERN)))
        samples_dirs_buttons_grid, samples_dirs_buttons = create_buttons_grid(((len(samples_dirs) - 1) // 2 + 1, 2))

        for samples_dir, button in (pbar := tqdm.autonotebook.tqdm(
                zip(samples_dirs, samples_dirs_buttons), total=len(samples_dirs), leave=False, disable=True
        )):
            [*_, checkpoint_idx, _, sampling_config, _] = os.path.normpath(samples_dir).split(os.path.sep)
            pbar.set_description(checkpoint_idx)

            [checkpoint_idx] = regex.findall(ExpsViewer.SAMPLE_CHECKPOINT_IDX_PATTERN, checkpoint_idx)
            (num_inference_steps, guidance_scale, other_inference_specs) = regex.match(
                ExpsViewer.SAMPLE_SPECS_PATTERN, sampling_config
            ).groups()
            other_inference_specs = [] if other_inference_specs is None else other_inference_specs.split('_')

            if checkpoint_filter_fn is not None and not checkpoint_filter_fn(
                    checkpoint_idx, num_inference_steps, guidance_scale, other_inference_specs
            ):
                continue

            button.description = 'ckpt: {0}, st: {1}, gs: {2}, others: {3}'.format(
                checkpoint_idx, num_inference_steps, guidance_scale, other_inference_specs
            )
            info = None
            specs = (checkpoint_idx, num_inference_steps, guidance_scale, *other_inference_specs)
            if all_info is not None and (exp_name, specs) in all_info:
                info = all_info[(exp_name, specs)]

            exp_widget = MultifolderViewer(
                samples_dir, lazy_load=lazy_load, info=info, filter_fn=filter_fn, set_name=set_name
            ).view(ncolumns=5)
            button.on_click(partial(_display_widget, output=output, widget=exp_widget))

        return ipywidgets.VBox([
            ipywidgets.Label(value=config['output_dir']),
            samples_dirs_buttons_grid,
            output
        ])

    def _load_exp_view(self, button, name, output):
        with output:
            if name not in self.exps_views:
                config = self._configs[name]
                self.exps_views[name] = self._view_exp(
                    name, config, self.all_info, self.set_name,
                    self.lazy_load, self.filter_fn, self.checkpoint_filter_fn
                )
            _display_widget(button, output, self.exps_views[name])

    def view(self):
        return self.widget

    def _read_config(self, exp_name, path_mapping):
        return _read_config(self.base_path, exp_name, path_mapping)

    def get_samples_dirs(self):
        """ For all experiments determine all folders with samples (for all checkpoint idxs, num_inference_steps and guidance_scales)
        :return: dictionary where for each experiment all its folders with samples are stored
        """
        exp_samples_dirs: defaultdict[str, defaultdict[Tuple[str, ...], Any]]
        exp_samples_dirs = defaultdict(lambda: defaultdict(str))
        for name in self.exps_names:
            config = self._configs[name]

            samples_dirs = natsorted(glob.glob(os.path.join(config['output_dir'], ExpsViewer.SAMPLES_DIRS_PATTERN)))
            for samples_dir in samples_dirs:
                [*_, checkpoint_idx, _, sampling_config, _] = os.path.normpath(samples_dir).split(os.path.sep)

                [checkpoint_idx] = regex.findall(ExpsViewer.SAMPLE_CHECKPOINT_IDX_PATTERN, checkpoint_idx)
                (num_inference_steps, guidance_scale, other_inference_specs) = regex.match(
                    ExpsViewer.SAMPLE_SPECS_PATTERN, sampling_config
                ).groups()
                other_inference_specs = [] if other_inference_specs is None else other_inference_specs.split('_')

                exp_samples_dirs[name][(checkpoint_idx, num_inference_steps, guidance_scale, *other_inference_specs)] = (
                    samples_dir, config
                )

        return exp_samples_dirs

    def _evaluate(
            self, exp_name: str, checkpoint_idx: str, inference_specs: Tuple[str, ...] = ('50', '7.5'),
            cache: Optional[Dict[Tuple[str, Tuple[str, ...]], Any]] = None, verbose: bool = False
    ) -> Optional[Tuple[str, Dict]]:
        """ Evaluate CLIP and DINO IS/TS for a given experiment
        :param str exp_name: target experiment
        :param str checkpoint_idx: target checkpoint idx
        :param Tuple[str, ...] inference_specs: target num inference steps, guidance scale, and other inference specs
        :param Optional[Dict[Tuple[str, Tuple[str, ...]], Any]] cache: dictionary with cached statistics.
            All keys are of form (exp_name, (checkpoint_idx, num_inference_steps, guidance_scale))
        :param bool verbose: whether to log errors or not
        :return: experiment name and all its statistics
        """
        specs: Tuple[str, ...] = (checkpoint_idx, *inference_specs)
        try:
            samples_dirs = self.get_samples_dirs()
            samples_path, config = samples_dirs[exp_name][specs]

            if cache is not None and (exp_name, specs) in cache:
                n_evaluated_prompts = len(cache[(exp_name, specs)].get('dino_image_similarities', []))

                if n_evaluated_prompts != len(os.listdir(samples_path)):
                    print(f'Reevaluate {exp_name, specs} for new samples')
                else:
                    print(f'Use cache for {exp_name, specs}')
                    return exp_name, cache[(exp_name, specs)]

            exp_viewer = MultifolderViewer(samples_path, lazy_load=False)

            results = self.evaluator(exp_viewer, config)

            return exp_name, {**results, **{'specs': specs, 'config': config}}
        except Exception as ex:
            print(f'Failure evaluating {exp_name}, {checkpoint_idx}:', ex, traceback.format_exc())
            return f'', {'specs': specs}

    def evaluate(
            self, exps_names: List[str], checkpoint_idx: str, inference_specs: Tuple[str, ...],
            processes: int = 0, cache: Optional[Dict[Tuple[str, Tuple[str, ...]], Any]] = None, verbose: bool = False
    ) -> Dict[Tuple[str, Tuple[str, ...]], Any]:
        """ Multithreading implementation of experiments evaluation
        :param List[str] exps_names: list of experiments to evaluate
        :param str checkpoint_idx: target checkpoint idx
        :param Tuple[str, ...] inference_specs: target num inference steps, guidance scale, and other inference specs
        :param int processes: maximum number of parallel processes
        :param Optional[Dict[Tuple[str, Tuple[str, ...]], Any]] cache: dictionary with cached statistics.
            All keys are of form (exp_name, (checkpoint_idx, *inference_specs))
                for example: (exp_name, (checkpoint_idx, num_inference_steps, guidance_scale))
        :param bool verbose: whether to log errors or not
        :return: statistics for experiments in the same form as cache
        """
        if isinstance(exps_names, str):
            exps_names = [exps_names]

        eval_fn = partial(
            self._evaluate, checkpoint_idx=checkpoint_idx, inference_specs=inference_specs,
            cache=cache, verbose=verbose
        )

        if processes > 0:
            pool = ThreadPool(processes=processes)
            mapper = pool.imap
        else:
            mapper = map

        all_stats = {}
        for exp_name, results in tqdm.tqdm(mapper(eval_fn, exps_names), total=len(exps_names)):
            if exp_name == '':
                continue
            all_stats[(exp_name, tuple(results['specs']))] = results

        if processes > 0:
            # noinspection PyUnboundLocalVariable
            pool.close()
            pool.join()

        return all_stats
