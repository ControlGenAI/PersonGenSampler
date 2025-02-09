import os
import copy
import string
from typing import Any, List, Union, Tuple, Optional

import glob
import yaml
import regex

import numpy as np

import matplotlib.axes


def prepare_axes(axes: List[matplotlib.axes.Axes]):
    """Prepare axes for image to be drawn: disable ticks and all spines.
    :param axes:
    :return:
    """
    for ax in np.array(axes).reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
        for key, spine in ax.spines.items():
            spine.set_visible(False)


def _safe_get(keys: Union[str, List[str], Tuple[str, ...]], dictionary: dict) -> Any:
    """Get value from recursive dictionary
    :param keys: list of keys for each level in dictionaries hierarchy
    :param dictionary: recursive dictionary of arbitrary depth
    :return:
    """
    result = dictionary

    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        if key in result:
            result = result[key]
        else:
            return None
    return result


def _read_config(base_path: str, exp_name: Optional[str] = None, path_mapping: Optional[dict] = None) -> dict:
    """Read config from hparams.yml file
    :param str base_path: base output path for TI/Dreambooth experiments
    :param Optional[str] exp_name: name of TI/Dreambooth experiment
    :return:
    """
    if exp_name is None:
        config_path = os.path.join(base_path, 'logs', 'hparams.yml')
    else:
        config_path = os.path.join(base_path, exp_name, 'logs', 'hparams.yml')
    [config_path] = glob.glob(config_path)

    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    new_config = copy.deepcopy(config)
    if path_mapping is not None:
        for key, value in config.items():
            for old_path, new_path in path_mapping.items():
                if isinstance(value, str):
                    value = value.replace(old_path, new_path)
                new_config[key] = value
    return new_config


def prompt_to_empty_prompts(prompt: str, placeholder_token: str, class_name: str) -> Tuple[str, str]:
    clean_prompt = (
        prompt
        .replace('{0} {1}'.format(placeholder_token, class_name), '{0}')
        .replace('{0}'.format(placeholder_token), '{0}')
    )
    empty_prompt = clean_prompt.replace('{0} ', '').replace(' {0}', '')
    empty_prompt_with_class = clean_prompt.format(class_name)

    return empty_prompt, empty_prompt_with_class


def prompt_to_regex(prompt: str, with_class: bool) -> str:
    """ Convert empty prompt (with one or two placeholders) into a regex that
        matches this prompt with any placeholder and class name

    :param str prompt: prompt with one or two placeholders
    :param bool with_class: whether to inject class name inside a prompt or not
    """

    fields = [
        field for _, field, _, _ in string.Formatter().parse(prompt)
        if field is not None
    ]
    if len(fields) not in {1, 2}:
        raise ValueError(f'Incorrect prompt to convert to regex: {prompt}')

    regex_str = '([^ ]+)'
    prompt = f'^{prompt}$'
    if with_class:
        if len(fields) == 1:
            prompt = prompt.format(f'{regex_str} {regex_str}')
        else:
            prompt = prompt.format(regex_str, regex_str)
    else:
        if len(fields) == 2:
            if f'{{{fields[1]}}} ' in prompt:
                prompt = prompt.replace(f'{{{fields[1]}}} ', '')
            else:
                prompt = prompt.replace(f' {{{fields[1]}}}', '')

        prompt = prompt.format(regex_str)

    return prompt


class RegexFilter:
    """Helper class that filters prompts and defines their order"""

    @staticmethod
    def from_prompts(prompts: Union[str, List[str]], with_class: bool):
        """ Create class from list of prompts
        :param Union[str, List[str]] prompts: list of prompts with one or two placeholders
        :param bool with_class: whether to inject class name inside a prompt or not
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        prompts = [prompt_to_regex(prompt, with_class) for prompt in prompts]
        regex_str = regex.compile('({0})'.format('|'.join(prompts)))

        return RegexFilter([regex_str])

    def __init__(self, patterns: List[Union[str, regex.Pattern[str]]]):
        """
        :param List[Union[str, regex.Pattern[str]]] patterns: list of regex patterns against which we will check matching
        """
        self.patterns = patterns

    def __call__(self, line: str) -> bool:
        """
        :param str line: line to check if it matches any of the patterns
        :return bool:
        """
        for pattern in self.patterns:
            if regex.match(pattern, line):
                return True

        return False

    def normalize(self, line: str) -> str:
        """
        :param str line: line to remove all placeholders that occurs in matched pattern
        :return str:
        """
        for pattern in self.patterns:
            if match := regex.match(pattern, line):
                (prompt, *matches) = [_ for _ in match.groups() if _ is not None]
                for idx, field in enumerate(matches):
                    line = line.replace(field, f'{{{idx}}}')
                break

        return line
