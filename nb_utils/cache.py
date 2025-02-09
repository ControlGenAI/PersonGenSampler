import os
from typing import Optional
from ast import literal_eval

import glob
import json


class Cache:
    def __init__(self, path: str, indent: Optional[int] = None):
        """
        :param str path: Path to the file with cached json
        :param Optional[int] indent: indent value for json.dump
        """
        self.path = path
        self._indent = indent

    def get(self) -> dict:
        """
        :return: read saved json from file and evaluate its keys
        """
        if not os.path.exists(self.path):
            return {}

        with open(self.path, 'r') as cache_file:
            cache = json.load(cache_file)

        return {literal_eval(k): v for k, v in cache.items()}

    def _put(self, data: dict):
        """ Store data overwriting existing file
        :param dict data: dictionary to store
        :return:
        """
        with open(self.path, 'w') as cache_file:
            # noinspection PyTypeChecker
            json.dump({str(k): v for k, v in data.items()}, cache_file, indent=self._indent)

    def update(self, data):
        """ Store data updating existing file
        :param dict data: dictionary to store
        :return:
        """
        cache = self.get()
        self._put(cache | data)


class DistributedCache:
    def __init__(self, path_template: str):
        """
        :param path_template: glob compatible template that points to cache files
        """
        self.path_template = path_template

    def get(self) -> dict:
        """
        :return: read saved jsons from all files that matches to the template and evaluate their keys
        """
        paths = glob.glob(self.path_template)

        result = {}
        for path in paths:
            try:
                current_cache = Cache(path).get()
            except Exception as ex:
                print(f'Could not load cache for path {path}. Exception: {ex}')
                continue

            keys_intersection = set(current_cache.keys()).intersection(result)
            if len(keys_intersection):
                print(f'Intersection over keys: {keys_intersection}')
            result |= current_cache

        return result
