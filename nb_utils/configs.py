import numpy as np

import cv2
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt


classes_data = {
    'backpack':               ('sks', '<backpack>',       'backpack'      ),    # noqa
    'backpack_dog':           ('sks', '<backpack>',       'backpack'      ),    # noqa
    'bear_plushie':           ('sks', '<stuffed animal>', 'stuffed animal'),    # noqa
    'berry_bowl':             ('sks', '<bowl>',           'bowl'          ),    # noqa
    'can':                    ('sks', '<can>',            'can'           ),    # noqa
    'candle':                 ('sks', '<candle>',         'candle'        ),    # noqa
    'cat':                    ('sks', '<cat>',            'cat'           ),    # noqa
    'cat2':                   ('sks', '<cat>',            'cat'           ),    # noqa
    'clock':                  ('sks', '<clock>',          'clock'         ),    # noqa
    'colorful_sneaker':       ('sks', '<sneaker>',        'sneaker'       ),    # noqa
    'dog':                    ('sks', '<dog>',            'dog'           ),    # noqa
    'dog2':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'dog3':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'dog5':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'dog6':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'dog7':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'dog8':                   ('sks', '<dog>',            'dog'           ),    # noqa
    'duck_toy':               ('sks', '<toy>',            'toy'           ),    # noqa
    'fancy_boot':             ('sks', '<boot>',           'boot'          ),    # noqa
    'grey_sloth_plushie':     ('sks', '<stuffed animal>', 'stuffed animal'),    # noqa
    'monster_toy':            ('sks', '<toy>',            'toy'           ),    # noqa
    'pink_sunglasses':        ('sks', '<glasses>',        'glasses'       ),    # noqa
    'poop_emoji':             ('sks', '<toy>',            'toy'           ),    # noqa
    'rc_car':                 ('sks', '<toy>',            'toy'           ),    # noqa
    'red_cartoon':            ('sks', '<cartoon>',        'cartoon'       ),    # noqa
    'robot_toy':              ('sks', '<toy>',            'toy'           ),    # noqa
    'shiny_sneaker':          ('sks', '<sneaker>',        'sneaker'       ),    # noqa
    'teapot':                 ('sks', '<teapot>',         'teapot'        ),    # noqa
    'vase':                   ('sks', '<vase>',           'vase'          ),    # noqa
    'wolf_plushie':           ('sks', '<stuffed animal>', 'stuffed animal'),    # noqa
}

live_object_data = {
    'backpack':               'object',    # noqa
    'stuffed animal':         'object',    # noqa
    'bowl':                   'object',    # noqa
    'can':                    'object',    # noqa
    'candle':                 'object',    # noqa
    'cat':                    'live',      # noqa
    'clock':                  'object',    # noqa
    'glasses':                'object',    # noqa
    'sneaker':                'object',    # noqa
    'dog':                    'live',      # noqa
    'toy':                    'object',    # noqa
    'boot':                   'object',    # noqa
    'cartoon':                'object',    # noqa
    'teapot':                 'object',    # noqa
    'vase':                   'object',    # noqa
}

best_imgs = {
    'backpack':               '02.jpg',    # noqa
    'backpack_dog':           '00.jpg',    # noqa
    'bear_plushie':           '03.jpg',    # noqa
    'berry_bowl':             '00.jpg',    # noqa
    'can':                    '04.jpg',    # noqa
    'candle':                 '04.jpg',    # noqa
    'cat':                    '03.jpg',    # noqa
    'cat2':                   '00.jpg',    # noqa
    'clock':                  '03.jpg',    # noqa
    'colorful_sneaker':       '01.jpg',    # noqa
    'dog':                    '02.jpg',    # noqa
    'dog2':                   '02.jpg',    # noqa
    'dog3':                   '00.jpg',    # noqa
    'dog5':                   '03.jpg',    # noqa
    'dog6':                   '02.jpg',    # noqa
    'dog7':                   '01.jpg',    # noqa
    'dog8':                   '01.jpg',    # noqa
    'duck_toy':               '01.jpg',    # noqa
    'fancy_boot':             '02.jpg',    # noqa
    'grey_sloth_plushie':     '01.jpg',    # noqa
    'monster_toy':            '00.jpg',    # noqa
    'pink_sunglasses':        '05.jpg',    # noqa
    'poop_emoji':             '00.jpg',    # noqa
    'rc_car':                 '03.jpg',    # noqa
    'red_cartoon':            '01.jpg',    # noqa
    'robot_toy':              '01.jpg',    # noqa
    'shiny_sneaker':          '04.jpg',    # noqa
    'teapot':                 '00.jpg',    # noqa
    'vase':                   '02.jpg',    # noqa
    'wolf_plushie':           '04.jpg',    # noqa
}

_LOAD_IMAGE_BACKENDS = {
    'PIL': lambda path: np.asarray(Image.open(path).convert('RGB')),
    'plt': lambda path: plt.imread(path),
    'skimage': lambda path: io.imread(path),
    'opencv': lambda path: cv2.imread(path),
    None: None,
}
_LOAD_IMAGE_BACKEND = _LOAD_IMAGE_BACKENDS['PIL']
