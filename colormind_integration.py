# @Author: Shounak Ray <Ray>
# @Date:   26-Aug-2020 13:08:90:901  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: colormind_integration.py
# @Last modified by:   Ray
# @Last modified time: 24-Feb-2021 01:02:51:515  GMT-0700
# @License: [Private IP]


import ast
from itertools import chain

import colormap
import requests
import seaborn as sns

# Display palette from given RGB List


def display_palette_from_RGB(RGB_colors):
    HEX_list = [colormap.rgb2hex(color[0], color[1], color[2]) for color in RGB_colors]
    sns.set_palette(HEX_list)
    sns.palplot(sns.color_palette())


all_sets = []
existing_palette = ''
number_colors_total = 30
input_colors = existing_palette + ","
for n in range(number_colors_total):
    data = '{"input":[[44,43,44],[90,83,82],[224,203,173],"N","N"], "model":"default"}'
    response = ast.literal_eval(requests.post('http://colormind.io/api/', data=data).text)
    fit_colors = list(chain.from_iterable(response.values()))
    all_sets.append(fit_colors)

[display_palette_from_RGB(fit_colors) for fit_colors in all_sets]
