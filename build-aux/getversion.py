#!/usr/bin/env python3

import configparser
import os

source_root = os.environ.get('MESON_SOURCE_ROOT', '../')

config = configparser.ConfigParser()
config.read(os.path.join(source_root, 'Cargo.toml'))

print(config['package']['version'].strip('\"'))

# ex: set ts=4 sw=4 et fenc=utf-8 syntax=python :
