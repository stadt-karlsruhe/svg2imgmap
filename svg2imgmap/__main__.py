#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2016, Stadt Karlsruhe (www.karlsruhe.de)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Command-line interface for svg2imgmap.
'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import logging
import textwrap

from . import log, svg2imgmap


def main():
    '''
    Command line entry point.
    '''
    description = textwrap.dedent('''\
    Create an HTML image map from an SVG image.

    <path> elements from the SVG are converted into <area> elements for the
    image map. The path's `id` is used for the area's `id` and `href`, and
    if the path has an Inkscape label then that is used for the area's `alt`
    text.

    This tool does not generate the background image for the image map, only
    the HTML code for the <map> element. The coordinate calculations assume
    that the whole SVG page is used for the background.

    The generated HTML code is printed to STDOUT.
    ''')
    parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('svg_file', metavar='SVG', help='SVG file')
    parser.add_argument('map_width', metavar='WIDTH', type=int,
                        help='Width of the image map in pixels')
    parser.add_argument('map_height', metavar='HEIGHT', type=int,
                        help='Height of the image map in pixels')
    parser.add_argument('--layer', '-l', metavar='LAYER', action='append',
                        help='Use only the layer with this Inkscape label '
                        + '(can be specified multiple times)')
    parser.add_argument('--debug', '-d', action='store_true', default=False,
                        help='Output debugging information')
    args = parser.parse_args()

    if args.debug:
        handler = logging.StreamHandler()
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)

    html = svg2imgmap(args.svg_file, (args.map_width, args.map_height),
                      layers=args.layer)
    print(html)


if __name__ == '__main__':
    main()

