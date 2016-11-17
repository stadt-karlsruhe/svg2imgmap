#!/usr/bin/env python

'''
Create a HTML image map from an SVG image.
'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
from cgi import escape
from itertools import islice
import math
import re
import textwrap

from lxml import etree
import svg.path


SVG_NS = '{http://www.w3.org/2000/svg}'

INKSCAPE_NS = '{http://www.inkscape.org/namespaces/inkscape}'


class SVGTransform(object):
    '''
    An SVG transform.
    '''
    @classmethod
    def identity(cls):
        return MatrixTransform(1, 0, 0, 1, 0, 0)

    @classmethod
    def matrix(cls, a, b, c, d, e, f):
        return MatrixTransform(a, b, c, d, e, f)

    @classmethod
    def translate(cls, x, y=0):
        return MatrixTransform(1, 0, 0, 1, x, y)

    @classmethod
    def scale(cls, x, y=None):
        if y is None:
            y = x
        return MatrixTransform(x, 0, 0, y, 0, 0)

    @classmethod
    def rotate(cls, a, x=None, y=None):
        '''
        Rotation transform.

        ``a`` is the angle in degrees.
        '''
        if (x is None) and (y is None):
            # Rotation about origin
            a = math.radians(a)
            c = math.cos(a)
            s = math.sin(a)
            return MatrixTransform(c, s, -s, c, 0, 0)
        elif (x is not None) and (y is not None):
            # Rotation about (x, y)
            return ChainedTransform([
                cls.translate(x, y),
                cls.rotate(a),
                cls.translate(-x, -y)
            ])
        else:
            raise ValueError('``x`` and ``y`` must both be given or not.')

    @classmethod
    def skewx(cls, a):
        '''
        Transform for skewing along X axis.

        ``a`` is the angle in degrees.
        '''
        return MatrixTransform(1, 0, math.tan(math.radians(a)), 1, 0, 0)

    @classmethod
    def skewy(cls, a):
        '''
        Transform for skewing along Y axis.

        ``a`` is the angle in degrees.
        '''
        return MatrixTransform(1, math.tan(math.radians(a)), 0, 1, 0, 0)

    @classmethod
    def parse(cls, s):
        '''
        Parse an SVG transform string.
        '''
        transforms = []
        methods = {name: getattr(cls, name.lower()) for name in
                   'matrix translate scale rotate skewX skewY'.split()}
        s = s.strip()
        while s:
            open_index = s.index('(')
            close_index = s.index(')', open_index)
            name = s[:open_index].strip()
            args = re.split(r'[^\d\.-]', s[open_index + 1:close_index])
            args = [float(a) for a in args if a]
            transforms.append(methods[name](*args))
            s = s[close_index + 1:]
        if len(transforms) == 1:
            return transforms[0]
        else:
            return ChainedTransform(transforms)

    def apply(self, points):
        '''
        Apply this transform to points.

        ``points`` yields 2-tuples ``(x, y)``.

        Yields the transformed points.
        '''
        raise NotImplementedError('Must be implemented by subclasses.')


class MatrixTransform(SVGTransform):

    def __init__(self, a, b, c, d, e, f):
        '''
        Constructor.

        The arguments specify the following components of the
        transform's matrix::

            a  c  e
            b  d  f
            0  0  1
        '''
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def apply(self, points):
        for x, y in points:
            tx = self.a * x + self.c * y + self.e
            ty = self.b * x + self.d * y + self.f
            yield tx, ty


class ChainedTransform(SVGTransform):
    '''
    A chain of multiple SVG transforms.
    '''
    def __init__(self, transforms):
        '''
        Constructor.

        ``transforms`` is a list of transforms that will be applied
        *from right to left*.
        '''
        self.transforms = list(transforms)

    def apply(self, points):
        for transform in reversed(self.transforms):
            points = transform.apply(points)
        for point in points:
            yield point


def _get_transform(node):
    '''
    Get the transform for an SVG element.

    Walks up the tree and collects the transforms and returns an
    ``SVGTransform``.
    '''
    transforms = []
    while node is not None:
        s = node.get('transform')
        if s:
            transforms.append(SVGTransform.parse(s))
        node = node.getparent()
    if not transforms:
        return SVGTransform.identity()
    if len(transforms) == 1:
        return transforms[0]
    return ChainedTransform(reversed(transforms))


# From the (old) itertools documentation
def window(seq, n=2):
    """
    Returns a sliding window (of width n) over data from the iterable.

    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def _get_layers(root):
    '''
    Yield Inkscape layers from an SVG document.
    '''
    return (g for g in root.iter(SVG_NS + 'g')
            if g.get(INKSCAPE_NS + 'groupmode', '') == 'layer')


def _get_paths(node):
    '''
    Yield path descendants of an SVG node.
    '''
    return node.iter(SVG_NS + 'path')


def _concatenate(a, b):
    '''
    Concatenate two lists and avoid a duplicate joint item.
    '''
    if a and b and (a[-1] == b[0]):
        return a + b[1:]
    return a + b


def _segment_to_polyline(segment, start=0, end=1, start_point=None,
                         end_point=None, max_error=1e-12, min_depth=5,
                         depth=0, close=False):
    '''
    Convert an ``svg.path`` path segment to a polyline.
    '''
    if start_point is None:
        start_point = segment.point(start)
    if end_point is None:
        end_point = segment.point(end)
    mid = (start + end) / 2
    mid_point = segment.point(mid)
    naive_mid_point = 0.5 * (start_point + end_point)
    error = abs(naive_mid_point - mid_point)
    if (error > max_error) or (depth < min_depth):
        poly = _concatenate(
            _segment_to_polyline(segment, start, mid, start_point, mid_point,
                                 max_error, min_depth, depth + 1, False),
            _segment_to_polyline(segment, mid, end, mid_point, end_point,
                                 max_error, min_depth, depth + 1, False),
        )
    else:
        poly = [start_point, end_point]
    if close and (poly[0] != poly[-1]):
        poly.append(poly[0])
    return poly


def _path_to_polyline(path, max_error=1e-12):
    '''
    Convert an ``svg.path`` path to a polyline.
    '''
    poly = []
    for segment in path:
        if isinstance(segment, svg.path.Line):
            segment_poly = [segment.start, segment.end]
        else:
            segment_poly = _segment_to_polyline(segment, max_error=max_error)
        poly = _concatenate(poly, segment_poly)
    return poly


def _create_area(poly, href, alt='', id=None):
    '''
    Create an HTML ``<area>`` element.
    '''
    area = etree.Element('area')
    if alt:
        area.set('alt', alt)
    coords = ','.join('{},{}'.format(int(p[0]), int(p[1])) for p in poly)
    area.set('coords', coords)
    area.set('href', href)
    area.set('shape', 'poly')
    if id:
        area.set('id', id)
    return area


def _strip_adjacent_duplicates(it):
    '''
    Strip adjacent duplicates from an iterator.
    '''
    it = iter(it)
    last = next(it)
    yield last
    while True:
        current = next(it)
        if current != last:
            yield current
            last = current


def _simplify_straight_lines(poly):
    '''
    Simplify straight lines in a polyline.
    '''
    p = poly[0]
    result = [p]
    for q, r in window(poly[1:]):
        pr = (r[0] - p[0], r[1] - p[1])
        qr = (r[0] - q[0], r[1] - q[1])
        angle_pr = math.atan2(pr[1], pr[0])
        angle_qr = math.atan2(qr[1], qr[0])
        if angle_pr != angle_qr:
            result.append(q)
            p = q
    result.append(poly[-1])
    return result


def _get_svg_size(root):
    '''
    Get the size of an SVG document in SVG units.
    '''
    viewbox = map(float, root.get('viewBox', '0 0 0 0').split())
    width = root.get('width', '100%')
    height = root.get('height',' 100%')
    if width.endswith('%'):
       width = 0.01 * float(width[:-1]) * viewbox[2]
    else:
        width = float(width)
    if height.endswith('%'):
       height = 0.01 * float(height[:-1]) * viewbox[3]
    else:
        height = float(height)
    return width, height


def svg2imgmap(svg_file, map_size, layers=None):
    '''
    Generate an HTML image map.

    ``svg_file`` s the filename of the SVG image and ``map_size`` is a
    2-tuple specifying the width and height of the image map in pixels.

    Layers is an option list of Inkscape layer names. If given then only
    listed layers are exported. By default all layers are exported.

    Returns the generated HTML code.
    '''
    root = etree.parse(svg_file).getroot()
    svg_width, svg_height = _get_svg_size(root)
    x_factor = map_size[0] / svg_width
    y_factor = map_size[1] / svg_height
    max_error = 1e-5
    lines = ['<map>']
    for layer in _get_layers(root):
        label = layer.get(INKSCAPE_NS + 'label')
        if layers and label not in layers:
            continue
        if label:
            lines.append('<!-- Layer "{}" -->'.format(escape(label)))
        for node in _get_paths(layer):
            path = svg.path.parse_path(node.get('d'))
            poly = _path_to_polyline(path, max_error=max_error)
            poly = _get_transform(node).apply((p.real, p.imag) for p in poly)
            poly = [(int(p[0] * x_factor), int(p[1] * y_factor)) for p in poly]
            poly = list(_strip_adjacent_duplicates(poly))
            poly = _simplify_straight_lines(poly)
            alt = node.get(INKSCAPE_NS + 'label', '')
            id = node.get('id')
            area = _create_area(poly, href=id, alt=alt, id=id)
            lines.append(etree.tostring(area))
    lines.append('</map>')
    return '\n'.join(lines)


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

    The coordinate calculation assume that the whole SVG page has been exported
    to the given raster image.

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
    args = parser.parse_args()

    html = svg2imgmap(args.svg_file, (args.map_width, args.map_height),
                      layers=args.layer)
    print(html)


if __name__ == '__main__':
    main()

