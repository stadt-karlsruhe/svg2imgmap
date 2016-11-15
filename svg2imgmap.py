#!/usr/bin/env python

'''
Create a HTML image map from an SVG image.

Does not generate the actual raster image, use your SVG editor for that.
Make sure that you export the whole page, otherwise the generated
coordinates won't match.

SVG transforms are ignored completely.
'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from itertools import islice
import math

from lxml import etree
import svg.path


SVG_NS = '{http://www.w3.org/2000/svg}'

INKSCAPE_NS = '{http://www.inkscape.org/namespaces/inkscape}'


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
    #print('{}@{} - {}@{} - {}@{}'.format(start, start_point, mid, mid_point, end, end_point))
    naive_mid_point = 0.5 * (start_point + end_point)
    error = abs(naive_mid_point - mid_point)
    #print('  naive: {}'.format(naive_mid_point))
    #print('  error: {}'.format(error))
    #print('  depth: {}'.format(depth))
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


if __name__ == '__main__':
    import argparse

    from PIL import Image

    parser = argparse.ArgumentParser(description='Create image maps from SVGs')
    parser.add_argument('svg_file', metavar='SVG', help='SVG file')
    parser.add_argument('img_file', metavar='RASTER', help='Raster image file')
    args = parser.parse_args()

    root = etree.parse(args.svg_file).getroot()
    img = Image.open(args.img_file)

    print('<!--')

    # Calculate size of one pixel in SVG units
    svg_width = float(root.get('width'))
    svg_height = float(root.get('height'))
    print('SVG size is {} x {}'.format(svg_width, svg_height))
    img_width, img_height = img.size
    print('Raster image size is {} x {}'.format(img_width, img_height))
    pixel_width = svg_width / img_width
    pixel_height = svg_height / img_height
    max_error = min(pixel_width, pixel_height)
    print('Pixel size in SVG units is {} x {}'.format(pixel_width,
          pixel_height))
    print('max_error = {}'.format(max_error))
    x_factor = img_width / svg_width
    y_factor = img_height / svg_height

    print('-->')

    print('<html><body><img src="{}" usemap="#imgmap">'.format(args.img_file))

    print('<map id="imgmap">')
    for layer in _get_layers(root):
        #print(layer.get(INKSCAPE_NS + 'label'))
        for node in _get_paths(layer):
            path = svg.path.parse_path(node.get('d'))
            poly = _path_to_polyline(path, max_error=max_error)
            poly = [(int(p.real * x_factor), int(p.imag * y_factor))
                    for p in poly]
            poly = list(_strip_adjacent_duplicates(poly))
            poly = _simplify_straight_lines(poly)
            alt = node.get(INKSCAPE_NS + 'label', '')
            id = node.get('id')
            area = _create_area(poly, 'FIXME ' + alt, alt=alt, id=id)
            print(etree.tostring(area))
    print('</map>')

    print('</body></html>')
