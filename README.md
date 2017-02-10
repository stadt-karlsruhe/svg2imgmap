# Create HTML image maps from SVG images

## Installation

    pip install -e git+https://github.com/stadt-karlsruhe/svg2imgmap#egg=svg2imgmap

## Usage

    $ svg2imgmap --help
    usage: svg2imgmap [-h] [--layer LAYER] [--debug] [--format FORMAT]
                      SVG WIDTH HEIGHT

    Create an HTML image map from an SVG image.

    <path> elements from the SVG are converted into <area> elements for the
    image map. The path's `id` is used for the area's `id` and `href`, and
    if the path has an Inkscape label then that is used for the area's `alt`
    text.

    This tool does not generate the background image for the image map, only
    the HTML code for the <map> element. The coordinate calculations assume
    that the whole SVG page is used for the background.

    The generated HTML code is printed to STDOUT.

    positional arguments:
      SVG                   SVG file
      WIDTH                 Width of the image map in pixels
      HEIGHT                Height of the image map in pixels

    optional arguments:
      -h, --help            show this help message and exit
      --debug, -d           Output debugging information
      --format FORMAT, -f FORMAT
                            Output format. Either "html" (default) or "json".
      --layer LAYER, -l LAYER
                            Use only the layer with this Inkscape label (can be
                            specified multiple times)

## License

Copyright (c) 2016, Stadt Karlsruhe (www.karlsruhe.de)

Distributed under the MIT license, see the file `LICENSE` for details.


## Changelog

### 0.1.1

- Fixed bug in SVG transform parsing
- Fixed missing console entry point

### 0.1.0

- Initial release

