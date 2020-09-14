import napari
from dask_image.imread import imread
import dask.array as da
from typing import List, Optional

from dask_image.imread import imread
import napari
from magicgui import magicgui

import os
import argparse
import numpy as np

unicode = str
COLOR_CYCLE = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf'
]

def as_type(value, types=None):
    """Return argument as one of types if possible."""
    if value[0] in unicode('\'"'):
        return value[1:-1]
    if types is None:
        types = int, float, str, unicode
    for typ in types:
        try:
            return typ(value)
        except (ValueError, TypeError, UnicodeEncodeError):
            pass
    return value


def format_dict(adict, prefix='', indent='  ', bullets=('* ', '* '),
                excludes=('_', ), linelen=79):
    """Return pretty-print of nested dictionary."""
    result = []
    for k, v in sorted(adict.items(), key=lambda x: x[0].lower()):
        if any(k.startswith(e) for e in excludes):
            continue
        if isinstance(v, dict):
            v = '\n' + format_dict(v, prefix=prefix+indent, excludes=excludes)
            result.append(prefix + bullets[1] + '%s: %s' % (k, v))
        else:
            result.append(
                (prefix + bullets[0] + '%s: %s' % (k, v))[:linelen].rstrip())
    return '\n'.join(result)


class SettingsFile(dict):
    """Olympus settings file (oif, txt, pyt, roi, lut).
    Settings files contain little endian utf-16 encoded strings, except for
    [ColorLUTData] sections, which contain uint8 binary arrays.
    Settings can be accessed as a nested dictionary {section: {key: value}},
    except for {'ColorLUTData': np array}.
    Examples
    --------
    >>> with CompoundFile('test.oib') as oib:
    ...     info = oib.open_file('OibInfo.txt')
    >>> info = SettingsFile(info, 'OibInfo.txt')
    >>> info['OibSaveInfo']['Version']
    '2.0.0.0'
    """
    def __init__(self, arg, name=None):
        """Read settings file and parse into nested dictionaries.
        Parameters
        ----------
        arg : str or file object
            Name of file or open file containing little endian UTF-16 string.
            File objects are closed by this function.
        name : str
            Human readable label of stream.
        """
        dict.__init__(self)
        if isinstance(arg, (str, unicode)):
            self.name = arg
            stream = open(arg, 'rb')
        else:
            self.name = str(name)
            stream = arg

        try:
            content = stream.read()
            if not content.startswith(b'\xFF\xFE'):  # UTF16 BOM
                raise ValueError('not a valid settings file')
            content = content.rsplit(
                b'[\x00C\x00o\x00l\x00o\x00r\x00L\x00U\x00T\x00'
                b'D\x00a\x00t\x00a\x00]\x00\x0D\x00\x0A\x00', 1)
            if len(content) > 1:
                self['ColorLUTData'] = np.fromstring(
                    content[1], 'uint8').reshape(-1, 4)
            content = content[0].decode('utf-16')
        finally:
            stream.close()

        for line in content.splitlines():
            line = line.strip()
            if line.startswith(';'):
                continue
            if line.startswith('[') and line.endswith(']'):
                self[line[1:-1]] = properties = {}
            else:
                key, value = line.split('=')
                properties[key] = as_type(value)

    def __str__(self):
        """Return string with information about settings file."""
        return '\n'.join((self.name, ' (Settings File)', format_dict(self)))

def get_z_scale(oif):
    info = SettingsFile(oif)
    axes = ['Axis 0 Parameters Common', 'Axis 3 Parameters Common']
    scale_info = {x: {} for x in axes}
    for axis in axes:
        scale_info[axis]['unit'] = info[axis]['PixUnit']
        scale_info[axis]['unit_per_pxl'] = (info[axis]['EndPosition']\
                                            - info[axis]['StartPosition'])\
                                         / info[axis]['MaxSize']
    scaler = 1
    if scale_info[axes[0]]['unit'] == 'um' and scale_info[axes[1]]['unit'] == 'nm':
        scaler = 1 / 10 ** 3
    elif scale_info[axes[0]]['unit'] == scale_info[axes[1]]['unit']:
        scaler = 1
    else:
        raise ValueError("Conversion between {} and {} not implemented.".format(
                         scale_info[axes[0]]['unit'],
                         scale_info[axes[1]]['unit']))
    out = scale_info[axes[1]]['unit_per_pxl'] * scaler \
        / scale_info[axes[0]]['unit_per_pxl']
    return out

# modified from https://github.com/kevinyamauchi/PointAnnotator/blob/master/pointannotator/gui.py
def create_label_menu(points_layer, labels):
    """Create a label menu widget that can be added to the napari viewer dock
    Parameters:
    -----------
    points_layer : napari.layers.Points
        a napari points layer
    labels : List[str]
        list of the labels for each keypoint to be annotated (e.g., the body parts to be labeled).
    Returns:
    --------
    label_menu : QComboBox
        the label menu qt widget
    """
    # Create the label selection menu
    @magicgui(label={'choices': labels})
    def label_selection(label):
        return label

    label_menu = label_selection.Gui()

    def update_label_menu(event):
        """Update the label menu when the point selection changes"""
        label_menu.label = points_layer.current_properties['label'][0]

    points_layer.events.current_properties.connect(update_label_menu)

    def label_changed(result):
        """Update the Points layer when the label menu selection changes"""
        selected_label = result
        current_properties = points_layer.current_properties
        current_properties['label'] = np.asarray([selected_label])
        points_layer.current_properties = current_properties

    label_menu.label_changed.connect(label_changed)

    return label_menu


def point_annotator(
        im_path: str,
        labels: List[str],
        z_scale: float,
        channels: Optional[List[str]]=None,
):
    """Create a GUI for annotating points in a series of images.
    Parameters
    ----------
    im_path : str
        glob-like string for the images to be labeled.
    labels : List[str]
        list of the labels for each keypoint to be annotated (e.g., the body parts to be labeled).
    """
    print(im_path)
    c1 = imread(os.path.join(im_path, 's_C001*.tif'))
    c2 = imread(os.path.join(im_path, 's_C002*.tif'))
    c3 = imread(os.path.join(im_path, 's_C003*.tif'))
    c4 = imread(os.path.join(im_path, 's_C004*.tif'))
    if channels is None:
        channels = [f'C{i + 1}' for i in range(4)]
    elif len(channels) < 4:
        channels += [f'C{i + 1}' for i in range(len(channels), 4)]
    elif len(channels) > 4:
        channels = channels[:4]
    with napari.gui_qt():
        viewer = napari.Viewer()
        for stack, name in zip([c1, c2, c3, c4], channels):
            viewer.add_image(stack, name=name, scale=[z_scale, 1, 1],
                             contrast_limits=[0, 2**12 - 1])
        
        points_layer = viewer.add_points(
            properties={'label': labels},
            edge_color='label',
            edge_color_cycle=COLOR_CYCLE,
            symbol='o',
            face_color='transparent',
            edge_width=8,
            size=12,
        )
        points_layer.edge_color_mode = 'cycle'

        # add the label menu widget to the viewer
        label_menu = create_label_menu(points_layer, labels)
        viewer.window.add_dock_widget(label_menu)

        @viewer.bind_key('.')
        def next_label(event=None):
            """Keybinding to advance to the next label with wraparound"""
            current_properties = points_layer.current_properties
            current_label = current_properties['label'][0]
            ind = list(labels).index(current_label)
            new_ind = (ind + 1) % len(labels)
            new_label = labels[new_ind]
            current_properties['label'] = np.array([new_label])
            points_layer.current_properties = current_properties

        def next_on_click(layer, event):
            """Mouse click binding to advance the label when a point is added"""
            if layer.mode == 'add':
                next_label()

                # by default, napari selects the point that was just added
                # disable that behavior, as the highlight gets in the way
                layer.selected_data = {}

        points_layer.mode = 'add'
        points_layer.mouse_drag_callbacks.append(next_on_click)

        @viewer.bind_key(',')
        def prev_label(event):
            """Keybinding to decrement to the previous label with wraparound"""
            current_properties = points_layer.current_properties
            current_label = current_properties['label'][0]
            ind = list(labels).index(current_label)
            n_labels = len(labels)
            new_ind = ((ind - 1) + n_labels) % n_labels
            new_label = labels[new_ind]
            current_properties['label'] = np.array([new_label])
            points_layer.current_properties = current_properties


if __name__ == '__main__':
    hpf_to_labels = {18: ['right-cluster', 'left-cluster', 'ventral-ring',
                          'dorsal-ring', 'right-tip', 'left-tip']}
    parser = argparse.ArgumentParser(description="Annotate points in an oif file.")
    parser.add_argument('oif', metavar='i', type=str,
                         help='Olympus Imaging Format image file. '\
                              'Expecting an accompanying `oif.files` image directory.')
    parser.add_argument('hpf', metavar='t', type=int, default=18, nargs='?',
                        help='Imaging time-point to add expected point labels.')
    parser.add_argument('--channels', metavar='C', type=str, nargs='+',
                         help='Ordered channel name.')

    args = parser.parse_args()
    img_dir = args.oif + '.files/'
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Expected directory of images at {img_dir}")
    if not os.path.isdir(img_dir):
        raise TypeError(f"{img_dir} is not a directory.")
    if args.hpf not in hpf_to_labels.keys():
        raise ValueError(f"Unsupported time point: {args.hpf} hpf.")
    scaler = get_z_scale(args.oif)
    labels = hpf_to_labels[args.hpf]
    point_annotator(img_dir, labels, scaler, args.channels)