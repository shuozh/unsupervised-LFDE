from collections import OrderedDict
import json
import yaml
import datetime
import torch
import torch.nn.functional as F
import numpy as np
import sys

def date_time():
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d, %H:%M:%S")
    return date_time

def log(log_file, str, also_print=True, with_time=True):
    with open(log_file, 'a+') as F:
        if with_time:
            F.write(date_time() + '  ')
        F.write(str)
    if also_print:
        if with_time:
            print(date_time(), end='  ')
        print(str, end='')


def parse(file_path):
    """
    读取YAML文件并返回字典
    
    参数:
        file_path (str): YAML文件的路径
        
    返回:
        dict: 从YAML文件解析得到的字典
        None: 如果读取失败
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 加载YAML文件内容为字典
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
    except yaml.YAMLError as e:
        print(f"错误: 解析YAML文件时出错 - {e}")
    except Exception as e:
        print(f"错误: 读取文件时发生意外错误 - {e}")
    return None


def recursive_print(src, dpth=0, key=None):
    """ Recursively prints nested elements."""
    tabs = lambda n: ' ' * n * 4 # or 2 or 8 or...

    if isinstance(src, dict):
        if key is not None:
            print(tabs(dpth) + '%s: ' % (key))
        for key, value in src.items():
            recursive_print(value, dpth + 1, key)
    elif isinstance(src, list):
        if key is not None:
            print(tabs(dpth) + '%s: ' % (key))
        for litem in src:
            recursive_print(litem, dpth)
    else:
        if key is not None:
            print(tabs(dpth) + '%s: %s' % (key, src))


def recursive_log(log_file, src, dpth=0, key=None):
    """ Recursively prints nested elements."""
    tabs = lambda n: ' ' * n * 4 # or 2 or 8 or...

    if isinstance(src, dict):
        if key is not None:
            log(log_file, tabs(dpth) + '%s: \n' % (key), with_time=False)
        for key, value in src.items():
            recursive_log(log_file, value, dpth + 1, key)
    elif isinstance(src, list):
        if key is not None:
            log(log_file, tabs(dpth) + '%s: \n' % (key), with_time=False)
        for litem in src:
            recursive_log(log_file, litem, dpth)
    else:
        if key is not None:
            log(log_file, tabs(dpth) + '%s: %s\n' % (key, src), with_time=False)



def warp_all(disparity, in_put, device, an=9):
    '''
    已知中心视角视差，将所有视图反向warp到中心视角
    :param device:
    :param disparity: b, h, w
    :param in_put:  b, c, n, h, w
    :return: b, c, n, h, w
    '''
    b, c, n, h, w = in_put.shape
    an2 = an // 2
    
    xx = torch.arange(0, w).view(1, 1, 1, w).expand(b, n, h, w).float().to(device)
    yy = torch.arange(0, h).view(1, 1, h, 1).expand(b, n, h, w).float().to(device)
    zz = torch.arange(0, n).view(1, n, 1, 1).expand(b, n, h, w).float().to(device)
    for i in range(n):
        ind_h_source = i // an
        ind_w_source = i % an
        xx[:, i, ...] = xx[:, i, ...] + disparity*(an2-ind_w_source)
        yy[:, i, ...] = yy[:, i, ...] + disparity*(an2-ind_h_source)
    xx = xx*2/(w-1)-1
    yy = yy*2/(h-1)-1
    zz = zz*2/(n-1)-1
    grid = torch.stack([xx, yy, zz], dim=4)  # N, d, h, w, 3
    grid = grid.to(device)
    out_put = F.grid_sample(in_put, grid, align_corners=True)
    return out_put

def read_pfm(fpath, expected_identifier="Pf"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    def _get_next_line(f):
        next_line = f.readline().decode('utf-8').rstrip()
        # ignore comments
        while next_line.startswith('#'):
            next_line = f.readline().rstrip()
        return next_line

    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception('Could not parse max value / endianess information: "%s". '
                            'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data).copy()  # 矩阵翻转
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

        return data
    
    
def write_pfm(data, fpath, scale=1, file_identifier=b'Pf', dtype="float32"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    data = np.flipud(data)
    height, width = np.shape(data)[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder
    print(endianess)

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1

    with open(fpath, 'wb') as file:
        file.write((file_identifier))
        file.write(('\n%d %d\n' % (width, height)).encode())
        file.write(('%d\n' % scale).encode())

        file.write(values)



