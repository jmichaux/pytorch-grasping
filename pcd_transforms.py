from PIL import Image

class PCDtoRGB(object):
    """
    Convert single-channel PCD PIL image to three-channel RGB-like image
    """
    def __call__(self, pcd):
        _pcd = pcd
        pcd = Image.new("RGB", _pcd.size)
        pcd.paste(_pcd)
        return pcd