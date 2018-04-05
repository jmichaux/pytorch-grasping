from PIL import Image

class PCDtoRGB(object):
    """
    Convert single-channel PCD PIL image to three-channel RGB-like image
    """
    def __call__(self, pcd):
        # _pcd = pcd
        _pcd = Image.new("RGB", pcd.size)
        _pcd.paste(pcd)
        return _pcd