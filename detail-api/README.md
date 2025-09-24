# PASCAL in Detail API

Python API for the [PASCAL in Detail](https://sites.google.com/view/pasd/dataset) multi-task computer vision challenge. This API is a fork of [MS COCO vision challenge API](https://github.com/pdollar/coco).

## To install:
Run `make` and `make install` under the [PythonAPI](PythonAPI/) directory.

In Python:

```python
from detail import Detail
details = Detail('json/trainval_merged.json', 'VOCdevkit/VOC2010/JPEGImages')
```

If you wish to use the API from MATLAB, see [MATLAB's documentation for calling Python code](https://www.mathworks.com/help/matlab/matlab_external/call-python-from-matlab.html). The Detail API no longer maintains a separate MATLAB API.

## To see a demo:

Run the IPython notebook [PythonAPI/ipynb/detailDemo.ipynb](PythonAPI/ipynb/detailDemo.ipynb).
