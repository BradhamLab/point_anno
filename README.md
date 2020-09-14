# point_anno
Annotate 3-dimensional points in embryos using napari. 


## Install environment

```
conda create -y -n napari-env python=3.8
conda activate napari-env
pip install napari[all] dask-image scikit-image ipykernel magicgui
```

## Usage

```
python main.py <your/oif/file> -t <hpf> --channels stain1 stain2 stain3 stain4
```
