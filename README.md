# UForm Benchmarks

To run PyTorch and ONNX benchmarks on CPUs, please run:

```sh
pip install -r requirments.txt
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
python3 main.py
```

You will afterwards find the charts in the `plots/` subdirectory.
