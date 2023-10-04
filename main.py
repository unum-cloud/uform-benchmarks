import os
import copy
from json import load
from typing import Optional
import time

import uform
import torch
import open_clip
import onnx
import onnxruntime as ort
import pandas as pd
import openvino as ov
from onnxconverter_common import float16, auto_mixed_precision
from torch import Tensor
from torch.ao.quantization import get_default_qconfig_mapping
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from uform import TextEncoder, VisualEncoder
from onnxruntime.quantization import quantize_dynamic, QuantType
from tabulate import tabulate


class TextEncoder_onnx(TextEncoder):
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> torch.Tensor:
        features = self.forward_features(input_ids, attention_mask)
        embeddings = self.forward_embedding(features, attention_mask)
        return features, embeddings


def get_onnx_model(model_name: str, token: Optional[str] = None):
    config_path, state, _ = uform.get_checkpoint(model_name, token)
    with open(config_path, "r") as f:
        model = TextEncoder_onnx(**load(f)["text_encoder"])

    model.load_state_dict(state["text_encoder"])
    return model.eval()


def get_vit_model(model_name: str, token: Optional[str] = None):
    config_path, state, _ = uform.get_checkpoint(model_name, token)
    with open(config_path, "r") as f:
        model = VisualEncoder(**load(f)["image_encoder"])

    model.load_state_dict(state["image_encoder"])
    return model.eval()


def get_onnx(dummy_text):
    model_onnx = get_onnx_model("unum-cloud/uform-vl-english")
    torch.onnx.export(
        model_onnx,
        (dummy_text["input_ids"], dummy_text["attention_mask"]),
        "./onnx/uform_text.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )


def onnx_quntize(model_path):
    # Quantize the model to int8
    new_path = model_path[:-5] + "_int8.onnx"
    quantize_dynamic(
        model_path,
        new_path,
        weight_type=QuantType.QUInt8,
        # optimize_model=True,
    )


def onnx_f16(model_path, dummy_text):
    model = onnx.load(model_path)
    model_fp16 = float16.convert_float_to_float16(model)
    # model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(
    #     model,
    #     {
    #         "input_ids": dummy_text["input_ids"].numpy(),
    #         "attention_mask": dummy_text["attention_mask"].numpy(),
    #     },
    #     rtol=0.01,
    #     atol=0.001,
    #     keep_io_types=True,
    # )
    new_path = model_path[:-5] + "_fp16.onnx"
    onnx.save(model_fp16, new_path)


def torch_test(model, batch_size, seconds):
    """
    Throughput and Latency evals for UForm
    """
    dummy_text = {
        "input_ids": torch.randint(0, 30522, (batch_size, 77)),
        "attention_mask": torch.ones(batch_size, 77, dtype=torch.int32),
    }
    t_end = time.time() + seconds
    cnt = 0
    with torch.no_grad():
        while time.time() < t_end:
            _ = model.encode_text(dummy_text)
            cnt += batch_size
    throughput = cnt / seconds
    latency = 1000 * batch_size / throughput

    return throughput, latency


def torch_test_openclip(model, batch_size, seconds):
    """
    Throughput and Latency evals for OpenCLIP Vit-16
    """
    dummy_text = torch.randint(0, 30522, (batch_size, 77))
    t_end = time.time() + seconds
    cnt = 0
    with torch.no_grad():
        while time.time() < t_end:
            _ = model.encode_text(dummy_text)
            cnt += batch_size

    throughput = cnt / seconds
    latency = 1000 * batch_size / throughput

    return throughput, latency


def onnx_test(model_name, batch_size, seconds):
    """
    Throughput and Latency evals for UForm OpenVINO
    """
    providers = ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_sess = ort.InferenceSession(model_name, sess_options, providers=providers)

    dummy_text = {
        "input_ids": torch.randint(0, 30522, (batch_size, 77)).numpy(),
        "attention_mask": torch.ones(batch_size, 77, dtype=torch.int32).numpy(),
    }
    t_end = time.time() + seconds
    cnt = 0
    with torch.no_grad():
        while time.time() < t_end:
            _ = ort_sess.run(
                None,
                dummy_text,
            )
            cnt += batch_size

    throughput = cnt / seconds
    latency = 1000 * batch_size / throughput

    return throughput, latency


def openvino_test(model_name, batch_size, seconds):
    """
    Throughput and Latency evals for UForm ONNX Vit-16
    """
    dummy_text = {
        "input_ids": torch.randint(0, 30522, (5000000, 77)),
        "attention_mask": torch.ones(5000000, 77, dtype=torch.int32),
    }

    ov_model = ov.convert_model(
        model_name,
        input=[
            ("input_ids", [-1, -1], ov.Type.i64),
            ("attention_mask", [-1, -1], ov.Type.i64),
        ],
    )
    core = ov.Core()
    compiled_model = core.compile_model(ov_model, "AUTO")
    t_end = time.time() + seconds
    cnt = 0
    with torch.no_grad():
        while time.time() < t_end:
            _ = compiled_model(
                {
                    "input_ids": dummy_text["input_ids"][cnt : cnt + batch_size],
                    "attention_mask": dummy_text["attention_mask"][
                        cnt : cnt + batch_size
                    ],
                }
            )
            cnt += batch_size

    throughput = cnt / seconds
    latency = 1000 * batch_size / throughput

    return throughput, latency


if __name__ == "__main__":
    model = uform.get_model("unum-cloud/uform-vl-english")
    model_openclip, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="laion400m_e31",
    )

    # ONNX
    newpath = "./onnx/"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    dummy_text = {
        "input_ids": torch.randint(0, 30522, (1, 77)),
        "attention_mask": torch.ones(1, 77, dtype=torch.int32),
    }
    get_onnx(dummy_text)
    onnx_quntize("./onnx/uform_text.onnx")
    onnx_f16("./onnx/uform_text.onnx", dummy_text)

    # BENCH
    SECONDS = 60
    throughput_table = [
        [
            "Batch_size",
            "UForm PyTorch",
            "Open_Clip Pytorch",
            "UFrom ONNX fp32",
            "UFrom ONNX fp16",
            "UFrom ONNX i8",
            "OpenVINO",
        ]
    ]
    latency_table = copy.deepcopy(throughput_table)

    for BATCH_SIZE in [1, 4, 16, 64, 128, 256]:
        throughput_u, latency_u = torch_test(model, BATCH_SIZE, SECONDS)
        throughput_oc, latency_oc = torch_test_openclip(
            model_openclip, BATCH_SIZE, SECONDS
        )
        throughput_onxf32, latency_onxf32 = onnx_test(
            "./onnx/uform_text.onnx", BATCH_SIZE, SECONDS
        )
        throughput_onxf16, latency_onxf16 = onnx_test(
            "./onnx/uform_text_fp16.onnx", BATCH_SIZE, SECONDS
        )
        throughput_onxi8, latency_onxi8 = onnx_test(
            "./onnx/uform_text_int8.onnx", BATCH_SIZE, SECONDS
        )
        throughput_openvino, latency_openvino = openvino_test(
            "./onnx/uform_text.onnx", BATCH_SIZE, SECONDS
        )
        throughput_table.append(
            [
                BATCH_SIZE,
                round(throughput_u, 1),
                round(throughput_oc, 1),
                round(throughput_onxf32, 1),
                round(throughput_onxf16, 1),
                round(throughput_onxi8, 1),
                round(throughput_openvino, 1),
            ]
        )
        latency_table.append(
            [
                BATCH_SIZE,
                round(latency_u, 1),
                round(latency_oc, 1),
                round(latency_onxf32, 1),
                round(latency_onxf16, 1),
                round(latency_onxi8, 1),
                round(latency_openvino, 1),
            ]
        )
        print(f"Batch size {BATCH_SIZE} is done!!")

    table1 = tabulate(throughput_table, headers="firstrow")
    table2 = tabulate(latency_table, headers="firstrow")

    # Save Tables
    df = pd.DataFrame(
        columns=throughput_table[0],
        data=throughput_table[1:],
    )
    dy = pd.DataFrame(
        columns=latency_table[0],
        data=latency_table[1:],
    )
    df.to_csv("stats/throughput.csv")
    dy.to_csv("stats/latency.csv")

    print("\n", "Throughput (sequences per second):")
    print(table1, "\n")
    print("Latency (ms):")
    print(table2)
