import os
import matplotlib.pyplot as plt

def plot_resource_utilization(
    lut_usage=None,
    ff_usage=None,
    bram_usage=None,
    dsp_usage=None,
    output_path=None,
    **kwargs
):
    # Default output path if caller doesn't provide one
    if output_path is None:
        output_path = os.path.join("evaluation", "test_output", "resource_test.png")

    data = {
        "LUT": lut_usage,
        "FF": ff_usage,
        "BRAM": bram_usage,
        "DSP": dsp_usage,
    }

    names = [k for k, v in data.items() if v is not None]
    values = [v for v in data.values() if v is not None]

    plt.figure(figsize=(6, 4))
    plt.bar(names, values)
    plt.ylabel("Count")
    plt.title("Estimated FPGA Resource Utilization")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path
