from contextlib import contextmanager
from functools import partial
import os
from pathlib import Path
import sys


try:
    # get DCGMPATH from env
    dcgm_path = (
        Path(os.environ["DCGMPATH"]) if "DCGMPATH" in os.environ else "/usr/local/dcgm"
    )
    # add dcgm bindings to path
    bindings_path = Path(dcgm_path) / "bindings" / "python3"
    assert bindings_path.exists()
    sys.path.append(str(bindings_path))
    import pydcgm
    import dcgm_structs
except Exception as e:
    pass
    print(e)
    print(
        "Unable to find DCGM python bindings, please set the `DCGMPATH` environment variable to the path of the dcgm folder, as demonstrated below: "
    )
    print("export DCGMPATH=/usr/local/dcgm")


DCGM_JOB_ID = "DCGM_JOB"


def init_hostengine(fieldIds, dcgmSamplingInterval, dcgmMaxKeepAge):
    opMode = dcgm_structs.DCGM_OPERATION_MODE_AUTO
    dcgmHandle = pydcgm.DcgmHandle(opMode=opMode)
    dcgmGroup = pydcgm.DcgmGroup(
        dcgmHandle,
        groupName="one_gpu_group",
        groupType=dcgm_structs.DCGM_GROUP_EMPTY,
    )

    ## Get a handle to the system level object for DCGM
    dcgmSystem = dcgmHandle.GetSystem()
    supportedGPUs = dcgmSystem.discovery.GetAllSupportedGpuIds()

    ## Create an empty group. Let's call the group as "one_gpus_group".
    ## We will add the first supported GPU in the system to this group.
    dcgmGroup = dcgmSystem.GetEmptyGroup("one_gpu_group")

    # Skip the test if no supported gpus are available
    if len(supportedGPUs) < 1:
        print("Unable to find supported GPUs on this system")
        sys.exit(0)

    for gpu in supportedGPUs:
        dcgmGroup.AddGpu(gpu)

    print("Running with supported GPUs", supportedGPUs)

    dcgmFieldGroup = pydcgm.DcgmFieldGroup(
        dcgmHandle, "profiling_metrics", fieldIds=fieldIds
    )
    dcgmGroup.samples.WatchFields(
        dcgmFieldGroup, dcgmSamplingInterval, dcgmMaxKeepAge, 0
    )
    dcgmSystem.profiling.Resume()

    dcgmSamples = dcgmGroup.samples
    dfvc = dcgmSamples.GetAllSinceLastCall(dfvc=None, fieldGroup=dcgmFieldGroup)

    return dcgmGroup, dcgmFieldGroup, dfvc, dcgmHandle


def get_metrics(
    dcgmGroup: pydcgm.DcgmGroup,
    dcgmFieldGroup: pydcgm.DcgmFieldGroup,
    dfvc: pydcgm.dcgm_field_helpers.DcgmFieldValueCollection,
    dcgmHandle: pydcgm.DcgmHandle,
):
    ## Get the current configuration for the group
    dcgmSamples = dcgmGroup.samples
    samples = dcgmSamples.GetAllSinceLastCall(dfvc=dfvc, fieldGroup=dcgmFieldGroup)
    return samples.values


vals = []


@contextmanager
def dcgm_profiling_decorator(
    fieldIds=[1003, 1004], dcgmSamplingInterval=1000, dcgmMaxKeepAge=3600
):
    dcgmGroup, dcgmFieldGroup, dfvc, dcgmHandle = init_hostengine(
        fieldIds=fieldIds,
        dcgmSamplingInterval=dcgmSamplingInterval,
        dcgmMaxKeepAge=dcgmMaxKeepAge,
    )
    vals.append((dcgmGroup, dcgmFieldGroup, dfvc, dcgmHandle))
    try:
        yield partial(get_metrics, dcgmGroup, dcgmFieldGroup, dfvc, dcgmHandle)
    finally:
        dcgmSystem = dcgmHandle.GetSystem()
        dcgmSystem.profiling.Pause()

        dcgmFieldGroup.Delete()
        dcgmGroup.Delete()
        dcgmHandle.Shutdown()

        del dfvc
        del dcgmFieldGroup
        del dcgmGroup
        del dcgmHandle


if __name__ == "__main__":
    import torch

    with dcgm_profiling_decorator() as gm:
        a, b = torch.rand(1000, 1000), torch.rand(1000, 1)
        a = a.cuda()
        b = b.cuda()

        for i in range(100000):
            torch.matmul(a, b)
        torch.cuda.synchronize()

        metrics = gm()
        print(metrics)

    with dcgm_profiling_decorator() as gm:
        a, b = torch.rand(1000, 1000), torch.rand(1000, 1)
        a = a.cuda()
        b = b.cuda()

        for i in range(100000):
            torch.matmul(a, b)
        torch.cuda.synchronize()

        metrics = gm()
        print(metrics)
