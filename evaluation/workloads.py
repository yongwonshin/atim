import tvm
from tvm.script import tir as T
from base import UPMEMWorkload, cleanup
from tensor import host_array
import numpy as np
import math
import argparse
from bench import *
import subprocess


def get_workload(op_type):
    if op_type == "mtv":
        return MTV
    elif op_type == "ttv":
        return TTV
    elif op_type == "poly_gemv1":
        return GEMV
    elif op_type == "polygemv1":
        return GEMV
    elif op_type == "va":
        return VA
    elif op_type == "ta":
        return TA
    elif op_type == "poly_va":
        return GEVA
    elif op_type == "polyva":
        return GEVA
    elif op_type == "poly_mixed":
        pass
    elif op_type == "dot":
        return DOT
    elif op_type == "innerprod":
        pass
    elif op_type == "mmtv":
        return MMTV
    elif op_type == "red":
        return RED
    else:
        raise Exception(f"Unknown operator type: {op_type}")


class VA(UPMEMWorkload):
    def __init__(self, **kwargs):
        super().__init__(
            profile="va",
            required=dict(),
            symbols=["A", "B", "C"],
            output_symbol="C",
            **kwargs,
        )

    def fetch_data(self):
        self.host.A = host_array(self.M, self.dtype)
        self.host.B = host_array(self.M, self.dtype)

    def host_version(self):
        self.host.C = self.host.A + self.host.B

    def benchmark_command(self, config):
        bl = int(math.log2(config["n_c"] * np.dtype(config["dtype"]).itemsize))
        pbtype = config["dtype"].upper()
        return f"make clean && NR_DPUS={config['n_b']} NR_TASKLETS={config['n_t']} \
            TYPE={pbtype} BL={bl} make && \
            ./bin/host_code -i {config['M']} -w {self.warmup} -e {self.repeat}"


class GEVA(UPMEMWorkload):
    def __init__(self, **kwargs):
        super().__init__(
            profile="poly",
            required=dict(),
            symbols=["A", "B", "C", "ALPHA", "BETA"],
            output_symbol="C",
            **kwargs,
        )

    def fetch_data(self):
        self.host.A = host_array(self.M, self.dtype)
        self.host.B = host_array(self.M, self.dtype)
        self.host.ALPHA = host_array((1,), self.dtype)
        self.host.BETA = host_array((1,), self.dtype)

    def host_version(self):
        alpha_val = self.host.ALPHA[0]
        beta_val = self.host.BETA[0]
        self.host.C = alpha_val * self.host.A + beta_val * self.host.B


class TA(UPMEMWorkload):
    def __init__(self, **kwargs):
        super().__init__(
            profile="ta",
            required=dict(),
            symbols=["A", "B", "C"],
            output_symbol="C",
            **kwargs,
        )

    def fetch_data(self):
        self.host.A = host_array((self.M, self.N, self.K), self.dtype)
        self.host.B = host_array((self.M, self.N, self.K), self.dtype)

    def host_version(self):
        self.host.C = self.host.A + self.host.B


class DOT(UPMEMWorkload):
    def __init__(self, **kwargs):
        super().__init__(
            profile="reduction",
            required=dict(),
            symbols=["A", "B"],
            output_symbol="B",
            **kwargs,
        )

    def fetch_data(self):
        self.host.A = host_array((self.M, 1), self.dtype, new=True)
        self.host.B = host_array((1,), self.dtype, new=True)

    def host_version(self):
        self.host.B = np.sum(self.host.A)


class RED(UPMEMWorkload):
    def __init__(self, **kwargs):
        super().__init__(
            profile="reduction",
            required=dict(),
            symbols=["A", "B"],
            output_symbol="B",
            **kwargs,
        )

    def fetch_data(self):
        self.host.A = host_array((self.M,), self.dtype, new=True)
        self.host.B = host_array((1,), self.dtype, new=True)

    def h2d(self):
        self.dev.A = tvm.nd.array(self.host.A, self.target_device, symbol="A")
        self.dev.B = tvm.nd.empty((1,), self.dtype, self.target_device)

    def host_version(self):
        self.host.B = np.sum(self.host.A)

    def benchmark_command(self, config):
        bl = int(math.log2(config["n_c"] * np.dtype(config["dtype"]).itemsize))
        pbtype = config["dtype"].upper()
        return f"make clean && NR_DPUS={config['n_b']} \
            NR_TASKLETS={config['n_t']} TYPE={pbtype} BL={bl} VERSION=HANDSHAKE make >/dev/null 2>/dev/null && \
            ./bin/host_code -i {config['L']} -w {self.warmup} -e {self.repeat}"

class MTV(UPMEMWorkload):
    def __init__(self, **kwargs):
        required = dict()
        super().__init__(
            profile="mtv", required=required, symbols=["A", "B", "C"], output_symbol="C", **kwargs
        )

    def fetch_data(self):
        self.host.A = host_array((self.M, self.K), self.dtype)
        self.host.B = host_array((self.K,), self.dtype)

    def host_version(self):
        self.host.C = np.dot(self.host.A, self.host.B)

    def benchmark_command(self, config):
        bl = int(math.log2(config["n_cache"] * np.dtype(config["dtype"]).itemsize))
        pbtype = config["dtype"].upper()
        return f"make clean && NR_DPUS={config['n_xb'] * config['n_yb']} \
            NR_TASKLETS={config['n_yt']} TYPE={pbtype} BL={bl} make && \
            ./bin/gemv_host -m {config['M']} -n {config['K']} -w {self.warmup} -e {self.repeat}"



class GEMV(UPMEMWorkload):
    def __init__(self, **kwargs):
        required = dict()
        super().__init__(
            profile="poly_gemv1",
            required=required,
            symbols=["A", "B", "C", "ALPHA"],
            output_symbol="C",
            **kwargs,
        )

    def fetch_data(self):
        self.host.A = host_array((self.M, self.K), self.dtype)
        self.host.B = host_array((self.K,), self.dtype)
        self.host.ALPHA = host_array((1,), self.dtype)

    def host_version(self):
        alpha_val = self.host.ALPHA[0]
        self.host.C = alpha_val * np.dot(self.host.A, self.host.B)


class TTV(UPMEMWorkload):
    def __init__(self, **kwargs):
        required = dict()
        super().__init__(
            profile="ttv", required=required, symbols=["A", "B", "C"], output_symbol="C", **kwargs
        )

    def fetch_data(self):
        self.host.A = host_array((self.M, self.N, self.K), self.dtype)
        self.host.B = host_array((self.K,), self.dtype)

    def host_version(self):
        self.host.C = np.einsum("mnk,k->mn", self.host.A, self.host.B)

    def benchmark_command(self, config):
        bl = int(math.log2(config["n_cache"] * np.dtype(config["dtype"]).itemsize))
        pbtype = config["dtype"].upper()
        return f"make clean && NR_DPUS={config['n_xb'] * config['n_yb']} \
            NR_TASKLETS={config['n_t']} TYPE={pbtype} BL={bl} make && \
            ./bin/mtv_host -m {config['M']} -n {config['K']} -w {self.warmup} -e {self.repeat}"


class MMTV(UPMEMWorkload):
    def __init__(self, **kwargs):
        required = dict()
        super().__init__(
            profile="mmtv", required=required, symbols=["A", "B", "C"], output_symbol="C", **kwargs
        )

    def fetch_data(self):
        self.host.A = host_array((self.M, self.N, self.K), self.dtype)
        self.host.B = host_array((self.M, self.K), self.dtype)

    def host_version(self):
        self.host.C = np.einsum("mnk,mk->mn", self.host.A, self.host.B)

    def benchmark_command(self, config):
        bl = int(math.log2(config["n_cache"] * np.dtype(config["dtype"]).itemsize))
        pbtype = config["dtype"].upper()
        return f"""
            make clean &&
            NR_DPUS_Y={config["n_yb"]} \
            NR_DPUS_B={config["n_bb"]} \
            NR_TASKLETS={config["n_yt"]} \
            BL={bl} TYPE={pbtype} make &&
            ./bin/gemv_host -b {config["B"]} \
                -m {config["M"]} \
                -n {config["N"]} \
                -w {self.warmup} \
                -e {self.repeat}
        """