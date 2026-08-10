# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import subprocess
import sys


def gpu_timer(closure, log_timings=True):
    """Helper to time gpu-time to execute closure()"""
    import torch

    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.0
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name=None, force=False):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=force)
    return logging.getLogger(name=name)


class CSVLogger(object):

    def __init__(self, fname, *argv, **kwargs):
        self.fname = fname
        self.types = []
        self.headers = []
        mode = kwargs.get("mode", "+a")
        self.delim = kwargs.get("delim", ",")
        self.comments = kwargs.get("comments", [])  # 支持注释行
        # Ensure parent directory exists so distributed workers can open per-rank logs safely.
        parent_dir = os.path.dirname(os.path.abspath(self.fname))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        # -- print headers
        with open(self.fname, mode) as f:
            # 写入注释行
            for comment in self.comments:
                print(f"# {comment}", file=f)
            # 写入表头
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                self.headers.append(v[1])
                if i < len(argv):
                    print(v[1], end=self.delim, file=f)
                else:
                    print(v[1], end="\n", file=f)

    def log(self, *argv, add_separator=False):
        """记录一行数据

        Args:
            *argv: 要记录的值
            add_separator: 是否在记录后添加空行分隔符
        """
        import math

        with open(self.fname, "+a") as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = self.delim if i < len(argv) else "\n"
                val = tv[1]
                # 处理nan值，显示为"-"
                if isinstance(val, float) and math.isnan(val):
                    print("-", end=end, file=f)
                else:
                    print(tv[0] % val, end=end, file=f)
            # 添加空行分隔符
            if add_separator:
                print("", file=f)


class TableLogger(object):
    """带边框的表格日志记录器，使用PrettyTable生成对齐的表格"""

    def __init__(self, fname, *argv, **kwargs):
        self.fname = fname
        self.types = [v[0] for v in argv]
        self.headers = [v[1] for v in argv]
        self.comments = kwargs.get("comments", [])
        mode = kwargs.get("mode", "+a")
        self._pending_rows = []  # 缓存当前组的行

        # Ensure parent directory exists so distributed workers can open per-rank logs safely.
        parent_dir = os.path.dirname(os.path.abspath(self.fname))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(self.fname, mode) as f:
            for comment in self.comments:
                f.write(f"# {comment}\n")

    def log(self, *argv, add_separator=False):
        """记录一行数据

        Args:
            *argv: 要记录的值
            add_separator: 是否在记录后输出表格并添加空行分隔符
        """
        import math

        row = []
        for fmt, val in zip(self.types, argv):
            if isinstance(val, float) and math.isnan(val):
                row.append("-")
            else:
                row.append(fmt % val)
        self._pending_rows.append(row)

        if add_separator:
            self._flush_table()

    def _flush_table(self):
        """输出缓存的行作为带边框的表格"""
        from prettytable import PrettyTable

        table = PrettyTable()
        table.field_names = self.headers
        table.align = "r"  # 默认右对齐
        table.align["type"] = "l"  # type列左对齐
        for row in self._pending_rows:
            table.add_row(row)

        with open(self.fname, "+a") as f:
            f.write(str(table) + "\n\n")
        self._pending_rows = []


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float("-inf")
        self.min = float("inf")
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def jepa_rootpath():
    this_file = os.path.abspath(__file__)
    return "/".join(this_file.split("/")[:-3])


def git_information():
    jepa_root = jepa_rootpath()
    try:
        resp = (
            subprocess.check_output(["git", "-C", jepa_root, "rev-parse", "HEAD", "--abbrev-ref", "HEAD"])
            .decode("ascii")
            .strip()
        )
        commit, branch = resp.split("\n")
        return f"branch: {branch}\ncommit: {commit}\n"
    except Exception:
        return "unknown"
