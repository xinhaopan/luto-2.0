# app.py
# 本机启动一个 Dash 网页：通过 Paramiko 持续 tail -F 远端日志，前端折线图动态刷新
# 依赖：pip install dash plotly paramiko pandas
import os.path
import threading
import time
import atexit
from collections import deque
from datetime import datetime
import re

import paramiko
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input
from tools.ssh_config import ssh_config

# ============ 可调参数 ==========================
REFRESH_MS = 1000  # 网页刷新间隔（毫秒）
RECONNECT_WAIT = 2.0  # 断线重连间隔（秒）
MAX_POINTS = 20000  # 内存中最多保留的点数
LINE_NAME = "Memory"  # 曲线名称
# ===============================================

def _load_pkey(path, passphrase):
    last_err = None
    for KeyCls in (paramiko.RSAKey, paramiko.Ed25519Key):
        try:
            return KeyCls.from_private_key_file(path, password=passphrase)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"加载私钥失败：{last_err}")


class TailFollower:
    """在后台线程中通过 SSH 持续 tail -F 远端文件，把 (ts, value) 放进 deque。"""

    def __init__(self):
        self._stop = threading.Event()
        self.data = deque(maxlen=MAX_POINTS)  # 元素：(datetime, float)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            client = None
            try:
                pkey = _load_pkey(PRIVATE_KEY_PATH, PRIVATE_KEY_PASSPHRASE)
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(
                    SSH_HOST,
                    port=SSH_PORT,
                    username=SSH_USER,
                    pkey=pkey,
                    look_for_keys=False,
                    allow_agent=False,
                    timeout=20,
                    banner_timeout=30,
                )
                client.get_transport().set_keepalive(15)
                # get_pty=True 让 tail -F 按行输出更稳定
                cmd = f"tail -n +1 -F {REMOTE_FILE}"
                stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)

                # 逐行读取
                while not self._stop.is_set():
                    line = stdout.readline()
                    if not line:
                        # 可能断线/命令结束
                        break
                    ts, val = self._parse_line(line)
                    if ts is not None:
                        self.data.append((ts, val))
            except Exception:
                # 静默重连（需要看详细原因可打印 traceback）
                pass
            finally:
                try:
                    client and client.close()
                except Exception:
                    pass
                # 断线/异常后稍等再重连
                if not self._stop.is_set():
                    time.sleep(RECONNECT_WAIT)

    @staticmethod
    def _parse_line(line: str):
        # 调试：看到真实字符（确认是否是 \t 还是空格）
        # print("raw line:", repr(line))

        # 先按 TAB 分；不行再按连续空白分
        parts = line.rstrip("\n").split("\t", 1)
        if len(parts) < 2:
            parts = re.split(r"\s+", line.strip(), maxsplit=1)
            if len(parts) < 2:
                return None, None  # 连两列都没解析出来

        ts_s, v_s = parts[0].strip(), parts[1].strip()

        # 严格按你给的格式解析：YYYY-mm-dd HH:MM:SS
        try:
            ts = datetime.strptime(ts_s, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None, None

        try:
            val = float(v_s)
        except ValueError:
            return None, None

        return ts, val

    def to_dataframe(self) -> pd.DataFrame:
        if not self.data:
            return pd.DataFrame(columns=["time", "value"])
        t, v = zip(*self.data)
        return pd.DataFrame({"time": list(t), "value": list(v)})


# 在本机启动网页服务
# 访问 http://127.0.0.1:8050
# ============ 必填：连接 & 文件配置 ============
platform = "HPC"  # "HPC" 或 "NCI"
mem_file = "20250922_Paper2_Results_NCI"
mem_path = f"{mem_file}/carbon_price/mem_log.txt"

cfg = ssh_config(platform)
SSH_HOST = cfg["linux_host"]
SSH_PORT = cfg["linux_port"]
SSH_USER = cfg["linux_username"]
PRIVATE_KEY_PATH = cfg["private_key_path"]
OUTPUT_DIR = cfg["project_dir"]
PRIVATE_KEY_PASSPHRASE = None  # 如有口令填这里，否则 None
REMOTE_FILE = f"{OUTPUT_DIR}/{mem_path}"
SEP = "\t"  # 日志分隔符（你的示例是 TAB）

# 启动后台 tail 线程
tail = TailFollower()
tail.start()
atexit.register(tail.stop)

# Dash 应用
app = Dash(__name__)
app.title = f"Live Memory Monitor in {platform}"

app.layout = html.Div(
    style={"maxWidth": "1000px", "margin": "20px auto", "fontFamily": "Arial, sans-serif"},
    children=[
        html.H3(f"Live Memory Monitor in {platform} {mem_file} (Remote via SSH)"),
        html.Div(id="max-label", style={"fontWeight": "bold", "marginBottom": "8px"}),
        dcc.Graph(id="mem-graph", config={"displayModeBar": True}),
        dcc.Interval(id="tick", interval=REFRESH_MS, n_intervals=0),
        html.Div(
            "显示的是远端文件的实时数据，左上角显示历史最大值。",
            style={"color": "#666", "marginTop": "6px"},
        ),
    ],
)

@app.callback(
    Output("mem-graph", "figure"),
    Output("max-label", "children"),
    Input("tick", "n_intervals"),
)
def update_graph(_):
    df = tail.to_dataframe()
    # 调试：看看有没有数据进来
    # print("rows:", len(df), df.tail(3))

    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", xaxis_title="Time", yaxis_title="GB")
        return fig, "Max: -"

    idxmax = df["value"].idxmax()
    max_val = df.loc[idxmax, "value"]
    max_ts = df.loc[idxmax, "time"]

    fig = go.Figure(go.Scatter(x=df["time"], y=df["value"], mode="lines", name="Memory"))
    fig.update_layout(template="plotly_white", xaxis_title="Time", yaxis_title="GB")
    fig.update_xaxes(tickformat="%Y-%m-%d %H:%M:%S", showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig, f"Max: {max_val:.3f} @ {max_ts.strftime('%Y-%m-%d %H:%M:%S')}"

app.run_server(debug=False, host="127.0.0.1", port=8050)
