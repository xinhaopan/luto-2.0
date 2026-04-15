"""
从NCI/HPC远程服务器下载文件到本地。

传输模式（transfer_mode）:
  "full"   — 完整下载整个任务文件夹，并删除远程副本（rsync 或 TAR 两种方式）
  "report" — 只下载 DATA_REPORT 报告目录，不删除远程
  "lz4"    — 只下载 Data_RES*.lz4 文件（并行 SFTP），不删除远程

使用方法:
    python 1_rsync_and_delete.py
"""

from tools.transfer_helper import rsync_and_delete

if __name__ == "__main__":
    rsync_and_delete(
        platform       = "NCI",                          # "NCI" 或 "HPC"
        task_name      = "20260414_paper4_NCI",   # 远程 output 目录下的文件夹名
        local_base_dir = "../../output",                 # 本地下载基础目录
        transfer_mode  = "full",   # "full" | "report" | "lz4"
        # ---- full 模式专用 ----
        use_tar_mode        = False,  # True=先打包再传输，False=rsync直传
        tar_use_compression = False,  # TAR模式：是否gzip压缩（NetCDF推荐False）
        confirm_delete      = False,  # True=下载完后直接删除远程，False=不删除远程
        debug_mode          = False,  # True=仅测试SSH连接
        # ---- lz4 模式专用 ----
        lz4_n_jobs     = 30,     # 并行下载线程数（仅 lz4 模式）
        log_file       = "rsync_download.log",
    )
