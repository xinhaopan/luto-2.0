from pathlib import Path

def delete_luto2_diff_files(root_dir, dry_run=True):
    """
    递归删除文件名包含以下任一子串的文件：
      - 'xr_biodiversity_GBF2_priority_ag_diff'
      - 'xr_GHG_ag_diff'
    dry_run=True 时仅打印将删除的文件，确认后设为 False 执行删除。
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    pat1 = "*xr_biodiversity_GBF2_priority_ag_diff*"
    pat2 = "*xr_GHG_ag_diff*"
    targets = {p for p in root.rglob(pat1) if p.is_file()} | \
              {p for p in root.rglob(pat2) if p.is_file()}

    if not targets:
        print("No matching files found.")
        return

    for p in sorted(targets):
        if dry_run:
            print(f"[DRY-RUN] Would delete: {p}")
        else:
            try:
                p.unlink()
                print(f"Deleted: {p}")
            except Exception as e:
                print(f"Failed to delete {p}: {e}")

    print(f"Total {'to delete' if dry_run else 'deleted'}: {len(targets)}")

# 用法：
# 先演练
# delete_luto2_diff_files(r"D:\your\folder", dry_run=True)
# 确认后删除
delete_luto2_diff_files("output/20250926_Paper2_Results_HPC/carbon_price", dry_run=False)
