切换到常用的目录
```
cd /home/remote/s222552331/LUTO2_XH/LUTO2
```
```
rm -rf output/20250812*/
```
查询作业状态
```
squeue --noheader --format="%.18i %.50j %.8u %.2t %.10M %.6D %R"
```
取消任务
```
scancel <job_id>
```
取消所有任务
```
squeue -u $USER -h -o "%A" | xargs scancel
```
查看节点状态
```
sinfo -N -l
```
查询存储使用情况
```
scontrol show node | grep -E "NodeName=|RealMemory=|AllocMem=|FreeMem="
```
```angular2html
scontrol show node | awk '
/NodeName=/ {
    if (node != "") {
        printf "%-20s %10.1f GB %10.1f GB %10.1f GB\n", node, real/1024, alloc/1024, free/1024
    }
    for(i=1;i<=NF;i++) if($i ~ /NodeName=/) node=substr($i,10)
}
/RealMemory=/ {
    for(i=1;i<=NF;i++) if($i ~ /RealMemory=/) real=substr($i,12)
}
/AllocMem=/ {
    for(i=1;i<=NF;i++) if($i ~ /AllocMem=/) alloc=substr($i,10)
}
/FreeMem=/ {
    for(i=1;i<=NF;i++) if($i ~ /FreeMem=/) free=substr($i,9)
}
END {
    if (node != "") {
        printf "%-20s %10.1f GB %10.1f GB %10.1f GB\n", node, real/1024, alloc/1024, free/1024
    }
}
'
```
```angular2html
git fetch origin

git reset --hard origin/master

git reset --hard origin/paper2
```

Res = 15, timeseries, 8GB,2CPU, 2h

Res = 5, 35GB, 15h


