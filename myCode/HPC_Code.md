切换到常用的目录
```
cd /home/remote/s222552331/LUTO_XH/LUTO2/output
```
查询作业状态
```
squeue -u $USER
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


Res = 15, timeseries, 8GB,2CPU, 2h

Res = 5, 48GB, 12CPU, 15h for RUN



