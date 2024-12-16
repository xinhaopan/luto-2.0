切换到常用的目录
```
cd /g/data/jk53/LUTO_XH/LUTO2/output
```
查询存储配额使用情况
```
lquota
```
查询作业状态
```
qstat
```
取消任务
```
qdel <job_id>
```
查看项目配额和使用情况
```
nci_account -P jk53 
```
查看home内存配额
```angular2html
quota -s
```

Res = 10, snapshot, 40GB, 0.5h
Res = 10, timeseries, 150GB, 3.5h
Res = 5, timeSeries, 40GB, 15h