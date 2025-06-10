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

取消所有任务
```
qselect | xargs qdel
```


Res = 15, timeseries, 8GB,2CPU, 2h

Res = 5, 48GB, 12CPU, 15h for RUN



