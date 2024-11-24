# 0. 通过.ssh使得本地电脑和NCI之间可以免密互相登录
# 1. 在window上建立反向隧道，连接本地电脑与NCI
代码在"C:\Program Files (x86)\ICW\ssh_tunnel\ssh_tunnel.bat"
注意每次建立的反向隧道可能会通向不同的NCI机器，因此在建立隧道时候需要记录连接的节点
每到整点监测通道是否还在，如果不在就再次建立
# 2. 在NCI上建立同步的脚本
脚本在/g/data/jk53/LUTO_XH/LUTO2/myCode/tasks_run/rsync/sync_files.sh
脚本定时调用在每小时的第2分钟执行
注意这个脚本包含，使用第1步中连接到的节点
如果没有挂载N盘就挂载一下
只同步更新的文件
# 3. 后台定时运行方法
使用nohup 可以忽略挂起信号 (SIGHUP)，让进程在你关闭 SSH 连接后仍然运行。
运行脚本：
```
nohup python3 Scheduled_sh.py > output.log 2>&1 &
```

nohup：忽略挂起信号。
> output.log 2>&1：将输出和错误日志重定向到 output.log 文件中。
确保任务持久化： 即使关闭 SSH 连接，任务仍然会继续运行。

关闭方式：
```
ps -ef | grep python
```
```
kill -9 <ID>
```