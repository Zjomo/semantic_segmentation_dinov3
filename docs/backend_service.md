# Flask 后端与 SQLite 数据库

## 启动

在项目根目录安装依赖并启动：

```powershell
pip install -r requirements.txt
python -m backend
```

默认监听 `127.0.0.1:5000`，浏览器打开 `http://127.0.0.1:5000/`。首次启动会创建 `instance/coresegment.db`，并索引 `scan_output/ToupView/test4` 和 `scan_output/Map` 两个内置样例。服务不会启动分割推理或调用显微镜运动控制。

可通过环境变量调整：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `CORESG_HOST` | `127.0.0.1` | Flask 监听地址 |
| `CORESG_PORT` | `5000` | Flask 端口 |
| `CORESG_DEBUG` | `0` | 开发调试模式；仅本机开发时设为 `1` |
| `CORESG_DATABASE` | `instance/coresegment.db` | SQLite 文件位置 |
| `CORESG_SCAN_ROOT` | `scan_output` | 扫描样例根目录 |

## 数据模型

- `datasets` 保存样例 ID、显示名称、磁盘目录和路径类型。
- `artifacts` 保存底图、掩码、骨架图和坐标 JSON 的文件名、媒体类型及文件信息。
- `motion_records` 按样例与序号保存扫描方向和步数，支持 SQLite 查询及 UI 展示。

坐标数组和大图保留在扫描目录中，SQLite 不复制这些大文件；数据库只记录受控样例目录下的文件清单。应用启动时会根据内置样例目录同步清单与动作记录。

## HTTP 接口

| 方法与路径 | 说明 |
| --- | --- |
| `GET /api/health` | 服务与 SQLite 健康状态 |
| `GET /api/datasets` | 返回可用样例及其产物 URL |
| `GET /api/datasets/{id}/motion` | 返回扫描动作和骨架统计 |
| `GET /api/datasets/{id}/artifacts/{kind}` | 读取已登记的底图、掩码、骨架图或坐标 JSON |

接口与页面同源，不需要配置跨域。未知样例或未登记产物返回 JSON `404`。
