# Multi-models-in-Tensorflow-Serving-using-Windows10-With-Docker
使用Docker在Windows 10中配置多模型的Tensorflow Serving服务

<br />
在 model_config_list配置文件编写时候，必须注意：
<br />
base_path是 相对路径，尤其是windows环境里面
<br />
比如 base_path:"/models/multiModel/tfServing_HyperSpectral"，这里不是模型文件的绝对路径，而是
<br />
#docker run -p 8501:8501 --mount type=bind,source="E:/tmp/,target=/models/multiModel" -t tensorflow/serving --model_config_file="/models/multiModel/models.config"
<br />
中 target=/models/multiModel 定义的位置的相对路径。
<br />
同时 --model_config_file的文件位置也是相对路径，/models/multiModel/models.config
<br />
否则会报错
<br />
Failed to start server. Error: Not found: E:/tmp/models.config; No such file or directory
<br />
Failed to start server. Error: Invalid argument: Expected model tfserving to have an absolute path or URI; got base_path()=E:/tmp/tfserving
