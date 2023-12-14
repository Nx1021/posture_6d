# myplugin.py
import mkdocs
from mkdocs.plugins import BasePlugin


class MyPluginConfig(mkdocs.config.base.Config):
    foo = mkdocs.config.config_options.Type(str, default='a default value')
    bar = mkdocs.config.config_options.Type(int, default=0)
    baz = mkdocs.config.config_options.Type(bool, default=True)

class MyPlugin(mkdocs.plugins.BasePlugin[MyPluginConfig]):
    def on_post_build(self, config, **kwargs):
        # 在文档生成完成后执行的操作
        print("文档生成完成！可以执行其他操作了。")
        # 你可以在这里调用你想要执行的任何代码