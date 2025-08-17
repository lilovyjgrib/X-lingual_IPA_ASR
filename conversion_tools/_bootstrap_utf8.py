# vibe-coded fix of the mistake made by PanPhon devs
import builtins


def install():
    _orig_open = builtins.open

    def open_utf8(file, mode="r", *args, **kwargs):
        # Only text mode; don't override if user explicitly set encoding
        if "b" not in mode and "encoding" not in kwargs:
            kwargs["encoding"] = "utf-8"
        return _orig_open(file, mode, *args, **kwargs)

    builtins.open = open_utf8
