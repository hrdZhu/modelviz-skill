"""Schemas used by the model visualization skill."""

from .requirement_schema import PlotRequirement

__all__ = ["PlotRequirement"]

"""
__init__.py 是 Python 包的“入口文件”，负责初始化包、整理导出接口。
你这份代码的作用就是把 PlotRequirement 暴露成 schemas.PlotRequirement，
让其他模块更方便调用。
"""
