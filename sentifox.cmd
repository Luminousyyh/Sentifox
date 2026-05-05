@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
C:\Users\Caesar\miniconda3\python.exe "%~dp0\cli.py" %*
