@echo off
for /D %%i in (*) do (dir %%i /b > %%i.txt)