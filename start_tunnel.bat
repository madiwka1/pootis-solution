@echo off
cd /d %~dp0
set CF=%~dp0cloudflared.exe
if not exist "%CF%" set CF=cloudflared
"%CF%" tunnel --url http://localhost:8000
pause
