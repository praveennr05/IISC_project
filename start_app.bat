@echo off
title CardioGuard AI - Web Server
echo =====================================================================
echo  Starting CardioGuard AI Web Application on http://127.0.0.1:8000
echo =====================================================================
echo.

cd /d "C:\Users\HP\Desktop\iisc_project"

REM Free port 8000 if previously occupied
powershell -Command "$conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue; if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo Launching CardioGuard AI backend...
echo Open in browser: http://127.0.0.1:8000
echo.
python question_b/app.py

pause
