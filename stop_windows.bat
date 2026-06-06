@echo off
REM ─────────────────────────────────────────────────────────────────
REM  stop_windows.bat
REM  Double-click to safely stop all containers (data is preserved)
REM ─────────────────────────────────────────────────────────────────

echo.
echo  Stopping Churn Prediction stack...
echo.

docker compose down

echo.
echo  ✅ All containers stopped. Your data is preserved.
echo.
echo  To start again: double-click start_windows.bat
echo.
pause
