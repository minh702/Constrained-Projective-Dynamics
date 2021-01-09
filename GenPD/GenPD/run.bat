@echo off

for /r %a in (./test/*.txt) do GenPD.exe %~nxa