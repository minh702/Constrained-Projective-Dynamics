@echo off 
for /r %a in (./ConfigForData/*.txt) do GenPD.exe ./ConfigForData/%~nxa