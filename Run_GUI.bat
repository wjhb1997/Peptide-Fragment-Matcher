Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.
@echo off
cd /d "%~dp0"
py -3 peptide_fragment_matcher_gui.py
if errorlevel 1 pause

