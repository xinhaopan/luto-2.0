Set fso = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")

scriptPath = fso.GetParentFolderName(WScript.ScriptFullName)

bat1Dir = scriptPath & "\from_HPC"
bat2Dir = scriptPath & "\from_NCI"

bat1 = Chr(34) & bat1Dir & "\run_in_windows.bat" & Chr(34)
bat2 = Chr(34) & bat2Dir & "\run_in_windows.bat" & Chr(34)

WshShell.CurrentDirectory = bat1Dir
WshShell.Run bat1, 0, False

WshShell.CurrentDirectory = bat2Dir
WshShell.Run bat2, 0, False
