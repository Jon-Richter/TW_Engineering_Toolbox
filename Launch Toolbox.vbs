Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")
scriptPath = WScript.ScriptFullName
scriptDir = FSO.GetParentFolderName(scriptPath)
logDir = scriptDir & "\logs"
If Not FSO.FolderExists(logDir) Then FSO.CreateFolder logDir
logFile = logDir & "\launch_vbs.log"
Function timestamp()
  timestamp = Year(Now) & "-" & Right("0" & Month(Now),2) & "-" & Right("0" & Day(Now),2) & " " & _
              Right("0" & Hour(Now),2) & ":" & Right("0" & Minute(Now),2) & ":" & Right("0" & Second(Now),2)
End Function
Sub Log(msg)
  On Error Resume Next
  Set f = FSO.OpenTextFile(logFile, 8, True)
  f.WriteLine timestamp() & " - " & msg
  f.Close
End Sub
Log "Launch request."
Dim commands()
ReDim commands(6)
commands(0) = Chr(34) & scriptDir & "\.venv\Scripts\pythonw.exe" & Chr(34) & " -m toolbox_app"
commands(1) = Chr(34) & scriptDir & "\.venv\Scripts\python.exe" & Chr(34) & " -m toolbox_app"
commands(2) = "pyw -3.13 -m toolbox_app"
commands(3) = "pyw -m toolbox_app"
commands(4) = "pythonw -m toolbox_app"
commands(5) = "python -m toolbox_app"
commands(6) = "cmd /c " & Chr(34) & scriptDir & "\scripts\launch_toolbox.bat" & Chr(34)
launched = False
For i = 0 To UBound(commands)
  cmd = commands(i)
  Log "Trying: " & cmd
  On Error Resume Next
  wnd = 0
  If LCase(Left(Trim(cmd),4)) = "cmd " Then wnd = 1
  ret = WshShell.Run(cmd, wnd, True)
  errnum = Err.Number
  If errnum <> 0 Then
    Log "Launch failed: " & Err.Description
    Err.Clear
  ElseIf ret <> 0 Then
    Log "Launch failed: return code " & ret
  Else
    Log "Launch succeeded."
    launched = True
    Exit For
  End If
Next
If Not launched Then
  Log "All launch attempts failed."
  WshShell.Popup "Engineering Toolbox failed to start. See: " & logFile, 10, "Launch failed", 16
End If
